import streamlit as st
import pandas as pd
import unicodedata
from openai import OpenAI

# ── OpenAI API key ─────────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shimaore ↔ French Translator",
    page_icon="🌊",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&family=Source+Sans+3:wght@300;400;600&display=swap');

    /* Force light mode */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="block-container"] {
        background-color: #ffffff !important;
        color: #1a3a5c !important;
        font-family: 'Source Sans 3', sans-serif !important;
    }

    /* Hide Streamlit chrome */
    [data-testid="stHeader"] { display: none !important; }
    #MainMenu, footer, header { visibility: hidden; }

    /* Typography */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #1a3a5c !important;
        margin-bottom: 0 !important;
    }
    .subtitle {
        color: #5a7fa0;
        font-size: 13px;
        margin-top: 2px;
        margin-bottom: 0;
    }
    p, label, div { color: #1a3a5c; }

    /* Text areas */
    .stTextArea textarea {
        background-color: #f5f9ff !important;
        color: #1a3a5c !important;
        border: 1.5px solid #c0d4ea !important;
        border-radius: 10px !important;
        font-size: 15px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        resize: none;
    }

    /* Radio */
    div[data-testid="stRadio"] label { color: #1a3a5c !important; }
    div[data-testid="stRadio"] > div { flex-direction: row; gap: 1.5rem; flex-wrap: wrap; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #1a3a5c, #2e6da4) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
        width: 100% !important;
        letter-spacing: 0.3px;
        margin-top: 6px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* Result boxes */
    .result-box {
        background: #eef5ff;
        border-left: 4px solid #2e6da4;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 16px;
        color: #1a3a5c !important;
        line-height: 1.65;
        min-height: 110px;
    }
    .result-box.exact {
        border-left-color: #1e9e6b;
        background: #e8faf3;
        color: #0f3d28 !important;
    }

    /* Divider */
    hr { border-color: #d0dde8 !important; margin: 0.6rem 0; }

    /* Alert boxes */
    [data-testid="stAlert"] { background: #fff8e6 !important; border-radius: 8px !important; }

    /* ── DESKTOP: side-by-side, scrollable if zoomed ── */
    @media (min-width: 768px) {
        [data-testid="stMain"] {
            padding-top: 1.2rem;
            padding-bottom: 1rem;
        }
        [data-testid="block-container"] {
            padding-top: 0;
            padding-bottom: 1rem;
        }
    }

    /* ── MOBILE: stacked, scrollable, pinned to top ── */
    @media (max-width: 767px) {
        html, body { margin: 0; padding: 0; }
        [data-testid="stMain"] {
            padding-top: 0.3rem !important;
            padding-bottom: 1rem;
        }
        [data-testid="block-container"] {
            padding-top: 0.3rem !important;
        }
        .main .block-container { padding-top: 0.3rem !important; }
        section[data-testid="stMain"] > div:first-child { padding-top: 0 !important; }
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    """Lowercase and strip diacritics so accented chars match plain ones."""
    return unicodedata.normalize("NFD", text.strip().lower()).encode("ascii", "ignore").decode("ascii")


@st.cache_data(show_spinner="Loading dataset…")
def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.iloc[:, :2]
    df.columns = ["shimaore", "french"]
    df["shimaore_norm"] = df["shimaore"].apply(normalize)
    df["french_norm"]   = df["french"].apply(normalize)
    examples = "\n".join(
        f'Shimaore: {row["shimaore"]} -> French: {row["french"]}'
        for _, row in df.iterrows()
    )
    return df, examples


def exact_match(text: str, direction: str, df: pd.DataFrame):
    key = normalize(text)
    if direction == "Shimaore → French":
        row = df[df["shimaore_norm"] == key]
        if not row.empty:
            return row.iloc[0]["french"]
    else:
        row = df[df["french_norm"] == key]
        if not row.empty:
            return row.iloc[0]["shimaore"]
    return None


def translate_with_ai(text: str, direction: str, examples: str) -> str:
    instruction = (
        "Translate the following Shimaore sentence into French."
        if direction == "Shimaore → French"
        else "Translate the following French sentence into Shimaore."
    )
    prompt = f"""You are a translation assistant specializing in Shimaore and French.

Below is the COMPLETE translation dataset between Shimaore and French:

{examples}

IMPORTANT RULES:
1. First, check if the sentence exists EXACTLY in the dataset above.
   - If found: return that EXACT translation, nothing else.
2. If the sentence is NOT in the dataset:
   - {instruction}
   - Aim for natural meaning, preserve sentiment and structure.
3. Output ONLY the translated text. No arrows, no original sentence, no labels, no explanation. Just the translation.
"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"{prompt}\nSentence: {text}"
    )
    return response.output_text.strip()


# ── Load dataset at startup ───────────────────────────────────────────────────
df, examples = load_dataset("shimaore_french_dataset.csv")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🌊 Shimaore ↔ French Translator")
st.markdown('<p class="subtitle">Maore Language Project</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Direction selector ────────────────────────────────────────────────────────
direction = st.radio(
    "direction",
    options=["Shimaore → French", "French → Shimaore"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("---")

# ── Two-column layout (auto-stacks on mobile) ─────────────────────────────────
left_col, _, right_col = st.columns([10, 0.3, 10])

with left_col:
    lang_label  = "Shimaore" if direction == "Shimaore → French" else "French"
    placeholder = "Enter Shimaore text…" if direction == "Shimaore → French" else "Entrez votre texte en français…"
    st.markdown(f"**✏️ {lang_label} — input**")
    user_input = st.text_area("input", placeholder=placeholder, height=200, label_visibility="collapsed")
    translate_clicked = st.button("Translate ↗")

with right_col:
    target_label = "French" if direction == "Shimaore → French" else "Shimaore"
    st.markdown(f"**🌐 {target_label} — translation**")

    if translate_clicked:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to translate.")
        else:
            with st.spinner("Translating…"):
                try:
                    exact = exact_match(user_input.strip(), direction, df)
                    if exact:
                        st.markdown(f'<div class="result-box exact">{exact}</div>', unsafe_allow_html=True)
                    else:
                        result = translate_with_ai(user_input.strip(), direction, examples)
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)
                except FileNotFoundError:
                    st.error("❌ `shimaore_french_dataset.csv` not found. Place it in the same folder as `app.py`.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    else:
        st.markdown(
            '<div class="result-box" style="color:#a0b4c8 !important; font-style:italic;">Translation will appear here…</div>',
            unsafe_allow_html=True,
        )