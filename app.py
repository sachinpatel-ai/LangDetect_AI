"""
AI Language Detector
A Streamlit frontend for a pre-trained NLP language detection model
(CountVectorizer + MultinomialNB).

This app only loads and uses the already-trained model files:
    - language_model.pkl
    - count_vectorizer.pkl

It does NOT retrain or refit anything.
"""

import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
MODEL_PATH = Path("language_model.pkl")
VECTORIZER_PATH = Path("count_vectorizer.pkl")
TEST_ACCURACY = 95.3  # reported accuracy of the trained model, in percent


# --------------------------------------------------------------------------
# Loading (cached so files are read only once per session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and vectorizer...")
def load_model():
    """Load the trained model and vectorizer from disk.

    Returns a tuple (model, vectorizer, error_message).
    If loading fails, model/vectorizer will be None and error_message
    will describe what went wrong.
    """
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        missing = [
            str(p) for p in (MODEL_PATH, VECTORIZER_PATH) if not p.exists()
        ]
        return None, None, f"Missing required file(s): {', '.join(missing)}"

    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except Exception as exc:  # noqa: BLE001 - we want to catch any load error
        return None, None, f"Could not load model files. Details: {exc}"

    # Basic sanity check that the objects look like what we expect
    if not hasattr(model, "predict") or not hasattr(vectorizer, "transform"):
        return None, None, "The loaded files do not look like a valid model/vectorizer pair."

    return model, vectorizer, None


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict_language(text, model, vectorizer):
    """Run the prediction pipeline on a single piece of text.

    Returns a dict with:
        - "language": predicted language label
        - "confidence": confidence percentage for the top prediction (or None)
        - "error": error message if something went wrong, else None
    """
    try:
        vectorized_text = vectorizer.transform([text])
        prediction = model.predict(vectorized_text)[0]

        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vectorized_text)[0]
            classes = model.classes_
            pred_index = list(classes).index(prediction)
            confidence = round(probabilities[pred_index] * 100, 2)

        return {
            "language": prediction,
            "confidence": confidence,
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "language": None,
            "confidence": None,
            "error": f"Something went wrong while predicting. Details: {exc}",
        }


# --------------------------------------------------------------------------
# Theming
# --------------------------------------------------------------------------
def get_theme_colors(theme: str) -> dict:
    """Return a dict of CSS colors for the selected theme."""
    if theme == "dark":
        return {
            "bg_gradient": "linear-gradient(160deg, #060b16 0%, #0a1526 45%, #0d1b30 100%)",
            "glow_a": "rgba(56, 189, 248, 0.18)",
            "glow_b": "rgba(99, 102, 241, 0.16)",
            "ring_color": "rgba(125, 211, 252, 0.10)",
            "card_bg": "rgba(255, 255, 255, 0.05)",
            "card_border": "rgba(255, 255, 255, 0.10)",
            "text": "#f0f2f6",
            "muted_text": "#9aa4b2",
            "accent": "#38bdf8",
            "accent_soft": "rgba(56, 189, 248, 0.15)",
            "success_bg": "rgba(56, 189, 248, 0.10)",
            "success_border": "rgba(56, 189, 248, 0.35)",
            "success_text": "#7dd3fc",
        }
    return {
        "bg_gradient": "linear-gradient(160deg, #eef4ff 0%, #f3f7ff 45%, #eaf1ff 100%)",
        "glow_a": "rgba(56, 130, 246, 0.16)",
        "glow_b": "rgba(99, 102, 241, 0.12)",
        "ring_color": "rgba(59, 130, 246, 0.10)",
        "card_bg": "rgba(255, 255, 255, 0.75)",
        "card_border": "rgba(15, 23, 42, 0.08)",
        "text": "#1a1f2b",
        "muted_text": "#5b6472",
        "accent": "#2563eb",
        "accent_soft": "rgba(37, 99, 235, 0.10)",
        "success_bg": "rgba(37, 99, 235, 0.08)",
        "success_border": "rgba(37, 99, 235, 0.30)",
        "success_text": "#1d4ed8",
    }


def render_background(colors: dict):
    """Render a fixed, decorative tech-style background (soft gradient glow
    with subtle circuit/wave lines), inspired by an AI voice/microphone
    tech visual. Purely decorative and non-interactive.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {colors['bg_gradient']};
            background-attachment: fixed;
        }}

        .bg-tech {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            overflow: hidden;
            z-index: 0;
            pointer-events: none;
        }}

        .bg-glow-a {{
            position: absolute;
            top: -10%;
            left: -10%;
            width: 55vw;
            height: 55vw;
            border-radius: 50%;
            background: radial-gradient(circle, {colors['glow_a']} 0%, transparent 70%);
            filter: blur(10px);
            animation: driftA 18s ease-in-out infinite;
        }}

        .bg-glow-b {{
            position: absolute;
            bottom: -15%;
            right: -10%;
            width: 60vw;
            height: 60vw;
            border-radius: 50%;
            background: radial-gradient(circle, {colors['glow_b']} 0%, transparent 70%);
            filter: blur(10px);
            animation: driftB 22s ease-in-out infinite;
        }}

        @keyframes driftA {{
            0%, 100% {{ transform: translate(0, 0) scale(1); }}
            50% {{ transform: translate(3%, 4%) scale(1.06); }}
        }}

        @keyframes driftB {{
            0%, 100% {{ transform: translate(0, 0) scale(1); }}
            50% {{ transform: translate(-3%, -3%) scale(1.05); }}
        }}

        .bg-rings {{
            position: absolute;
            top: 50%;
            left: 50%;
            width: 900px;
            height: 900px;
            transform: translate(-50%, -50%);
            border-radius: 50%;
            border: 1px solid {colors['ring_color']};
            opacity: 0.5;
        }}

        .bg-rings::before, .bg-rings::after {{
            content: "";
            position: absolute;
            top: 50%; left: 50%;
            border-radius: 50%;
            border: 1px solid {colors['ring_color']};
            transform: translate(-50%, -50%);
        }}

        .bg-rings::before {{ width: 650px; height: 650px; }}
        .bg-rings::after {{ width: 420px; height: 420px; }}

        /* Keep real content above the decorative background */
        .block-container, [data-testid="stSidebar"] {{
            position: relative;
            z-index: 1;
        }}
        </style>
        <div class="bg-tech">
            <div class="bg-glow-a"></div>
            <div class="bg-glow-b"></div>
            <div class="bg-rings"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_theme_css(colors: dict):
    """Apply theme-aware styling for text, cards, and result box."""
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            color: {colors['text']};
        }}

        h1, h2, h3, p, span, label, .stMarkdown {{
            color: {colors['text']} !important;
        }}

        [data-testid="stSidebar"] {{
            background: {colors['card_bg']};
            border-right: 1px solid {colors['card_border']};
        }}

        .stTextArea textarea {{
            background: {colors['card_bg']} !important;
            color: {colors['text']} !important;
            border: 1px solid {colors['card_border']} !important;
            border-radius: 12px !important;
        }}

        div[data-testid="stButton"] button {{
            border-radius: 10px !important;
            border: 1px solid {colors['card_border']} !important;
        }}

        div[data-testid="stButton"] button[kind="primary"] {{
            background: {colors['accent']} !important;
            border: none !important;
        }}

        .result-card {{
            background: {colors['success_bg']};
            border: 1px solid {colors['success_border']};
            border-radius: 16px;
            padding: 28px 32px;
            text-align: center;
            margin-top: 18px;
        }}

        .result-label {{
            font-size: 14px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {colors['muted_text']};
            margin-bottom: 6px;
        }}

        .result-language {{
            font-size: 42px;
            font-weight: 800;
            color: {colors['success_text']};
            line-height: 1.1;
        }}

        .result-confidence {{
            margin-top: 10px;
            font-size: 15px;
            color: {colors['muted_text']};
        }}

        .info-card {{
            background: {colors['card_bg']};
            border: 1px solid {colors['card_border']};
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# UI helper functions
# --------------------------------------------------------------------------
def render_sidebar(model, vectorizer):
    with st.sidebar:
        st.header("⚙️ Settings")
        theme_choice = st.toggle(
            "🌙 Dark mode",
            value=(st.session_state.get("theme", "light") == "dark"),
        )
        st.session_state["theme"] = "dark" if theme_choice else "light"

        st.divider()


def render_history():
    history = st.session_state.get("history", [])
    if not history:
        return

    st.subheader("🕘 Recent Predictions")
    for entry in reversed(history[-5:]):
        conf_text = f" ({entry['confidence']}%)" if entry["confidence"] is not None else ""
        with st.expander(f"{entry['language']}{conf_text} — “{entry['text'][:40]}…”"):
            st.write(f"**Input:** {entry['text']}")
            st.write(f"**Predicted language:** {entry['language']}")
            if entry["confidence"] is not None:
                st.write(f"**Confidence:** {entry['confidence']}%")


def add_to_history(text, language, confidence):
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append(
        {"text": text, "language": language, "confidence": confidence}
    )


def render_result_card(result: dict):
    """Show only the single predicted language, prominently, in a themed card."""
    confidence_html = ""
    if result["confidence"] is not None:
        confidence_html = (
            f'<div class="result-confidence">Model confidence: '
            f'{result["confidence"]}%</div>'
        )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Detected Language</div>
            <div class="result-language">{result['language']}</div>
            {confidence_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="AI Language Detector",
        page_icon="🌐",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Session state defaults
    st.session_state.setdefault("text_input", "")
    st.session_state.setdefault("run_prediction", False)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_text", "")
    st.session_state.setdefault("theme", "light")

    model, vectorizer, load_error = load_model()

    # Sidebar renders first so the theme toggle is applied before the
    # rest of the page is styled.
    render_sidebar(model, vectorizer)

    colors = get_theme_colors(st.session_state["theme"])
    render_background(colors)
    render_theme_css(colors)

    st.title("🌐 AI Language Detector")
    st.write("Enter any text and let the machine-learning model predict its language.")

    if load_error:
        st.error(
            "⚠️ The app could not load the required model files.\n\n"
            f"{load_error}\n\n"
            "Please make sure `language_model.pkl` and `count_vectorizer.pkl` "
            "are placed in the same folder as `app.py`."
        )
        st.stop()

    st.text_area(
        "Enter text to analyze",
        key="text_input",
        height=160,
        placeholder="Type or paste a sentence here...",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        detect_clicked = st.button("🔍 Detect Language", type="primary", use_container_width=True)
    with col2:
        clear_clicked = st.button("🧹 Clear", use_container_width=True)

    if clear_clicked:
        st.session_state["text_input"] = ""
        st.session_state["run_prediction"] = False
        st.session_state["last_result"] = None
        st.session_state["last_text"] = ""
        st.rerun()

    should_predict = detect_clicked or st.session_state.get("run_prediction", False)
    st.session_state["run_prediction"] = False  # reset the flag now that we've read it

    if should_predict:
        text = st.session_state["text_input"].strip()

        if not text:
            st.warning("Please enter some text before detecting the language.")
            st.session_state["last_result"] = None
        else:
            result = predict_language(text, model, vectorizer)

            if result["error"]:
                st.error(f"⚠️ {result['error']}")
                st.session_state["last_result"] = None
            else:
                # Persist the result so it stays visible across reruns
                # (e.g. theme toggle, widget edits) until the next
                # prediction or Clear.
                st.session_state["last_result"] = result
                st.session_state["last_text"] = text
                add_to_history(text, result["language"], result["confidence"])

    # Always render the most recent successful result, if any, so it
    # doesn't disappear on unrelated reruns.
    result = st.session_state.get("last_result")
    if result:
        render_result_card(result)

    st.divider()
    render_history()

    st.markdown(
        f"""
        <div style="text-align:center; margin-top:36px; padding-top:14px;
                    border-top:1px solid {colors['card_border']};
                    color:{colors['muted_text']}; font-size:13px;">
            Developed by <strong>Sachin Patel</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
