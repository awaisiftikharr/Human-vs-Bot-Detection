from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bot_detector.pkl"
FEATURE_COLUMNS = ["Distance", "Time", "Jitter"]

st.set_page_config(page_title="Human vs Bot Detector", page_icon="🤖", layout="centered")


@st.cache_resource
def load_model():
    """Load the repository's trained scikit-learn model once per app session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "bot_detector.pkl is missing. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def build_features(distance: float, elapsed_time: float, jitter: float) -> pd.DataFrame:
    """Create model input with the exact feature names used during training."""
    if distance < 0 or elapsed_time <= 0 or jitter < 0:
        raise ValueError("Distance and jitter must be non-negative; time must be positive.")
    return pd.DataFrame(
        [[float(distance), float(elapsed_time), float(jitter)]],
        columns=FEATURE_COLUMNS,
    )


st.title("Human vs. Bot Detection")
st.caption("A behavioural-signal classification demo using a Random Forest model.")
st.info(
    "This is an educational prototype. Predictions depend on the quality and "
    "representativeness of the training data; do not use it as a production "
    "security, identity, or access-control decision."
)

with st.form("prediction_form"):
    st.subheader("Session behaviour")
    distance = st.slider("Pointer movement distance (px)", 0, 1000, 300, step=10)
    elapsed_time = st.slider("Elapsed interaction time (seconds)", 0.1, 60.0, 5.0, step=0.1)
    jitter = st.slider("Movement jitter", 0.0, 10.0, 2.0, step=0.1)
    submitted = st.form_submit_button("Analyse session", type="primary")

if submitted:
    try:
        model = load_model()
        features = build_features(distance, elapsed_time, jitter)
        prediction = int(model.predict(features)[0])

        # The repository's training CSV and original interface use 1 = human, 0 = bot.
        label = "Human-like behaviour" if prediction == 1 else "Bot-like behaviour"
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            class_index = list(model.classes_).index(prediction)
            confidence = float(probabilities[class_index])
            st.metric("Model confidence", f"{confidence:.1%}")
        else:
            st.caption("This model does not expose class probabilities.")

        if prediction == 1:
            st.success(f"Prediction: {label}")
        else:
            st.warning(f"Prediction: {label}")

        st.caption("Class mapping follows the repository's current app convention: 1 = human, 0 = bot.")
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
