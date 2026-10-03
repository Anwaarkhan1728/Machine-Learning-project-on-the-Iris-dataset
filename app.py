"""
🌸 Iris Flower Classifier — Streamlit App
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle, os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🌸 Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        box-shadow: 0 10px 25px rgba(102,126,234,0.4);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 12px;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        transition: 0.2s;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<h1 class="main-title">🌸 Iris Flower Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning powered species prediction</p>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "models/iris_best_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at `{MODEL_PATH}`. Run `python train_model.py` first.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

bundle   = load_model()
model    = bundle["model"]
scaler   = bundle["scaler"]
encoder  = bundle["encoder"]
features = bundle["features"]

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🎛️ Input Features")
st.sidebar.markdown("Adjust the sliders to describe the flower:")

# sensible default ranges
DEFAULTS = {
    "sepal_length": (4.0, 8.0, 5.8),
    "sepal_width":  (2.0, 4.5, 3.0),
    "petal_length": (1.0, 7.0, 3.8),
    "petal_width":  (0.1, 2.5, 1.2),
}

input_vals = {}
for feat in features:
    lo, hi, dv = DEFAULTS.get(feat.lower(), (0.0, 10.0, 5.0))
    input_vals[feat] = st.sidebar.slider(
        feat.replace("_", " ").title(),
        min_value=float(lo), max_value=float(hi),
        value=float(dv), step=0.1,
    )

# ---------------- MODEL INFO ----------------
with st.sidebar.expander("ℹ️ Model Info"):
    st.write(f"**Algorithm:** {bundle['model_name']}")
    st.write(f"**Accuracy:** {bundle['accuracy']*100:.2f}%")
    st.write(f"**Classes:** {list(encoder.classes_)}")

# ---------------- MAIN AREA ----------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Your Input")
    input_df = pd.DataFrame([input_vals]).T
    input_df.columns = ["Value"]
    st.dataframe(input_df, use_container_width=True)

    predict_btn = st.button("🚀 Predict Species", use_container_width=True)

with col2:
    st.markdown("### 🎯 Prediction")
    if predict_btn:
        X = np.array([list(input_vals.values())])
        X_scaled = scaler.transform(X)
        pred_idx = model.predict(X_scaled)[0]
        pred_label = encoder.inverse_transform([pred_idx])[0]
        proba = model.predict_proba(X_scaled)[0]

        st.markdown(f'<div class="result-box">🌼 {pred_label.title()} 🌼</div>',
                    unsafe_allow_html=True)

        st.markdown("#### Confidence per Class")
        prob_df = pd.DataFrame({
            "Species": encoder.classes_,
            "Probability": proba
        }).set_index("Species")
        st.bar_chart(prob_df)

        # flower emoji reference
        emoji_map = {"setosa": "🌷", "versicolor": "🌺", "virginica": "🌸"}
        st.info(f"{emoji_map.get(pred_label.lower(), '🌼')} "
                f"Predicted: **{pred_label}** with "
                f"{proba[pred_idx]*100:.1f}% confidence")
    else:
        st.info("👈 Adjust sliders and click **Predict Species**")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;'>"
    "Built with ❤️ using Streamlit & scikit-learn"
    "</p>",
    unsafe_allow_html=True
)
