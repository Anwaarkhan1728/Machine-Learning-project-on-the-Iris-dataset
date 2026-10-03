"""
🌸 Iris Flower Classifier — Streamlit App (self-healing)
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
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0;
    }
    .subtitle { text-align: center; color: #666; font-size: 1.1rem;
        margin-top: -10px; margin-bottom: 30px; }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px; border-radius: 15px; color: white;
        text-align: center; font-size: 1.8rem; font-weight: bold;
        box-shadow: 0 10px 25px rgba(102,126,234,0.4);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white; font-weight: bold; font-size: 1.1rem;
        padding: 12px; border-radius: 10px; border: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🌸 Iris Flower Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning powered species prediction</p>',
            unsafe_allow_html=True)

# ---------------- AUTO-TRAIN HELPER ----------------
MODEL_PATH = "models/iris_best_model.pkl"

def train_if_missing():
    """Train model if pickle doesn't exist."""
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs("models", exist_ok=True)
    with st.spinner("🧠 Training model for the first time... please wait"):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score

        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        le = LabelEncoder().fit(iris.target_names)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        X_tr, X_va, y_tr, y_va = train_test_split(
            Xs, y, test_size=0.2, random_state=42, stratify=y)

        models = {
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
            "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=42),
            "SVM":                SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
            "KNN":                KNeighborsClassifier(n_neighbors=5),
        }
        best_name, best_model, best_cv, best_acc = None, None, -1, 0
        for name, m in models.items():
            m.fit(X_tr, y_tr)
            cv = cross_val_score(m, Xs, y, cv=5).mean()
            if cv > best_cv:
                best_cv, best_name, best_model = cv, name, m
                best_acc = accuracy_score(y_va, m.predict(X_va))

        best_model.fit(Xs, y)

        bundle = {
            "model":      best_model,
            "scaler":     scaler,
            "encoder":    le,
            "features":   X.columns.tolist(),
            "model_name": best_name,
            "accuracy":   float(best_acc),
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(bundle, f)

@st.cache_resource
def load_bundle():
    train_if_missing()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

bundle   = load_bundle()
model    = bundle["model"]
scaler   = bundle["scaler"]
encoder  = bundle["encoder"]
features = bundle["features"]

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🎛️ Input Features")
st.sidebar.markdown("Adjust the sliders:")

DEFAULTS = {
    "sepal length (cm)": (4.0, 8.0, 5.8),
    "sepal width (cm)":  (2.0, 4.5, 3.0),
    "petal length (cm)": (1.0, 7.0, 3.8),
    "petal width (cm)":  (0.1, 2.5, 1.2),
}

input_vals = {}
for feat in features:
    lo, hi, dv = DEFAULTS.get(feat, (0.0, 10.0, 5.0))
    input_vals[feat] = st.sidebar.slider(
        feat.replace(" (cm)", "").replace("_", " ").title(),
        min_value=float(lo), max_value=float(hi),
        value=float(dv), step=0.1,
    )

with st.sidebar.expander("ℹ️ Model Info"):
    st.write(f"**Algorithm:** {bundle['model_name']}")
    st.write(f"**Accuracy:** {bundle['accuracy']*100:.2f}%")
    st.write(f"**Classes:** {list(encoder.classes_)}")

# ---------------- MAIN ----------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Your Input")
    input_df = pd.DataFrame(input_vals.items(), columns=["Feature", "Value"])
    st.dataframe(input_df, use_container_width=True, hide_index=True)
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

        emoji_map = {"setosa": "🌷", "versicolor": "🌺", "virginica": "🌸"}
        st.info(f"{emoji_map.get(pred_label.lower(), '🌼')} "
                f"Predicted: **{pred_label}** with "
                f"{proba[pred_idx]*100:.1f}% confidence")
    else:
        st.info("👈 Adjust sliders and click **Predict Species**")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#888;'>"
            "Built with ❤️ using Streamlit & scikit-learn</p>",
            unsafe_allow_html=True)
