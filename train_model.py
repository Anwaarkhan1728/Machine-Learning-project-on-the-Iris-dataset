"""
Iris Flower Classification — Training Script
Trains 4 models, picks the best, saves pickle bundle.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, pickle
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

RANDOM_STATE = 42
DATA_PATH = "iris.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- 1. Load ----------
print("📥 Loading data...")
df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
print("Shape:", df.shape)

# Auto-detect target column
target_col = df.select_dtypes(include='object').columns[-1]
print("Target:", target_col)

X = df.drop(columns=[target_col])
y = df[target_col]

# ---------- 2. Encode + Scale ----------
le = LabelEncoder()
y_enc = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- 3. Split ----------
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
)

# ---------- 4. Models ----------
models = {
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
    "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "SVM":                SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE),
    "KNN":                KNeighborsClassifier(n_neighbors=5),
}

results, trained = [], {}
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    pred = mdl.predict(X_val)
    acc = accuracy_score(y_val, pred)
    f1  = f1_score(y_val, pred, average="weighted")
    cv  = cross_val_score(mdl, X_scaled, y_enc, cv=5).mean()
    results.append({"Model": name, "Accuracy": acc, "F1": f1, "CV": cv})
    trained[name] = mdl
    print(f"{name:20s} | Acc={acc:.4f} | F1={f1:.4f} | CV={cv:.4f}")

res_df = pd.DataFrame(results).sort_values("CV", ascending=False).reset_index(drop=True)
print("\n🏆 Best:", res_df.iloc[0]["Model"])

# ---------- 5. Retrain best on FULL data ----------
best_name = res_df.iloc[0]["Model"]
best_model = trained[best_name]
best_model.fit(X_scaled, y_enc)

# ---------- 6. Save pickle ----------
bundle = {
    "model":      best_model,
    "scaler":     scaler,
    "encoder":    le,
    "features":   X.columns.tolist(),
    "model_name": best_name,
    "accuracy":   float(res_df.iloc[0]["Accuracy"]),
}
with open(f"{MODEL_DIR}/iris_best_model.pkl", "wb") as f:
    pickle.dump(bundle, f)
print(f"✅ Saved → {MODEL_DIR}/iris_best_model.pkl")
