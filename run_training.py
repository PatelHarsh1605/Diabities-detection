import warnings; warnings.filterwarnings("ignore")
import os, json, sys, numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve)

print("Loading data...")
df = pd.read_csv("diabetes_prediction_dataset.csv")
print("Shape:", df.shape)

SMOKING_ORDER = {"never":0,"No Info":1,"not current":2,"ever":3,"former":4,"current":5}
GENDER_MAP = {"Female":0,"Male":1,"Other":2}
df["gender"] = df["gender"].map(GENDER_MAP).fillna(2).astype(int)
df["smoking_history"] = df["smoking_history"].map(SMOKING_ORDER).fillna(1).astype(int)
df["metabolic_risk"] = df["HbA1c_level"] * df["blood_glucose_level"] / 100.0
df["age_bmi"] = df["age"] * df["bmi"] / 100.0
df["vascular_risk"] = df["hypertension"] + df["heart_disease"]

FEATURE_NAMES = ["gender","age","hypertension","heart_disease","smoking_history",
                 "bmi","HbA1c_level","blood_glucose_level","metabolic_risk","age_bmi","vascular_risk"]

X = df[FEATURE_NAMES].values
y = df["diabetes"].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

neg, pos = np.bincount(y_train)
cw = {0: (neg+pos)/(2*neg), 1: (neg+pos)/(2*pos)}

models = {
    "Logistic Regression": LogisticRegression(class_weight=cw, max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight=cw, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
}

results = {}
for name, m in models.items():
    print(f"Training {name}...")
    m.fit(X_train, y_train)
    proba = m.predict_proba(X_test)[:,1]
    pred = (proba >= 0.5).astype(int)
    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred),4),
        "recall": round(recall_score(y_test, pred),4),
        "precision": round(precision_score(y_test, pred, zero_division=0),4),
        "f1": round(f1_score(y_test, pred),4),
        "roc_auc": round(roc_auc_score(y_test, proba),4),
    }
    print(f"  recall={results[name]['recall']:.3f} auc={results[name]['roc_auc']:.3f}")

best_name = max(results, key=lambda n: (results[n]["recall"], results[n]["roc_auc"]))
best_model = models[best_name]
print(f"Best model: {best_name}")

best_thresh, best_f1 = 0.5, 0.0
proba = best_model.predict_proba(X_test)[:,1]
for t in np.arange(0.10, 0.80, 0.01):
    pred = (proba >= t).astype(int)
    r = recall_score(y_test, pred, zero_division=0)
    f = f1_score(y_test, pred, zero_division=0)
    if r >= 0.85 and f > best_f1:
        best_f1 = f; best_thresh = t

pred_final = (proba >= best_thresh).astype(int)
final = {
    "accuracy": round(accuracy_score(y_test, pred_final),4),
    "recall": round(recall_score(y_test, pred_final),4),
    "precision": round(precision_score(y_test, pred_final, zero_division=0),4),
    "f1": round(f1_score(y_test, pred_final),4),
    "roc_auc": round(roc_auc_score(y_test, proba),4),
}
print(f"Threshold={best_thresh:.2f} -> {final}")

os.makedirs("model", exist_ok=True)
os.makedirs("assets", exist_ok=True)
joblib.dump(best_model, "model/trained_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(FEATURE_NAMES, "model/feature_names.pkl")
meta = {"model_name": best_name, "threshold": round(float(best_thresh),2),
        "feature_names": FEATURE_NAMES, "metrics": final, "all_results": results}
with open("model/meta.json","w") as f: json.dump(meta, f, indent=2)
print("Saved model")

# Plots
numeric = df[FEATURE_NAMES + ["diabetes"]]
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True, annot_kws={"size":7})
ax.set_title("Feature Correlation Heatmap"); fig.tight_layout()
fig.savefig("assets/correlation_heatmap.png", dpi=150); plt.close(fig)
print("Saved correlation heatmap")

num_cols = ["age","bmi","HbA1c_level","blood_glucose_level"]
fig, axes = plt.subplots(2,2,figsize=(12,8))
for ax, col in zip(axes.flat, num_cols):
    for val, label, color in [(0,"No Diabetes","steelblue"),(1,"Diabetes","tomato")]:
        ax.hist(df.loc[df["diabetes"]==val, col], bins=40, alpha=0.6, label=label, color=color, density=True)
    ax.set_title(col); ax.legend()
fig.suptitle("Feature Distributions by Outcome"); fig.tight_layout()
fig.savefig("assets/distributions.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("Saved distributions")

cm = confusion_matrix(y_test, pred_final)
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes","Diabetes"], yticklabels=["No Diabetes","Diabetes"], ax=ax)
ax.set_title(f"Confusion Matrix - {best_name}"); ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
fig.tight_layout(); fig.savefig("assets/confusion_matrix.png", dpi=150); plt.close(fig)
print("Saved confusion matrix")

fig, ax = plt.subplots(figsize=(7,5))
for name, m in models.items():
    p = m.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, p)
    auc = roc_auc_score(y_test, p)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
ax.plot([0,1],[0,1],"k--"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curves"); ax.legend(); fig.tight_layout()
fig.savefig("assets/roc_curves.png", dpi=150); plt.close(fig)
print("Saved ROC curves")

if hasattr(best_model, "feature_importances_"):
    imp = best_model.feature_importances_
else:
    imp = np.abs(best_model.coef_[0])
idx = np.argsort(imp)
fig, ax = plt.subplots(figsize=(8,5))
colors = plt.cm.RdYlGn_r(np.linspace(0.2,0.8,len(FEATURE_NAMES)))
ax.barh([FEATURE_NAMES[i] for i in idx], [imp[i] for i in idx], color=colors)
ax.set_title("Feature Importance"); ax.set_xlabel("Importance")
fig.tight_layout(); fig.savefig("assets/feature_importance.png", dpi=150); plt.close(fig)
print("Saved feature importance")
print("ALL DONE")
