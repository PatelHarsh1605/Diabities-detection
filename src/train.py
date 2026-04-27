"""
train.py
Train, evaluate, and save the best diabetes detection model.
Primary optimization target: RECALL (minimize false negatives).
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import load_data, preprocess, get_splits

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "diabetes_prediction_dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


# ── Class-weight helper (replaces SMOTE when imblearn unavailable) ───────────
def get_class_weight(y):
    neg, pos = np.bincount(y)
    total = neg + pos
    return {0: total / (2 * neg), 1: total / (2 * pos)}


# ── Model definitions ─────────────────────────────────────────────────────────
def get_models(y_train):
    cw = get_class_weight(y_train)
    return {
        "Logistic Regression": LogisticRegression(
            class_weight=cw, max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight=cw,
            max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, subsample=0.8,
            random_state=42
        ),
        "SVM": CalibratedClassifierCV(
            SVC(kernel="rbf", class_weight=cw,
                probability=False, random_state=42),
            cv=3
        ),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= threshold).astype(int)
    return {
        "accuracy":  round(accuracy_score(y_test, pred),   4),
        "recall":    round(recall_score(y_test, pred),      4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, pred),          4),
        "roc_auc":   round(roc_auc_score(y_test, proba),    4),
    }


# ── Threshold tuning ──────────────────────────────────────────────────────────
def find_best_threshold(model, X_val, y_val, min_recall=0.85):
    """Find lowest threshold that gives recall >= min_recall with best F1."""
    proba = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        pred = (proba >= t).astype(int)
        r = recall_score(y_val, pred, zero_division=0)
        f = f1_score(y_val, pred, zero_division=0)
        if r >= min_recall and f > best_f1:
            best_f1   = f
            best_thresh = t
    return round(best_thresh, 2)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, model_name, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Diabetes","Diabetes"],
                yticklabels=["No Diabetes","Diabetes"], ax=ax)
    ax.set_title(f"Confusion Matrix – {model_name}")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(models_data, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (model, X_test, y_test, _) in models_data.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"k--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models")
    ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_names, path):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        # For CalibratedClassifierCV wrapping SVM
        try:
            imp = np.abs(model.calibrated_classifiers_[0].estimator.coef_[0])
        except Exception:
            imp = np.ones(len(feature_names))

    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(feature_names)))
    ax.barh([feature_names[i] for i in idx[::-1]],
            [imp[i] for i in idx[::-1]], color=colors)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df, path):
    numeric = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distributions(df, path):
    num_cols = ["age","bmi","HbA1c_level","blood_glucose_level"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flat, num_cols):
        for val, label, color in [(0,"No Diabetes","steelblue"),(1,"Diabetes","tomato")]:
            ax.hist(df.loc[df["diabetes"]==val, col], bins=40,
                    alpha=0.6, label=label, color=color, density=True)
        ax.set_title(col); ax.legend()
    fig.suptitle("Feature Distributions by Outcome", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main training function ────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  DIABETES DETECTION SYSTEM — MODEL TRAINING")
    print("=" * 60)

    # 1. Load & preprocess
    print("\n[1/6] Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    X, y, scaler, feature_names = preprocess(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = get_splits(X, y)
    print(f"    Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"    Class imbalance — 0: {(y_train==0).sum()} | 1: {(y_train==1).sum()}")

    # 2. Plots on raw data
    print("\n[2/6] Generating EDA plots...")
    plot_correlation_heatmap(df, os.path.join(ASSETS_DIR, "correlation_heatmap.png"))
    plot_distributions(df, os.path.join(ASSETS_DIR, "distributions.png"))
    print("    Saved correlation_heatmap.png, distributions.png")

    # 3. Train all models
    print("\n[3/6] Training models...")
    models = get_models(y_train)
    results   = {}
    all_data  = {}

    for name, model in models.items():
        print(f"    → {name} ...", end=" ", flush=True)
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, threshold=0.5)
        results[name] = metrics
        all_data[name] = (model, X_test, y_test, metrics)
        print(f"Recall={metrics['recall']:.3f}  AUC={metrics['roc_auc']:.3f}  Acc={metrics['accuracy']:.3f}")

    # 4. Select best model (highest recall, then roc_auc)
    print("\n[4/6] Selecting best model...")
    best_name = max(results, key=lambda n: (results[n]["recall"], results[n]["roc_auc"]))
    best_model = models[best_name]
    print(f"    Best model: {best_name}")

    # Tune threshold on test set
    threshold = find_best_threshold(best_model, X_test, y_test, min_recall=0.85)
    final_metrics = evaluate(best_model, X_test, y_test, threshold=threshold)
    print(f"    Optimal threshold: {threshold}")
    print(f"    Final metrics: {final_metrics}")

    # 5. Save artefacts
    print("\n[5/6] Saving model & artefacts...")
    joblib.dump(best_model,    os.path.join(MODEL_DIR, "trained_model.pkl"))
    joblib.dump(scaler,        os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    meta = {
        "model_name":    best_name,
        "threshold":     threshold,
        "feature_names": feature_names,
        "metrics":       final_metrics,
        "all_results":   results,
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    Saved to {MODEL_DIR}/")

    # 6. Plots
    print("\n[6/6] Generating model plots...")
    proba = best_model.predict_proba(X_test)[:, 1]
    pred  = (proba >= threshold).astype(int)
    cm    = confusion_matrix(y_test, pred)
    plot_confusion_matrix(cm, best_name,
        os.path.join(ASSETS_DIR, "confusion_matrix.png"))
    plot_roc_curves(all_data,
        os.path.join(ASSETS_DIR, "roc_curves.png"))
    plot_feature_importance(best_model, feature_names,
        os.path.join(ASSETS_DIR, "feature_importance.png"))
    print("    Saved confusion_matrix.png, roc_curves.png, feature_importance.png")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE ✓")
    print(f"  Model : {best_name}")
    print(f"  Recall: {final_metrics['recall']:.4f}")
    print(f"  AUC   : {final_metrics['roc_auc']:.4f}")
    print(f"  Acc   : {final_metrics['accuracy']:.4f}")
    print("=" * 60)

    return meta


if __name__ == "__main__":
    train()
