"""
predict.py
Inference utilities for the Diabetes Detection System.
"""

import os
import json
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

import sys
sys.path.insert(0, BASE_DIR)
from src.preprocess import preprocess_single


def load_artifacts():
    model         = joblib.load(os.path.join(MODEL_DIR, "trained_model.pkl"))
    scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        meta = json.load(f)
    return model, scaler, feature_names, meta


def predict_patient(patient_dict: dict, model, scaler, meta: dict):
    """
    Returns:
        prediction : 0 or 1
        probability: float 0–1
        risk_level : 'Low' | 'Medium' | 'High'
        confidence : 'Low' | 'Medium' | 'High'
    """
    threshold = meta.get("threshold", 0.5)
    X = preprocess_single(patient_dict, scaler)
    proba = model.predict_proba(X)[0, 1]
    pred  = int(proba >= threshold)

    if proba < 0.30:
        risk = "Low"
    elif proba < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    # Confidence based on distance from threshold
    distance = abs(proba - threshold)
    if distance < 0.10:
        confidence = "Low"
    elif distance < 0.25:
        confidence = "Medium"
    else:
        confidence = "High"

    return {
        "prediction":  pred,
        "probability": round(float(proba), 4),
        "risk_level":  risk,
        "confidence":  confidence,
        "threshold":   threshold,
    }


def get_feature_contributions(patient_dict: dict, model, scaler, feature_names: list):
    """
    Approximate SHAP-like feature contributions using finite differences.
    Returns list of (feature_name, contribution_value) sorted by |contribution|.
    """
    X = preprocess_single(patient_dict, scaler)
    base_proba = model.predict_proba(X)[0, 1]

    contribs = []
    eps = 0.1
    for i, name in enumerate(feature_names):
        X_plus = X.copy(); X_plus[0, i] += eps
        X_minus = X.copy(); X_minus[0, i] -= eps
        p_plus  = model.predict_proba(X_plus)[0, 1]
        p_minus = model.predict_proba(X_minus)[0, 1]
        grad = (p_plus - p_minus) / (2 * eps)
        contribs.append((name, round(float(grad * X[0, i]), 5)))

    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    return contribs
