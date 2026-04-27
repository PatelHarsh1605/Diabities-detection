"""
preprocess.py
Data preprocessing pipeline for the Diabetes Detection System.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# ── Column names ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"
]
TARGET_COL = "diabetes"

SMOKING_ORDER = {
    "never": 0,
    "No Info": 1,
    "not current": 2,
    "ever": 3,
    "former": 4,
    "current": 5,
}

GENDER_MAP = {"Female": 0, "Male": 1, "Other": 2}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender"] = df["gender"].map(GENDER_MAP).fillna(2).astype(int)
    df["smoking_history"] = df["smoking_history"].map(SMOKING_ORDER).fillna(1).astype(int)
    return df


def handle_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Clip outliers using IQR method."""
    df = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.01)
        Q3 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=Q1, upper=Q3)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features that improve diabetes detection."""
    df = df.copy()
    # Composite metabolic risk score
    df["metabolic_risk"] = (df["HbA1c_level"] * df["blood_glucose_level"]) / 100.0
    # Age-BMI interaction
    df["age_bmi"] = df["age"] * df["bmi"] / 100.0
    # Vascular risk: hypertension + heart disease
    df["vascular_risk"] = df["hypertension"] + df["heart_disease"]
    return df


def get_feature_names(with_engineered: bool = True) -> list:
    base = ["gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]
    if with_engineered:
        return base + ["metabolic_risk", "age_bmi", "vascular_risk"]
    return base


def preprocess(
    df: pd.DataFrame,
    scaler=None,
    fit_scaler: bool = True,
    with_engineered: bool = True,
):
    """
    Full preprocessing pipeline.
    Returns (X_scaled, y, scaler, feature_names).
    """
    df = encode_categoricals(df)
    numeric_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    df = handle_outliers(df, numeric_cols)

    if with_engineered:
        df = engineer_features(df)

    feature_names = get_feature_names(with_engineered)
    X = df[feature_names].values
    y = df[TARGET_COL].values if TARGET_COL in df.columns else None

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler, feature_names


def preprocess_single(input_dict: dict, scaler, with_engineered: bool = True) -> np.ndarray:
    """Preprocess a single patient dict for inference."""
    df = pd.DataFrame([input_dict])
    df = encode_categoricals(df)
    if with_engineered:
        df = engineer_features(df)
    feature_names = get_feature_names(with_engineered)
    X = df[feature_names].values
    return scaler.transform(X)


def get_splits(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
