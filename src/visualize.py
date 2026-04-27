"""
visualize.py
Plotly-based interactive visualization functions for the Streamlit dashboard.
"""

import json
import os
import numpy as np
import pandas as pd

# ── Safe Plotly import ────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Colour palette ────────────────────────────────────────────────────────────
C_POSITIVE  = "#EF4444"   # red  – diabetic
C_NEGATIVE  = "#22C55E"   # green – non-diabetic
C_ACCENT    = "#6366F1"   # indigo
C_SECONDARY = "#F59E0B"   # amber
TEMPLATE    = "plotly_white"


def fig_risk_gauge(probability: float):
    """Speedometer-style gauge for diabetes risk."""
    if not PLOTLY_AVAILABLE:
        return None
    pct = probability * 100
    color = C_NEGATIVE if pct < 30 else (C_SECONDARY if pct < 60 else C_POSITIVE)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        title={"text": "Diabetes Risk Score", "font": {"size": 18}},
        number={"suffix": "%", "font": {"size": 28}},
        delta={"reference": 50, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  30], "color": "#DCFCE7"},
                {"range": [30, 60], "color": "#FEF9C3"},
                {"range": [60, 100],"color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=20, l=20, r=20),
                      template=TEMPLATE)
    return fig


def fig_feature_contributions(contribs: list, n: int = 10):
    """Horizontal bar chart of feature contributions."""
    if not PLOTLY_AVAILABLE:
        return None
    names  = [c[0] for c in contribs[:n]]
    values = [c[1] for c in contribs[:n]]
    colors = [C_POSITIVE if v > 0 else C_NEGATIVE for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Contributions to Prediction",
        xaxis_title="Contribution (positive = increases risk)",
        height=400, template=TEMPLATE,
        margin=dict(l=150, r=60, t=50, b=40),
    )
    return fig


def fig_model_comparison(all_results: dict):
    """Grouped bar chart comparing model metrics."""
    if not PLOTLY_AVAILABLE:
        return None
    metrics = ["accuracy", "recall", "precision", "f1", "roc_auc"]
    fig = go.Figure()
    for model_name, vals in all_results.items():
        fig.add_trace(go.Bar(
            name=model_name,
            x=metrics,
            y=[vals.get(m, 0) for m in metrics],
            text=[f"{vals.get(m,0):.3f}" for m in metrics],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title="Model Performance Comparison",
        yaxis=dict(range=[0, 1.1]),
        height=420, template=TEMPLATE,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def fig_class_distribution(df: pd.DataFrame):
    """Donut chart of class balance."""
    if not PLOTLY_AVAILABLE:
        return None
    counts = df["diabetes"].value_counts()
    fig = go.Figure(go.Pie(
        labels=["No Diabetes", "Diabetes"],
        values=[counts.get(0, 0), counts.get(1, 0)],
        hole=0.45,
        marker_colors=[C_NEGATIVE, C_POSITIVE],
    ))
    fig.update_layout(title="Class Distribution", height=320,
                      template=TEMPLATE)
    return fig


def fig_age_bmi_scatter(df: pd.DataFrame, sample_n: int = 3000):
    """Age vs BMI coloured by outcome."""
    if not PLOTLY_AVAILABLE:
        return None
    s = df.sample(min(sample_n, len(df)), random_state=42)
    s["Outcome"] = s["diabetes"].map({0: "No Diabetes", 1: "Diabetes"})
    fig = px.scatter(
        s, x="age", y="bmi", color="Outcome",
        color_discrete_map={"No Diabetes": C_NEGATIVE, "Diabetes": C_POSITIVE},
        opacity=0.5,
        title="Age vs BMI by Outcome",
        labels={"age": "Age", "bmi": "BMI"},
        template=TEMPLATE,
    )
    fig.update_layout(height=380)
    return fig


def fig_glucose_hba1c(df: pd.DataFrame, sample_n: int = 3000):
    """Blood Glucose vs HbA1c coloured by outcome."""
    if not PLOTLY_AVAILABLE:
        return None
    s = df.sample(min(sample_n, len(df)), random_state=42)
    s["Outcome"] = s["diabetes"].map({0: "No Diabetes", 1: "Diabetes"})
    fig = px.scatter(
        s, x="blood_glucose_level", y="HbA1c_level", color="Outcome",
        color_discrete_map={"No Diabetes": C_NEGATIVE, "Diabetes": C_POSITIVE},
        opacity=0.5,
        title="Blood Glucose vs HbA1c",
        labels={"blood_glucose_level": "Blood Glucose (mg/dL)",
                "HbA1c_level": "HbA1c (%)"},
        template=TEMPLATE,
    )
    fig.update_layout(height=380)
    return fig


def fig_distribution_histograms(df: pd.DataFrame):
    """2×2 distribution histograms for key numeric features."""
    if not PLOTLY_AVAILABLE:
        return None
    features = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    labels   = ["Age", "BMI", "HbA1c (%)", "Blood Glucose (mg/dL)"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=labels)
    positions = [(1,1),(1,2),(2,1),(2,2)]
    for feat, label, (row, col) in zip(features, labels, positions):
        for outcome, name, color in [(0,"No Diabetes",C_NEGATIVE),(1,"Diabetes",C_POSITIVE)]:
            vals = df.loc[df["diabetes"]==outcome, feat]
            fig.add_trace(go.Histogram(
                x=vals, name=name, legendgroup=name,
                showlegend=(row==1 and col==1),
                marker_color=color, opacity=0.6,
                nbinsx=40,
            ), row=row, col=col)
    fig.update_layout(
        title="Feature Distributions by Outcome",
        barmode="overlay", height=500, template=TEMPLATE,
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


def fig_feature_importance_bar(feature_names: list, importances: list):
    """Horizontal bar chart of model feature importances."""
    if not PLOTLY_AVAILABLE:
        return None
    idx = np.argsort(importances)
    names = [feature_names[i] for i in idx]
    vals  = [importances[i] for i in idx]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(
            color=vals,
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Importance"),
        ),
    ))
    fig.update_layout(
        title="Feature Importances (Best Model)",
        xaxis_title="Importance",
        height=420, template=TEMPLATE,
        margin=dict(l=160),
    )
    return fig


def fig_risk_stratification(probas: np.ndarray):
    """Histogram of predicted probabilities, coloured by risk zone."""
    if not PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    bins = np.linspace(0, 1, 41)
    for lo, hi, label, color in [
        (0.0, 0.3, "Low Risk",    "#22C55E"),
        (0.3, 0.6, "Medium Risk", "#F59E0B"),
        (0.6, 1.0, "High Risk",   "#EF4444"),
    ]:
        mask = (probas >= lo) & (probas < hi)
        fig.add_trace(go.Histogram(
            x=probas[mask], name=label, marker_color=color,
            xbins=dict(start=lo, end=hi, size=0.025),
            opacity=0.8,
        ))
    fig.update_layout(
        title="Risk Stratification Distribution",
        xaxis_title="Predicted Probability",
        yaxis_title="Count",
        barmode="stack", height=360, template=TEMPLATE,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig
