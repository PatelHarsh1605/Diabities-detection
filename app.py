"""
app.py — Diabetes Detection System
Main Streamlit application entry point.
Run: streamlit run app.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Diabetes Detection System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Imports from src ──────────────────────────────────────────────────────────
from src.preprocess import preprocess_single
from src.predict    import load_artifacts, predict_patient, get_feature_contributions

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global font */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #0f2340 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px; }

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    text-align: center;
    border-top: 4px solid #6366f1;
}
.metric-card h2 { margin: 4px 0; font-size: 2rem; color: #1e3a5f; }
.metric-card p  { margin: 0; color: #64748b; font-size: 13px; }

/* Risk badges */
.risk-low    { background:#dcfce7; color:#166534; padding:6px 14px; border-radius:20px; font-weight:600; }
.risk-medium { background:#fef9c3; color:#854d0e; padding:6px 14px; border-radius:20px; font-weight:600; }
.risk-high   { background:#fee2e2; color:#991b1b; padding:6px 14px; border-radius:20px; font-weight:600; }

/* Prediction result box */
.result-positive {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.result-positive h2, .result-negative h2 { margin: 8px 0; font-size: 1.8rem; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: 16px;
    padding: 32px 40px;
    color: white;
    margin-bottom: 24px;
}
.hero h1 { margin: 0 0 8px; font-size: 2.2rem; }
.hero p  { margin: 0; opacity: 0.85; font-size: 1.05rem; }

/* Section headers */
.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1e3a5f;
    border-left: 4px solid #6366f1;
    padding-left: 12px;
    margin: 20px 0 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Load model artefacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model, scaler, feature_names, meta = load_artifacts()
        return model, scaler, feature_names, meta, None
    except Exception as e:
        return None, None, None, None, str(e)


# ── Load dataset ──────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    path = os.path.join(BASE_DIR, "diabetes_prediction_dataset.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 DiabetesAI")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔬 Prediction", "📊 Dashboard", "📈 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Model Status**")
    model, scaler, feature_names, meta, err = load_model()
    if model is not None:
        st.success("✅ Model loaded")
        if meta:
            st.markdown(f"**{meta.get('model_name','N/A')}**")
            m = meta.get("metrics", {})
            st.metric("Recall",   f"{m.get('recall',0):.1%}")
            st.metric("AUC",      f"{m.get('roc_auc',0):.3f}")
            st.metric("Accuracy", f"{m.get('accuracy',0):.1%}")
    else:
        st.error("⚠️ Model not found")
        st.info("Run `python run_training.py` first.")
    st.markdown("---")
    st.caption("v1.0 · Built with Streamlit")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <h1>🩺 Diabetes Detection System</h1>
        <p>AI-powered early diabetes risk assessment using clinical biomarkers.
        Trained on 100,000 patient records with a focus on maximizing recall
        to minimize missed diagnoses.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <p>Training Records</p><h2>100K</h2><p>Patient dataset</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        recall_val = f"{meta['metrics']['recall']:.1%}" if meta else "N/A"
        st.markdown(f"""<div class="metric-card" style="border-color:#22c55e">
            <p>Model Recall</p><h2>{recall_val}</h2><p>Low false negatives</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        auc_val = f"{meta['metrics']['roc_auc']:.3f}" if meta else "N/A"
        st.markdown(f"""<div class="metric-card" style="border-color:#f59e0b">
            <p>ROC-AUC Score</p><h2>{auc_val}</h2><p>Discrimination power</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card" style="border-color:#ef4444">
            <p>Features Used</p><h2>11</h2><p>Incl. engineered</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">🧬 Key Risk Factors</div>', unsafe_allow_html=True)
        st.markdown("""
        | Factor | Clinical Significance |
        |---|---|
        | **HbA1c Level** | Average blood sugar over 3 months; >6.5% indicates diabetes |
        | **Blood Glucose** | Fasting glucose >126 mg/dL is diagnostic |
        | **BMI** | Obesity (≥30) is a major modifiable risk factor |
        | **Age** | Risk increases significantly after 45 years |
        | **Hypertension** | Often co-occurs with insulin resistance |
        | **Heart Disease** | Metabolic syndrome indicator |
        """)
    with col_r:
        st.markdown('<div class="section-header">⚠️ Risk Thresholds</div>', unsafe_allow_html=True)
        st.markdown("""
        | Risk Level | Probability | Action |
        |---|---|---|
        | 🟢 **Low** | < 30% | Regular check-ups, healthy lifestyle |
        | 🟡 **Medium** | 30–60% | Consult physician, lifestyle changes |
        | 🔴 **High** | > 60% | Immediate medical evaluation |
        """)
        st.info("💡 This tool is for **screening purposes only**. Always consult a qualified healthcare provider for diagnosis and treatment.")

    st.markdown('<div class="section-header">🚀 How to Use</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Step 1 — Enter Data**\n\nNavigate to the *Prediction* page and fill in your clinical measurements.")
    with c2:
        st.markdown("**Step 2 — Get Prediction**\n\nClick *Predict* to receive instant risk assessment with probability score.")
    with c3:
        st.markdown("**Step 3 — Review Insights**\n\nSee which features contribute most to your risk and explore the Dashboard.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Prediction":
    st.markdown('<div class="section-header">🔬 Patient Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown("Enter the patient's clinical measurements below to generate a diabetes risk prediction.")

    if model is None:
        st.error("Model not loaded. Please run `python run_training.py` first.")
        st.stop()

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.markdown("#### 📋 Patient Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            age    = st.slider("Age (years)", 0, 80, 45)
            bmi    = st.slider("BMI (kg/m²)", 10.0, 70.0, 28.0, step=0.1)

        with col2:
            st.markdown("**Metabolic Markers**")
            hba1c   = st.slider("HbA1c Level (%)", 3.5, 9.0, 5.5, step=0.1)
            glucose = st.slider("Blood Glucose (mg/dL)", 80, 300, 120)
            smoking = st.selectbox("Smoking History",
                                   ["never","No Info","not current","ever","former","current"])

        with col3:
            st.markdown("**Medical History**")
            hypertension  = st.selectbox("Hypertension",  ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease",  ["No", "Yes"])
            st.markdown("")
            st.markdown("")
            submitted = st.form_submit_button("🔍 Predict Diabetes Risk",
                                              use_container_width=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    if submitted:
        patient = {
            "gender":              gender,
            "age":                 float(age),
            "hypertension":        1 if hypertension == "Yes" else 0,
            "heart_disease":       1 if heart_disease == "Yes" else 0,
            "smoking_history":     smoking,
            "bmi":                 float(bmi),
            "HbA1c_level":         float(hba1c),
            "blood_glucose_level": float(glucose),
        }

        with st.spinner("Analysing patient data…"):
            result = predict_patient(patient, model, scaler, meta)
            contribs = get_feature_contributions(patient, model, scaler, feature_names)

        pred  = result["prediction"]
        proba = result["probability"]
        risk  = result["risk_level"]

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            if pred == 1:
                st.markdown(f"""<div class="result-positive">
                    <h2>⚠️ Diabetic</h2>
                    <p style="font-size:1.1rem">Probability: <strong>{proba:.1%}</strong></p>
                    <p>Risk Level: <span class="risk-high">🔴 {risk} Risk</span></p>
                    <p style="margin-top:12px;font-size:13px;color:#7f1d1d">
                    Please seek immediate medical consultation.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-negative">
                    <h2>✅ Not Diabetic</h2>
                    <p style="font-size:1.1rem">Probability: <strong>{proba:.1%}</strong></p>
                    <p>Risk Level: <span class="risk-{'medium' if risk=='Medium' else 'low'}">
                    {'🟡' if risk=='Medium' else '🟢'} {risk} Risk</span></p>
                    <p style="margin-top:12px;font-size:13px;color:#14532d">
                    Maintain healthy lifestyle habits.</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("**Prediction Details**")
            det1, det2, det3 = st.columns(3)
            det1.metric("Probability",  f"{proba:.1%}")
            det2.metric("Threshold",    f"{meta.get('threshold', 0.5):.2f}")
            det3.metric("Confidence",   result["confidence"])

        with gauge_col:
            # Gauge via progress bars (no plotly needed)
            st.markdown("**Risk Probability Gauge**")
            pct = int(proba * 100)
            color = "🟢" if pct < 30 else ("🟡" if pct < 60 else "🔴")
            st.markdown(f"### {color} {pct}%")
            st.progress(proba)

            # Risk zone indicator
            st.markdown("""
            | Zone | Range |
            |------|-------|
            | 🟢 Low | 0–30% |
            | 🟡 Medium | 30–60% |
            | 🔴 High | 60–100% |
            """)

        # ── Feature contributions ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 Feature Contribution Analysis")
        st.caption("Shows which factors increased (+) or decreased (−) the diabetes risk for this patient.")

        top_n = min(8, len(contribs))
        names_plot  = [c[0] for c in contribs[:top_n]]
        values_plot = [c[1] for c in contribs[:top_n]]

        try:
            import plotly.graph_objects as go
            colors = ["#ef4444" if v > 0 else "#22c55e" for v in values_plot]
            fig = go.Figure(go.Bar(
                x=values_plot, y=names_plot, orientation="h",
                marker_color=colors,
                text=[f"{v:+.4f}" for v in values_plot],
                textposition="outside",
            ))
            fig.update_layout(
                height=350, template="plotly_white",
                xaxis_title="Contribution (+ increases risk)",
                margin=dict(l=160, r=60, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            # Fallback to text table
            contrib_df = pd.DataFrame(contribs[:top_n], columns=["Feature", "Contribution"])
            contrib_df["Direction"] = contrib_df["Contribution"].apply(
                lambda x: "⬆️ Increases Risk" if x > 0 else "⬇️ Decreases Risk")
            contrib_df["Contribution"] = contrib_df["Contribution"].apply(lambda x: f"{x:+.4f}")
            st.dataframe(contrib_df, use_container_width=True)

        # ── Clinical recommendations ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💊 Clinical Recommendations")
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            if hba1c >= 6.5:
                st.error(f"🚨 **HbA1c {hba1c}%** — Above diagnostic threshold (≥6.5%). Immediate evaluation needed.")
            elif hba1c >= 5.7:
                st.warning(f"⚠️ **HbA1c {hba1c}%** — Pre-diabetic range (5.7–6.4%). Monitor closely.")
            else:
                st.success(f"✅ **HbA1c {hba1c}%** — Normal range. Keep maintaining.")

            if glucose >= 126:
                st.error(f"🚨 **Fasting Glucose {glucose} mg/dL** — Above diagnostic threshold (≥126).")
            elif glucose >= 100:
                st.warning(f"⚠️ **Glucose {glucose} mg/dL** — Impaired fasting glucose (100–125).")
            else:
                st.success(f"✅ **Glucose {glucose} mg/dL** — Normal range.")

        with rec_col2:
            if bmi >= 30:
                st.warning(f"⚠️ **BMI {bmi:.1f}** — Obese category. Weight management recommended.")
            elif bmi >= 25:
                st.info(f"ℹ️ **BMI {bmi:.1f}** — Overweight. Moderate lifestyle changes advised.")
            else:
                st.success(f"✅ **BMI {bmi:.1f}** — Healthy weight range.")

            if hypertension == "Yes":
                st.warning("⚠️ **Hypertension present** — Increases cardiovascular and metabolic risk.")
            if heart_disease == "Yes":
                st.error("🚨 **Heart disease present** — Significantly elevated metabolic syndrome risk.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown('<div class="section-header">📊 Data Analytics Dashboard</div>', unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.error("Dataset not found. Place `diabetes_prediction_dataset.csv` in the project root.")
        st.stop()

    # Summary stats
    total     = len(df)
    diabetic  = df["diabetes"].sum()
    non_diab  = total - diabetic
    prev_rate = diabetic / total

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",    f"{total:,}")
    c2.metric("Diabetic Cases",   f"{diabetic:,}")
    c3.metric("Non-Diabetic",     f"{non_diab:,}")
    c4.metric("Prevalence Rate",  f"{prev_rate:.1%}")

    st.markdown("---")

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 Distributions", "🔥 Correlations", "📊 Risk Factors", "🗺️ Feature Interactions"])

        # ── Tab 1: Distributions ─────────────────────────────────────────────
        with tab1:
            feat = st.selectbox("Select Feature",
                ["age","bmi","HbA1c_level","blood_glucose_level"], key="dist_feat")
            fig = go.Figure()
            for val, label, color in [(0,"No Diabetes","#22c55e"),(1,"Diabetes","#ef4444")]:
                fig.add_trace(go.Histogram(
                    x=df.loc[df["diabetes"]==val, feat],
                    name=label, marker_color=color,
                    opacity=0.65, nbinsx=50,
                ))
            fig.update_layout(barmode="overlay", title=f"Distribution of {feat} by Outcome",
                              xaxis_title=feat, yaxis_title="Count",
                              height=400, template="plotly_white",
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)

            # Box plots
            df_plot = df.copy()
            df_plot["Outcome"] = df_plot["diabetes"].map({0:"No Diabetes",1:"Diabetes"})
            fig2 = px.box(df_plot, x="Outcome", y=feat, color="Outcome",
                          color_discrete_map={"No Diabetes":"#22c55e","Diabetes":"#ef4444"},
                          title=f"{feat} Box Plot by Outcome", template="plotly_white")
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

        # ── Tab 2: Correlations ──────────────────────────────────────────────
        with tab2:
            numeric_cols = ["age","hypertension","heart_disease","bmi",
                            "HbA1c_level","blood_glucose_level","diabetes"]
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, title="Feature Correlation Heatmap",
                            template="plotly_white")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Correlations with Diabetes Outcome**")
            corr_target = corr["diabetes"].drop("diabetes").sort_values(ascending=False)
            fig2 = go.Figure(go.Bar(
                x=corr_target.values, y=corr_target.index,
                orientation="h",
                marker_color=["#ef4444" if v > 0 else "#22c55e" for v in corr_target.values],
            ))
            fig2.update_layout(title="Feature Correlations with Diabetes",
                               xaxis_title="Pearson Correlation",
                               height=350, template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

        # ── Tab 3: Risk Factors ──────────────────────────────────────────────
        with tab3:
            col_l, col_r = st.columns(2)
            with col_l:
                # Class distribution donut
                counts = df["diabetes"].value_counts()
                fig = go.Figure(go.Pie(
                    labels=["No Diabetes","Diabetes"],
                    values=[counts.get(0,0), counts.get(1,0)],
                    hole=0.45,
                    marker_colors=["#22c55e","#ef4444"],
                ))
                fig.update_layout(title="Class Distribution", height=320, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with col_r:
                # Smoking history
                smk = df.groupby("smoking_history")["diabetes"].mean().reset_index()
                smk.columns = ["Smoking","Diabetes Rate"]
                fig2 = px.bar(smk.sort_values("Diabetes Rate", ascending=False),
                              x="Smoking", y="Diabetes Rate", color="Diabetes Rate",
                              color_continuous_scale="Reds",
                              title="Diabetes Rate by Smoking History",
                              template="plotly_white")
                fig2.update_layout(height=320)
                st.plotly_chart(fig2, use_container_width=True)

            # Age group analysis
            df["age_group"] = pd.cut(df["age"], bins=[0,20,35,50,65,80],
                                     labels=["<20","20-35","35-50","50-65","65+"])
            age_diab = df.groupby("age_group", observed=True)["diabetes"].mean().reset_index()
            fig3 = px.bar(age_diab, x="age_group", y="diabetes",
                          color="diabetes", color_continuous_scale="RdYlGn_r",
                          title="Diabetes Prevalence by Age Group",
                          labels={"diabetes":"Prevalence Rate","age_group":"Age Group"},
                          template="plotly_white")
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)

        # ── Tab 4: Feature Interactions ──────────────────────────────────────
        with tab4:
            sample = df.sample(min(4000, len(df)), random_state=42)
            sample["Outcome"] = sample["diabetes"].map({0:"No Diabetes",1:"Diabetes"})

            col_x = st.selectbox("X-axis", ["age","bmi","HbA1c_level","blood_glucose_level"], index=0)
            col_y = st.selectbox("Y-axis", ["age","bmi","HbA1c_level","blood_glucose_level"], index=2)

            fig = px.scatter(sample, x=col_x, y=col_y, color="Outcome",
                             color_discrete_map={"No Diabetes":"#22c55e","Diabetes":"#ef4444"},
                             opacity=0.5,
                             title=f"{col_x} vs {col_y} by Outcome",
                             template="plotly_white")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("Plotly not installed. Install with: `pip install plotly`")
        st.markdown("**Dataset Summary Statistics**")
        st.dataframe(df.describe(), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown('<div class="section-header">📈 Model Performance & Evaluation</div>', unsafe_allow_html=True)

    if model is None:
        st.error("Model not loaded. Run `python run_training.py` first.")
        st.stop()

    # ── Best model summary ────────────────────────────────────────────────────
    m = meta.get("metrics", {})
    st.markdown(f"**Best Model:** `{meta.get('model_name','N/A')}`  |  **Decision Threshold:** `{meta.get('threshold',0.5)}`")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{m.get('accuracy',0):.2%}")
    c2.metric("Recall",    f"{m.get('recall',0):.2%}",    delta="Primary target")
    c3.metric("Precision", f"{m.get('precision',0):.2%}")
    c4.metric("F1-Score",  f"{m.get('f1',0):.2%}")
    c5.metric("ROC-AUC",   f"{m.get('roc_auc',0):.3f}")

    st.markdown("---")

    # ── Saved plots ───────────────────────────────────────────────────────────
    assets = os.path.join(BASE_DIR, "assets")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏆 Model Comparison", "🎯 Confusion Matrix", "📉 ROC Curves", "🔍 Feature Importance"])

    with tab1:
        all_results = meta.get("all_results", {})
        if all_results:
            try:
                import plotly.graph_objects as go
                metrics_list = ["accuracy","recall","precision","f1","roc_auc"]
                fig = go.Figure()
                for mname, vals in all_results.items():
                    fig.add_trace(go.Bar(
                        name=mname,
                        x=metrics_list,
                        y=[vals.get(k,0) for k in metrics_list],
                        text=[f"{vals.get(k,0):.3f}" for k in metrics_list],
                        textposition="outside",
                    ))
                fig.update_layout(
                    barmode="group",
                    title="Model Performance Comparison",
                    yaxis=dict(range=[0,1.12]),
                    height=450, template="plotly_white",
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.dataframe(pd.DataFrame(all_results).T, use_container_width=True)
        else:
            st.info("Train multiple models to see comparison.")

    with tab2:
        img_path = os.path.join(assets, "confusion_matrix.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Confusion Matrix", use_container_width=False)
        else:
            st.info("Run training to generate confusion matrix plot.")
        st.markdown("""
        **Interpretation:**
        - **True Positive (TP):** Correctly identified diabetic patients
        - **True Negative (TN):** Correctly identified healthy patients
        - **False Negative (FN):** Missed diabetic cases — *minimized by high recall*
        - **False Positive (FP):** Healthy patients flagged as diabetic
        """)

    with tab3:
        img_path = os.path.join(assets, "roc_curves.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="ROC Curves — All Models", use_container_width=True)
        else:
            st.info("Run training to generate ROC curves.")

    with tab4:
        img_path = os.path.join(assets, "feature_importance.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Feature Importance", use_container_width=True)
        else:
            st.info("Run training to generate feature importance plot.")

    # ── Threshold analysis ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚖️ Decision Threshold Analysis")
    st.markdown("""
    The model uses a **custom decision threshold** (not the default 0.5) to maximize recall.
    Lowering the threshold catches more diabetic cases but increases false positives.
    """)
    thresh = meta.get("threshold", 0.5)
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(f"📌 Optimal threshold found: **{thresh}**")
        st.markdown(f"""
        - Threshold was tuned to achieve **recall ≥ 85%**
        - Best F1-score at this recall level
        - Patients with predicted probability ≥ {thresh} are classified as diabetic
        """)
    with col_b:
        st.warning("**Clinical Rationale for High Recall:**\n\n"
                   "Missing a diabetic diagnosis (false negative) has much higher "
                   "clinical cost than a false alarm. This model is tuned to minimize "
                   "missed diagnoses even at the expense of more false positives.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<div class="section-header">ℹ️ About This System</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🧠 System Architecture

        **Machine Learning Pipeline:**
        - Data preprocessing with feature engineering
        - Class imbalance handled via `class_weight`
        - 3 models trained and compared
        - Threshold-tuned for maximum recall

        **Models Evaluated:**
        - Logistic Regression (baseline)
        - Random Forest (ensemble)
        - Gradient Boosting (boosting)

        **Engineered Features:**
        - `metabolic_risk` = HbA1c × Glucose / 100
        - `age_bmi` = Age × BMI / 100
        - `vascular_risk` = Hypertension + Heart Disease
        """)

    with col2:
        st.markdown("""
        ### 📦 Tech Stack

        | Component | Library |
        |---|---|
        | ML Models | scikit-learn |
        | UI | Streamlit |
        | Visualizations | Plotly |
        | Data | pandas, numpy |
        | Model Storage | joblib |
        | Plots | matplotlib, seaborn |

        ### ⚠️ Medical Disclaimer
        This system is a **screening tool** for educational and
        research purposes only. It is **not a substitute** for
        professional medical diagnosis. Always consult a qualified
        healthcare provider.
        """)

    st.markdown("---")
    st.markdown("### 📁 Project Structure")
    st.code("""
diabetes_system/
├── app.py                          # Streamlit main app
├── run_training.py                 # Model training script
├── diabetes_prediction_dataset.csv # Dataset
├── requirements.txt
├── model/
│   ├── trained_model.pkl           # Saved best model
│   ├── scaler.pkl                  # Feature scaler
│   ├── feature_names.pkl           # Feature list
│   └── meta.json                   # Metrics & threshold
├── src/
│   ├── preprocess.py               # Data preprocessing
│   ├── predict.py                  # Inference utilities
│   ├── train.py                    # Full training module
│   └── visualize.py                # Plotly chart functions
├── assets/
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── distributions.png
└── tests/
    └── test_prediction.py
    """, language="text")
