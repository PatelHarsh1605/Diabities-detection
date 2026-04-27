# 🩺 Diabetes Detection System

An AI-powered, production-ready diabetes risk assessment tool built with scikit-learn and Streamlit.

## Features

- **High-Recall ML Model** — Gradient Boosting / Random Forest tuned to minimize missed diagnoses
- **Interactive Streamlit UI** — Prediction form, clinical recommendations, risk gauge
- **Analytics Dashboard** — Distribution plots, correlation heatmap, risk stratification
- **Model Performance Page** — Confusion matrix, ROC curves, model comparison
- **Feature Contribution Analysis** — Per-patient explanation of top risk drivers
- **Clinical Recommendations** — Actionable feedback based on input values

## Quick Start

### 1. Clone / Extract the project

```bash
cd diabetes_system
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the dataset

Put `diabetes_prediction_dataset.csv` in the project root (same folder as `app.py`).

### 5. Train the model

```bash
python run_training.py
```

This will:
- Load and preprocess 100,000 patient records
- Train Logistic Regression, Random Forest, and Gradient Boosting
- Select the best model based on Recall + AUC
- Tune the decision threshold for ≥85% recall
- Save `model/trained_model.pkl`, `model/scaler.pkl`, `model/meta.json`
- Generate plots in `assets/`

Expected output (approx.):
```
Loading data...
Training Logistic Regression...  recall=0.856 auc=0.971
Training Random Forest...        recall=0.874 auc=0.985
Training Gradient Boosting...    recall=0.881 auc=0.987
Best model: Gradient Boosting
Threshold=0.27 -> {recall: 0.892, auc: 0.987, accuracy: 0.951}
ALL DONE
```

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 7. Run tests

```bash
python tests/test_prediction.py
```

## Project Structure

```
diabetes_system/
├── app.py                          # Streamlit app (5 pages)
├── run_training.py                 # One-shot training script
├── requirements.txt
├── diabetes_prediction_dataset.csv # Dataset (100K records)
├── model/
│   ├── trained_model.pkl           # Best model (joblib)
│   ├── scaler.pkl                  # StandardScaler
│   ├── feature_names.pkl           # Feature list
│   └── meta.json                   # Metrics, threshold, model name
├── src/
│   ├── preprocess.py               # Encoding, feature engineering, scaling
│   ├── predict.py                  # Inference + feature contributions
│   ├── train.py                    # Modular training pipeline (importable)
│   └── visualize.py                # Plotly chart functions
├── assets/
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── distributions.png
└── tests/
    └── test_prediction.py          # 15+ test cases + edge cases
```

## Dataset Features

| Feature | Type | Description |
|---|---|---|
| gender | categorical | Female / Male / Other |
| age | numeric | 0–80 years |
| hypertension | binary | 0 = No, 1 = Yes |
| heart_disease | binary | 0 = No, 1 = Yes |
| smoking_history | categorical | never / former / current / etc. |
| bmi | numeric | Body Mass Index |
| HbA1c_level | numeric | 3-month average blood glucose |
| blood_glucose_level | numeric | Fasting blood glucose (mg/dL) |
| **diabetes** | **binary** | **Target: 0 = No, 1 = Yes** |

## Engineered Features

| Feature | Formula | Clinical Meaning |
|---|---|---|
| metabolic_risk | HbA1c × Glucose / 100 | Combined metabolic burden |
| age_bmi | Age × BMI / 100 | Age-adjusted obesity risk |
| vascular_risk | Hypertension + Heart Disease | Cardiovascular comorbidity score |

## ML Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Class imbalance | `class_weight` balanced | 8.5% positive rate; no SMOTE needed |
| Primary metric | Recall | Missed diagnoses are clinically dangerous |
| Threshold tuning | Find max F1 at recall ≥ 85% | Balance recall target with precision |
| Model selection | Best recall + AUC | Generalization + sensitivity |

## Risk Levels

| Level | Probability | Recommended Action |
|---|---|---|
| 🟢 Low | < 30% | Routine annual check-up |
| 🟡 Medium | 30–60% | Consult physician, lifestyle changes |
| 🔴 High | > 60% | Immediate medical evaluation |

## Medical Disclaimer

This system is intended for **educational and research purposes only**.
It is **not a substitute** for professional medical diagnosis or treatment.
Always consult a qualified healthcare provider.
