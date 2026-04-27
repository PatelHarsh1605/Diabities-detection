"""
tests/test_prediction.py
Unit tests and edge cases for the Diabetes Detection System.
Run: python tests/test_prediction.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_single
from src.predict import load_artifacts, predict_patient, get_feature_contributions

# ── Colour output ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; RESET = "\033[0m"; BOLD = "\033[1m"
PASS  = f"{GREEN}✓ PASS{RESET}"
FAIL  = f"{RED}✗ FAIL{RESET}"

results = {"passed": 0, "failed": 0}

def run_test(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results["passed"] += 1
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        results["failed"] += 1


# ── Test cases ────────────────────────────────────────────────────────────────
TEST_CASES = {
    "high_risk": {
        "patient": {
            "gender": "Female", "age": 62.0, "hypertension": 1,
            "heart_disease": 1, "smoking_history": "current",
            "bmi": 38.5, "HbA1c_level": 8.2, "blood_glucose_level": 210,
        },
        "expected_risk": "High",
        "expected_pred": 1,
        "description": "Obese elderly woman with hypertension, heart disease, high HbA1c",
    },
    "low_risk": {
        "patient": {
            "gender": "Male", "age": 24.0, "hypertension": 0,
            "heart_disease": 0, "smoking_history": "never",
            "bmi": 22.1, "HbA1c_level": 4.8, "blood_glucose_level": 88,
        },
        "expected_risk": "Low",
        "expected_pred": 0,
        "description": "Young healthy male with no risk factors",
    },
    "borderline": {
        "patient": {
            "gender": "Female", "age": 48.0, "hypertension": 1,
            "heart_disease": 0, "smoking_history": "former",
            "bmi": 29.5, "HbA1c_level": 6.0, "blood_glucose_level": 118,
        },
        "expected_risk": None,  # borderline — no fixed expectation
        "expected_pred": None,
        "description": "Middle-aged woman, borderline pre-diabetic markers",
    },
    "edge_min": {
        "patient": {
            "gender": "Female", "age": 0.08, "hypertension": 0,
            "heart_disease": 0, "smoking_history": "never",
            "bmi": 10.1, "HbA1c_level": 3.5, "blood_glucose_level": 80,
        },
        "expected_risk": "Low",
        "expected_pred": 0,
        "description": "Minimum possible values — should not crash",
    },
    "edge_max": {
        "patient": {
            "gender": "Male", "age": 80.0, "hypertension": 1,
            "heart_disease": 1, "smoking_history": "current",
            "bmi": 95.0, "HbA1c_level": 9.0, "blood_glucose_level": 300,
        },
        "expected_risk": "High",
        "expected_pred": 1,
        "description": "Maximum possible values — model must handle without error",
    },
    "gender_other": {
        "patient": {
            "gender": "Other", "age": 35.0, "hypertension": 0,
            "heart_disease": 0, "smoking_history": "No Info",
            "bmi": 25.0, "HbA1c_level": 5.5, "blood_glucose_level": 100,
        },
        "expected_risk": None,
        "expected_pred": None,
        "description": "Non-binary gender encoding — must not crash",
    },
}


def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  DIABETES DETECTION SYSTEM — TEST SUITE{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # ── 1. Load artefacts ─────────────────────────────────────────────────────
    print(f"\n{BOLD}[1] Loading model artefacts{RESET}")
    try:
        model, scaler, feature_names, meta = load_artifacts()
        print(f"  {PASS}  Model loaded: {meta.get('model_name')}")
        print(f"  {PASS}  Threshold: {meta.get('threshold')}")
        print(f"  {PASS}  Features: {len(feature_names)}")
    except Exception as e:
        print(f"  {FAIL}  Could not load model: {e}")
        print(f"\n  {YELLOW}Run `python run_training.py` first, then re-run tests.{RESET}\n")
        return

    # ── 2. Preprocessing tests ────────────────────────────────────────────────
    print(f"\n{BOLD}[2] Preprocessing Tests{RESET}")
    for name, tc in TEST_CASES.items():
        def _test(tc=tc):
            X = preprocess_single(tc["patient"], scaler)
            assert X.shape == (1, len(feature_names)), \
                f"Shape mismatch: {X.shape} vs (1,{len(feature_names)})"
            assert not np.any(np.isnan(X)), "NaN values in preprocessed output"
        run_test(f"Preprocess — {name}", _test)

    # ── 3. Prediction tests ───────────────────────────────────────────────────
    print(f"\n{BOLD}[3] Prediction Tests{RESET}")
    for name, tc in TEST_CASES.items():
        def _test(tc=tc):
            result = predict_patient(tc["patient"], model, scaler, meta)
            # Output keys
            for key in ["prediction","probability","risk_level","confidence","threshold"]:
                assert key in result, f"Missing key: {key}"
            # Types
            assert result["prediction"] in [0, 1]
            assert 0.0 <= result["probability"] <= 1.0
            assert result["risk_level"] in ["Low","Medium","High"]
            assert result["confidence"] in ["Low","Medium","High"]
            # Expected checks
            if tc["expected_pred"] is not None:
                assert result["prediction"] == tc["expected_pred"], \
                    f"Expected pred={tc['expected_pred']}, got {result['prediction']} (prob={result['probability']:.3f})"
            if tc["expected_risk"] is not None:
                assert result["risk_level"] == tc["expected_risk"], \
                    f"Expected risk={tc['expected_risk']}, got {result['risk_level']}"
        run_test(f"Predict    — {name}", _test)

    # ── 4. Feature contributions ──────────────────────────────────────────────
    print(f"\n{BOLD}[4] Feature Contribution Tests{RESET}")
    for name, tc in list(TEST_CASES.items())[:3]:  # test first 3 for speed
        def _test(tc=tc):
            contribs = get_feature_contributions(tc["patient"], model, scaler, feature_names)
            assert len(contribs) == len(feature_names)
            for fname, val in contribs:
                assert isinstance(val, float), f"Contribution not float: {val}"
        run_test(f"Contribs   — {name}", _test)

    # ── 5. Model metadata integrity ───────────────────────────────────────────
    print(f"\n{BOLD}[5] Metadata & Model Integrity Tests{RESET}")

    def test_meta_keys():
        for key in ["model_name","threshold","feature_names","metrics","all_results"]:
            assert key in meta, f"Missing meta key: {key}"
    run_test("Meta keys present", test_meta_keys)

    def test_metrics_range():
        for k in ["accuracy","recall","precision","f1","roc_auc"]:
            v = meta["metrics"].get(k, 0)
            assert 0.0 <= v <= 1.0, f"{k}={v} out of range"
    run_test("Metrics in [0,1]", test_metrics_range)

    def test_recall_target():
        recall = meta["metrics"].get("recall", 0)
        assert recall >= 0.80, f"Recall {recall:.3f} below 0.80 target"
    run_test("Recall ≥ 80%", test_recall_target)

    def test_auc_target():
        auc = meta["metrics"].get("roc_auc", 0)
        assert auc >= 0.90, f"AUC {auc:.3f} below 0.90 target"
    run_test("ROC-AUC ≥ 0.90", test_auc_target)

    def test_consistency():
        # Same input always gives same output
        patient = TEST_CASES["high_risk"]["patient"]
        r1 = predict_patient(patient, model, scaler, meta)
        r2 = predict_patient(patient, model, scaler, meta)
        assert r1["probability"] == r2["probability"], "Predictions are not deterministic"
    run_test("Deterministic predictions", test_consistency)

    # ── Summary ───────────────────────────────────────────────────────────────
    total  = results["passed"] + results["failed"]
    passed = results["passed"]
    print(f"\n{BOLD}{'='*60}{RESET}")
    if results["failed"] == 0:
        print(f"{GREEN}{BOLD}  ALL {total} TESTS PASSED ✓{RESET}")
    else:
        print(f"{YELLOW}{BOLD}  {passed}/{total} TESTS PASSED  |  {results['failed']} FAILED{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    # Detailed results
    print("Model Performance Summary:")
    m = meta.get("metrics", {})
    print(f"  Model   : {meta.get('model_name')}")
    print(f"  Recall  : {m.get('recall',0):.4f}")
    print(f"  AUC     : {m.get('roc_auc',0):.4f}")
    print(f"  Accuracy: {m.get('accuracy',0):.4f}")
    print(f"  F1      : {m.get('f1',0):.4f}")

    print("\nSample Predictions:")
    print(f"  {'Case':<20} {'Pred':<6} {'Prob':>6} {'Risk':<8}")
    print(f"  {'-'*44}")
    for name, tc in TEST_CASES.items():
        r = predict_patient(tc["patient"], model, scaler, meta)
        label = "DIABETIC" if r["prediction"] else "healthy "
        print(f"  {name:<20} {label:<8} {r['probability']:>5.1%} {r['risk_level']}")


if __name__ == "__main__":
    main()
