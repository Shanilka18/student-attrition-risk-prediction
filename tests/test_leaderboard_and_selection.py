"""Unit and integration tests for Phase 5: Leaderboard, Selection, Calibration, and Audit."""

import json
import numpy as np
import pandas as pd
import pytest

from src.evaluation.audit import evaluate_slice, run_subgroup_fairness_audit
from src.evaluation.calibration import assign_risk_band, calculate_eri, fit_calibrators
from src.evaluation.leaderboard import compute_classification_composite, load_weights_config
from src.evaluation.significance import compare_models_cv, corrected_resampled_ttest, evaluate_parsimony_rule


class TestLeaderboardScoring:
    def test_composite_scoring_calculation(self):
        val_row = {
            "pr_auc": 0.50,
            "roc_auc": 0.70,
            "f1": 0.45,
            "balanced_accuracy": 0.65,
            "brier": 0.15,
            "ece": 0.08,
        }
        train_row = {"roc_auc": 0.75}
        weights = {
            "pr_auc": 0.30,
            "roc_auc": 0.20,
            "f1": 0.15,
            "balanced_accuracy": 0.10,
            "brier_weight": 0.15,
            "ece_weight": 0.10,
            "overfit_penalty_threshold": 0.08,
            "ece_penalty_threshold": 0.10,
        }
        score, gap, pen = compute_classification_composite(
            val_row, train_row, weights, brier_min=0.10, brier_max=0.20, ece_min=0.05, ece_max=0.15
        )
        assert 0.0 <= score <= 1.0
        assert np.isclose(gap, 0.05)
        assert pen == 0.0  # Under 0.08 overfit threshold and under 0.10 ECE threshold

    def test_overfitting_penalty_trigger(self):
        val_row = {"pr_auc": 0.50, "roc_auc": 0.60, "f1": 0.4, "balanced_accuracy": 0.6, "brier": 0.15, "ece": 0.05}
        train_row = {"roc_auc": 0.85}  # Gap = 0.25 > 0.08
        weights = {
            "pr_auc": 0.30, "roc_auc": 0.20, "f1": 0.15, "balanced_accuracy": 0.10,
            "brier_weight": 0.15, "ece_weight": 0.10,
            "overfit_penalty_threshold": 0.08, "ece_penalty_threshold": 0.10
        }
        score, gap, pen = compute_classification_composite(
            val_row, train_row, weights, brier_min=0.10, brier_max=0.20, ece_min=0.05, ece_max=0.15
        )
        assert gap == 0.25
        assert pen >= (0.25 - 0.08)


class TestStatisticalSignificance:
    def test_corrected_ttest_identical_scores(self):
        diffs = np.zeros(10)
        t, p = corrected_resampled_ttest(diffs, n_train=1000, n_test=200)
        assert p == 1.0

    def test_parsimony_decision_logistic_fallback(self):
        # Differences are statistically insignificant / small
        scores_a = [0.60, 0.61, 0.59, 0.60, 0.60]
        scores_b = [0.595, 0.605, 0.592, 0.598, 0.601]
        res = evaluate_parsimony_rule("xgboost", scores_a, scores_b, parsimony_margin=0.015)
        assert res["selected_model"] == "logistic_l2"
        assert res["fallback_to_logistic"] is True


class TestCalibrationAndERI:
    def test_calibration_fitting(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.2, size=200)
        y_prob = np.clip(y_true * 0.6 + np.random.normal(0.2, 0.1, size=200), 0.01, 0.99)
        calibrator, name, metrics = fit_calibrators(y_prob, y_true)
        assert name in ["platt_sigmoid", "isotonic"]
        assert "platt_brier" in metrics

    def test_eri_threshold_labeling(self):
        probs = np.array([0.1, 0.5, 0.8])
        # Under threshold
        eri, label, is_cal = calculate_eri(probs, ece=0.05, ece_threshold=0.15)
        assert is_cal is True
        assert label == "Environmental Risk Index (ERI)"
        assert np.allclose(eri, [10.0, 50.0, 80.0])

        # Exceeds threshold
        _, label_bad, is_cal_bad = calculate_eri(probs, ece=0.20, ece_threshold=0.15)
        assert is_cal_bad is False
        assert label_bad == "normalised model-risk score"

    def test_risk_bands(self):
        assert assign_risk_band(15.0)["band"] == "Low"
        assert assign_risk_band(35.0)["band"] == "Moderate"
        assert assign_risk_band(65.0)["band"] == "High"
        assert assign_risk_band(85.0)["band"] == "Critical"


class TestFairnessAudit:
    def test_subgroup_evaluation(self):
        df_dummy = pd.DataFrame({
            "gender": ["Female", "Female", "Male", "Male"] * 10,
            "district_type": ["Urban", "Rural", "Urban", "Rural"] * 10,
            "household_income_bracket": [1, 2, 3, 4] * 10,
        })
        y_true = np.array([0, 1, 0, 1] * 10)
        y_prob = np.array([0.2, 0.7, 0.3, 0.8] * 10)

        rep = run_subgroup_fairness_audit(df_dummy, y_true, y_prob, "test_model")
        assert "gender" in rep["subgroups"]
        assert "district_type" in rep["subgroups"]
        assert "disparity_summary" in rep
