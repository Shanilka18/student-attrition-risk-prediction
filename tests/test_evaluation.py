"""Unit and integration tests for Phase 5: Evaluation Metrics and Leaderboard.

Tests:
    - compute_regression_metrics: correctness on known arrays.
    - compute_classification_metrics: all expected keys present; per-class recall.
    - compute_ece: boundary values.
    - wilcoxon_comparison: both significant and non-significant cases.
    - _composite_rank: monotonicity with single-metric variance.
    - run_evaluation (smoke): runs without crashing on Phase 4 artifacts (val-only).
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    compute_ece,
    compute_multiclass_ece,
    wilcoxon_comparison,
    RISK_LABELS,
)
from src.evaluation.leaderboard import _composite_rank


# ---------------------------------------------------------------------------
# Regression metric tests
# ---------------------------------------------------------------------------

class TestRegressionMetrics:
    def test_perfect_prediction(self) -> None:
        y = np.linspace(0, 100, 50)
        metrics = compute_regression_metrics(y, y, n_features=5)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-5)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-5)
        assert metrics["pearson_r"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["spearman_r"] == pytest.approx(1.0, abs=1e-5)
        assert metrics[f"within_pm10_acc"] == pytest.approx(1.0, abs=1e-5)

    def test_all_keys_present(self) -> None:
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 100, 100)
        y_pred = y_true + rng.normal(0, 5, 100)
        metrics = compute_regression_metrics(y_true, y_pred, n_features=5)
        required = [
            "r2", "adj_r2", "rmse", "mae", "median_ae",
            "smape", "explained_variance", "pearson_r",
            "spearman_r", "within_pm10_acc", "primary", "tiebreak",
        ]
        for key in required:
            assert key in metrics, f"Missing key: {key}"

    def test_within_10_acc_deterministic(self) -> None:
        y_true = np.array([50.0, 50.0, 50.0, 50.0])
        y_pred = np.array([45.0, 55.0, 60.1, 80.0])
        # First 3 within ±10 (|50-45|=5, |50-55|=5, |50-60.1|=10.1 > 10)
        metrics = compute_regression_metrics(y_true, y_pred, n_features=1)
        # 45 and 55 are within ±10 → 2/4 = 0.5
        assert metrics["within_pm10_acc"] == pytest.approx(0.5, abs=1e-5)

    def test_adj_r2_less_than_r2(self) -> None:
        rng = np.random.default_rng(1)
        y_true = rng.uniform(0, 100, 200)
        y_pred = y_true + rng.normal(0, 8, 200)
        m = compute_regression_metrics(y_true, y_pred, n_features=15)
        assert m["adj_r2"] is not None
        assert m["adj_r2"] <= m["r2"]


# ---------------------------------------------------------------------------
# Classification metric tests
# ---------------------------------------------------------------------------

class TestClassificationMetrics:
    def _make_data(
        self, n: int = 100, seed: int = 0
    ):
        rng = np.random.default_rng(seed)
        y_true = rng.choice(RISK_LABELS, size=n)
        # 80% correct, 20% random error
        y_pred = y_true.copy()
        idx_wrong = rng.choice(n, size=int(n * 0.2), replace=False)
        y_pred[idx_wrong] = rng.choice(RISK_LABELS, size=len(idx_wrong))
        # Fake prob matrix
        y_prob = rng.dirichlet(np.ones(3), size=n)
        return y_true, y_pred, y_prob

    def test_all_keys_present(self) -> None:
        y_true, y_pred, y_prob = self._make_data()
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        required = [
            "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
            "high_risk_recall", "mcc", "quadratic_weighted_kappa",
            "confusion_matrix", "roc_auc_ovr_macro", "pr_auc_macro",
            "log_loss", "brier_score", "ece", "primary", "tiebreak",
        ]
        for key in required:
            assert key in metrics, f"Missing key: {key}"

    def test_high_risk_recall_is_recall_for_high_class(self) -> None:
        y_true, y_pred, y_prob = self._make_data(n=300, seed=7)
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        per_class = metrics["per_class_recall"]
        assert metrics["high_risk_recall"] == pytest.approx(per_class["High"], abs=1e-6)

    def test_perfect_classification(self) -> None:
        y = np.array(["Low"] * 33 + ["Moderate"] * 34 + ["High"] * 33)
        metrics = compute_classification_metrics(y, y)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)
        assert metrics["mcc"] == pytest.approx(1.0)

    def test_confusion_matrix_shape(self) -> None:
        y_true, y_pred, _ = self._make_data()
        metrics = compute_classification_metrics(y_true, y_pred)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_roc_auc_in_range(self) -> None:
        y_true, y_pred, y_prob = self._make_data(n=300)
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        if metrics.get("roc_auc_ovr_macro") is not None:
            assert 0.0 <= metrics["roc_auc_ovr_macro"] <= 1.0


# ---------------------------------------------------------------------------
# ECE tests
# ---------------------------------------------------------------------------

class TestECE:
    def test_perfect_calibration(self) -> None:
        # Perfectly calibrated: prob 0.9 → 90% correct
        y_true = np.array([1] * 9 + [0] * 1)
        y_prob = np.array([0.9] * 10)
        ece = compute_ece(y_true, y_prob, n_bins=5)
        assert ece < 0.05

    def test_maximally_miscalibrated(self) -> None:
        # Confident wrong: predicts ~1.0, almost all incorrect
        rng = np.random.default_rng(42)
        y_true = np.zeros(100, dtype=int)
        y_prob = rng.uniform(0.85, 0.99, size=100)  # high confidence but all wrong
        ece = compute_ece(y_true, y_prob, n_bins=10)
        assert ece > 0.3  # ECE must be substantial



# ---------------------------------------------------------------------------
# Statistical comparison tests
# ---------------------------------------------------------------------------

class TestWilcoxon:
    def test_identical_models_not_significant(self) -> None:
        scores = np.array([0.85, 0.86, 0.84, 0.85, 0.86])
        _, p = wilcoxon_comparison(scores, scores)
        # Identical arrays → Wilcoxon returns p=1.0
        assert p >= 0.05 or np.isnan(p)  # allow nan for zero-difference case

    def test_very_different_models_significant(self) -> None:
        # Need more observations for Wilcoxon to reach significance
        rng = np.random.default_rng(0)
        scores_a = rng.uniform(0.88, 0.95, size=15)  # consistently high
        scores_b = rng.uniform(0.45, 0.55, size=15)  # consistently low
        _, p = wilcoxon_comparison(scores_a, scores_b)
        assert p < 0.05



# ---------------------------------------------------------------------------
# Composite rank test
# ---------------------------------------------------------------------------

class TestCompositeRank:
    def test_rank_is_series_correct_length(self) -> None:
        data = pd.DataFrame({
            "model_name": ["ridge", "random_forest", "dummy"],
            "val_primary_metric": [0.99, 0.95, 0.0],
            "overfitting_gap": [0.1, 0.3, 0.0],
            "latency_ms": [1.0, 30.0, 0.1],
        })
        weights = {
            "accuracy": 0.35, "calibration": 0.20,
            "stability": 0.20, "explainability": 0.10, "latency": 0.15,
        }
        ranks = _composite_rank(data, "regression", weights)
        assert len(ranks) == 3
        assert all(isinstance(v, float) for v in ranks)


# ---------------------------------------------------------------------------
# Smoke test: run_evaluation (validation-only, no test set)
# ---------------------------------------------------------------------------

class TestRunEvaluationSmoke:
    """Smoke test that the full evaluation pipeline runs on existing Phase 4 artifacts."""

    def test_evaluation_smoke_no_test_set(self) -> None:
        """Full pipeline smoke test using validation set only (no test-set touch)."""
        from src.evaluation.leaderboard import run_evaluation
        results = run_evaluation(
            tracks=["regression", "classification"],
            seed=42,
            run_test_set=False,
        )
        assert isinstance(results, dict)
        for track in ["regression", "classification"]:
            if track in results:
                out = results[track]
                assert "leaderboard" in out
                assert "champion" in out
                lb = out["leaderboard"]
                assert len(lb) > 0
                assert "model_name" in lb.columns
                assert "composite_rank" in lb.columns
                champ = out["champion"]
                assert "champion" in champ
                assert "justification" in champ
