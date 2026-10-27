"""
Phase 6 Evaluation Engine Unit Tests.
Tests: metrics computation, statistical tests, provenance, leaderboard logic.

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def binary_labels():
    """Perfect-ish binary scenario: 200 samples, ~10% positive."""
    rng = np.random.default_rng(42)
    n = 200
    y_true = np.zeros(n, dtype=int)
    y_true[:20] = 1  # 10% positive rate
    return y_true


@pytest.fixture
def good_probs(binary_labels):
    """Probabilities that correlate well with ground truth."""
    rng = np.random.default_rng(42)
    probs = np.where(
        binary_labels == 1,
        rng.uniform(0.6, 0.95, size=len(binary_labels)),
        rng.uniform(0.01, 0.15, size=len(binary_labels)),
    )
    return probs.astype(np.float64)


@pytest.fixture
def random_probs(binary_labels):
    """Completely random probabilities."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 1.0, size=len(binary_labels)).astype(np.float64)


@pytest.fixture
def df_test_samples():
    """Minimal DataFrame for early-warning metric tests."""
    return pd.DataFrame({
        "sample_id": ["S1_t2", "S1_t3", "S2_t2", "S2_t3", "S3_t2"],
        "student_id": ["S1", "S1", "S2", "S2", "S3"],
        "window_t": [2, 3, 2, 3, 2],
        "target": [0, 0, 1, 1, 0],
        "last_gpa": [3.2, 3.0, 1.8, 1.5, 2.5],
    })


@pytest.fixture
def df_outcomes():
    """Minimal outcomes for early-warning tests."""
    return pd.DataFrame({
        "student_id": ["S1", "S2", "S3"],
        "attrited": [0, 1, 0],
        "attrition_semester": [np.nan, 4.0, np.nan],
        "last_semester_observed": [8, 4, 8],
    })


@pytest.fixture
def df_boot_samples():
    """Larger DataFrame for bootstrap tests — multiple students, multiple semesters."""
    rng = np.random.default_rng(42)
    rows = []
    for sid_num in range(50):
        sid = f"STU_{sid_num:04d}"
        for t in range(2, 5):
            rows.append({
                "sample_id": f"{sid}_t{t}",
                "student_id": sid,
                "window_t": t,
                "target": 1 if sid_num < 5 else 0,  # 10% positive
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
#  Group 1: Metrics Computation
# ──────────────────────────────────────────────────────────────────────────────


class TestExpectedCalibrationError:
    """Tests for ECE computation."""

    def test_ece_perfect_calibration(self):
        """ECE should be zero when predicted probabilities perfectly match labels."""
        from backend.src.evaluation.metrics import compute_expected_calibration_error

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ece, _ = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece < 0.01, f"ECE should be near zero for perfect calibration, got {ece}"

    def test_ece_worst_calibration(self):
        """ECE should be high when all probs are ~1.0 but many labels are 0."""
        from backend.src.evaluation.metrics import compute_expected_calibration_error

        y_true = np.array([0] * 80 + [1] * 20)
        y_prob = np.array([0.95] * 100)
        ece, _ = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece > 0.5, f"ECE should be high for badly calibrated probs, got {ece}"


class TestComprehensiveMetrics:
    """Tests for the full metric suite."""

    def test_returns_all_sections(self, binary_labels, good_probs):
        """Result dict should have all four metric categories."""
        from backend.src.evaluation.metrics import compute_comprehensive_metrics

        result = compute_comprehensive_metrics(binary_labels, good_probs, threshold=0.5)
        assert "classification" in result
        assert "risk_index_quality" in result
        assert "early_warning" in result
        assert "efficiency" in result

    def test_perfect_classifier_high_auc(self):
        """Near-perfect predictions should yield ROC-AUC close to 1.0."""
        from backend.src.evaluation.metrics import compute_comprehensive_metrics

        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.01, 0.02, 0.03, 0.04, 0.96, 0.97, 0.98, 0.99])
        result = compute_comprehensive_metrics(y_true, y_prob, threshold=0.5)
        assert result["classification"]["roc_auc"] > 0.99, "Perfect classifier should have AUC near 1.0"

    def test_random_classifier_auc_near_half(self, binary_labels, random_probs):
        """Random predictions should yield ROC-AUC near 0.5."""
        from backend.src.evaluation.metrics import compute_comprehensive_metrics

        result = compute_comprehensive_metrics(binary_labels, random_probs, threshold=0.5)
        auc = result["classification"]["roc_auc"]
        assert 0.35 < auc < 0.65, f"Random classifier AUC should be near 0.5, got {auc}"

    def test_brier_skill_score_positive(self, binary_labels, good_probs):
        """Good predictions should yield positive Brier Skill Score."""
        from backend.src.evaluation.metrics import compute_comprehensive_metrics

        result = compute_comprehensive_metrics(binary_labels, good_probs, threshold=0.5)
        bss = result["risk_index_quality"]["brier_skill_score"]
        assert bss > 0, f"Good predictions should have positive BSS, got {bss}"

    def test_pseudo_r2_bounded(self, binary_labels, good_probs):
        """Efron's pseudo-R² should be ≤ 1.0."""
        from backend.src.evaluation.metrics import compute_comprehensive_metrics

        result = compute_comprehensive_metrics(binary_labels, good_probs, threshold=0.5)
        pr2 = result["risk_index_quality"]["pseudo_r2"]
        assert pr2 <= 1.0, f"Pseudo-R² should be ≤ 1.0, got {pr2}"

    def test_f2_closer_to_recall_than_precision(self, binary_labels, good_probs):
        """F2 (beta=2) emphasizes recall; it should be closer to recall than precision."""
        from backend.src.evaluation.metrics import compute_comprehensive_metrics

        result = compute_comprehensive_metrics(binary_labels, good_probs, threshold=0.3)
        clf = result["classification"]
        dist_to_recall = abs(clf["f2_score"] - clf["recall"])
        dist_to_prec = abs(clf["f2_score"] - clf["precision"])
        assert dist_to_recall <= dist_to_prec, (
            f"F2 should be closer to recall ({clf['recall']}) than precision ({clf['precision']}), "
            f"but F2={clf['f2_score']}"
        )


class TestEarlyWarningMetrics:
    """Tests for lead time and top-k metrics."""

    def test_lead_time_calculation(self, df_test_samples, df_outcomes):
        """Lead time for known attrition_semester=4, first flagged at window_t=2, should be 2."""
        from backend.src.evaluation.metrics import compute_early_warning_metrics

        # S2 attrites at semester 4; flag all S2 samples
        y_prob = np.array([0.1, 0.1, 0.9, 0.9, 0.1])  # S2 flagged at t=2
        result = compute_early_warning_metrics(df_test_samples, df_outcomes, y_prob, threshold=0.5)
        assert result["average_lead_time_semesters"] == 2.0, (
            f"Expected lead time 2.0, got {result['average_lead_time_semesters']}"
        )

    def test_recall_at_top_k(self, df_test_samples, df_outcomes):
        """If highest-prob samples are the true positives, recall@top should be high."""
        from backend.src.evaluation.metrics import compute_early_warning_metrics

        # 5 samples, 2 positives. If top-2 probs are on positives, recall@top20% (=1 sample) captures 1/2
        y_prob = np.array([0.1, 0.1, 0.95, 0.9, 0.05])  # targets 2,3 are positive
        result = compute_early_warning_metrics(df_test_samples, df_outcomes, y_prob, threshold=0.5)
        assert result["recall_at_top_20_pct"] > 0.0, "Should flag at least some positives in top 20%"


# ──────────────────────────────────────────────────────────────────────────────
#  Group 2: Statistical Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestGroupedBootstrapCIs:
    """Tests for student-grouped bootstrap confidence intervals."""

    def test_ci_structure(self, df_boot_samples, good_probs):
        """Bootstrap result should have correct keys and subkeys."""
        from backend.src.evaluation.statistical_tests import compute_grouped_bootstrap_cis

        y_true = df_boot_samples["target"].values
        n = len(y_true)
        probs = np.where(y_true == 1, 0.8, 0.1)[:n]

        result = compute_grouped_bootstrap_cis(
            df_boot_samples, y_true, probs, threshold=0.5, n_resamples=50, seed=42,
        )
        for metric in ["roc_auc", "pr_auc", "f1_score", "f2_score", "brier_score"]:
            assert metric in result, f"Missing metric key: {metric}"
            for subkey in ["mean", "ci_lower", "ci_upper", "std"]:
                assert subkey in result[metric], f"Missing subkey {subkey} in {metric}"

    def test_ci_contains_point_estimate(self, df_boot_samples):
        """Point estimate should approximately fall within the 95% CI."""
        from backend.src.evaluation.statistical_tests import compute_grouped_bootstrap_cis
        from sklearn.metrics import roc_auc_score

        y_true = df_boot_samples["target"].values
        rng = np.random.default_rng(42)
        probs = np.where(y_true == 1, rng.uniform(0.6, 0.9, len(y_true)), rng.uniform(0.05, 0.3, len(y_true)))
        point_est = roc_auc_score(y_true, probs)

        result = compute_grouped_bootstrap_cis(
            df_boot_samples, y_true, probs, threshold=0.5, n_resamples=200, seed=42,
        )
        ci = result["roc_auc"]
        # Allow slight margin for grouped bootstrap variance
        assert ci["ci_lower"] - 0.1 <= point_est <= ci["ci_upper"] + 0.1, (
            f"Point estimate {point_est:.4f} should be near CI [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
        )


class TestPairedBootstrapTest:
    """Tests for paired bootstrap hypothesis test."""

    def test_identical_models_not_significant(self, df_boot_samples):
        """Two identical probability vectors should yield p ≈ 1.0."""
        from backend.src.evaluation.statistical_tests import paired_bootstrap_test

        y_true = df_boot_samples["target"].values
        probs = np.where(y_true == 1, 0.8, 0.1)

        result = paired_bootstrap_test(
            df_boot_samples, y_true, probs, probs, n_resamples=100, seed=42,
        )
        assert result["p_value"] >= 0.5, (
            f"Identical models should have high p-value, got {result['p_value']}"
        )
        assert not result["statistically_significant"]

    def test_very_different_models_significant(self, df_boot_samples):
        """One good model and one terrible model should be significantly different."""
        from backend.src.evaluation.statistical_tests import paired_bootstrap_test

        y_true = df_boot_samples["target"].values
        good = np.where(y_true == 1, 0.95, 0.05)
        bad = np.full_like(y_true, 0.5, dtype=float)

        result = paired_bootstrap_test(
            df_boot_samples, y_true, good, bad, n_resamples=200, seed=42,
        )
        assert result["p_value"] < 0.05, (
            f"Very different models should have p < 0.05, got {result['p_value']}"
        )


class TestMcNemarTest:
    """Tests for McNemar's paired test."""

    def test_identical_predictions_not_significant(self):
        """Identical binary predictions should yield p = 1.0."""
        from backend.src.evaluation.statistical_tests import mcnemar_test

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        result = mcnemar_test(y_true, y_pred, y_pred)
        assert result["p_value"] == 1.0
        assert not result["statistically_significant"]

    def test_different_predictions_detected(self):
        """Large systematic disagreement should be detectable."""
        from backend.src.evaluation.statistical_tests import mcnemar_test

        n = 200
        y_true = np.zeros(n, dtype=int)
        y_true[:40] = 1
        # Model A gets first 20 positive right, Model B gets last 20 right
        y_pred_a = np.zeros(n, dtype=int)
        y_pred_a[:20] = 1
        y_pred_b = np.zeros(n, dtype=int)
        y_pred_b[20:40] = 1
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        # There should be discordant pairs
        assert result["discordant_a_correct"] + result["discordant_b_correct"] > 0


# ──────────────────────────────────────────────────────────────────────────────
#  Group 3: Provenance
# ──────────────────────────────────────────────────────────────────────────────


class TestProvenance:
    """Tests for cryptographic test-set provenance logging."""

    def test_creates_audit_file(self, tmp_path):
        """Provenance function should create a JSON file with SHA-256 fingerprint."""
        from backend.src.evaluation.provenance import record_test_evaluation_provenance

        df = pd.DataFrame({
            "sample_id": ["S1_t2", "S2_t2"],
            "target": [0, 1],
        })
        output = tmp_path / "prov.json"
        result = record_test_evaluation_provenance(output, df, ["S1", "S2"])
        assert output.exists()
        assert "current_fingerprint" in result
        assert len(result["current_fingerprint"]) == 64  # SHA-256 hex

    def test_increments_evaluation_count(self, tmp_path):
        """Running provenance twice should increment the evaluation count to 2."""
        from backend.src.evaluation.provenance import record_test_evaluation_provenance

        df = pd.DataFrame({"sample_id": ["S1_t2"], "target": [0]})
        output = tmp_path / "prov.json"
        record_test_evaluation_provenance(output, df, ["S1"])
        result2 = record_test_evaluation_provenance(output, df, ["S1"])
        assert result2["total_test_evaluations"] == 2

    def test_hash_deterministic(self, tmp_path):
        """Same data should produce the same SHA-256 hash."""
        from backend.src.evaluation.provenance import record_test_evaluation_provenance

        df = pd.DataFrame({"sample_id": ["S1_t2", "S2_t2"], "target": [0, 1]})
        r1 = record_test_evaluation_provenance(tmp_path / "p1.json", df, ["S1", "S2"])
        r2 = record_test_evaluation_provenance(tmp_path / "p2.json", df, ["S1", "S2"])
        assert r1["current_fingerprint"] == r2["current_fingerprint"]

    def test_hash_changes_with_data(self, tmp_path):
        """Different data should produce different hashes."""
        from backend.src.evaluation.provenance import record_test_evaluation_provenance

        df1 = pd.DataFrame({"sample_id": ["S1_t2"], "target": [0]})
        df2 = pd.DataFrame({"sample_id": ["S1_t2"], "target": [1]})
        r1 = record_test_evaluation_provenance(tmp_path / "p1.json", df1, ["S1"])
        r2 = record_test_evaluation_provenance(tmp_path / "p2.json", df2, ["S1"])
        assert r1["current_fingerprint"] != r2["current_fingerprint"]


# ──────────────────────────────────────────────────────────────────────────────
#  Group 4: Leaderboard Logic
# ──────────────────────────────────────────────────────────────────────────────


def _make_mock_evaluation_results(overrides: Dict[str, Dict] = None) -> Dict[str, Any]:
    """Builds a mock evaluation_results dict for leaderboard testing."""
    base_models = {
        "gru": {
            "model_key": "gru",
            "category": "E",
            "label": "GRU (Proposal Model)",
            "is_proposal_model": True,
            "is_sequence_model": True,
            "threshold": 0.05,
            "seed_std": 0.004,
            "validation_roc_auc": 0.82,
            "metrics": {
                "classification": {
                    "roc_auc": 0.83, "pr_auc": 0.22, "f1_score": 0.24,
                    "f2_score": 0.35, "recall": 0.82, "precision": 0.14,
                    "balanced_accuracy": 0.75, "matthews_corrcoef": 0.25,
                    "accuracy": 0.68, "specificity": 0.67, "log_loss": 0.3,
                    "cohen_kappa": 0.18,
                    "confusion_matrix": {"tn": 2100, "fp": 1000, "fn": 35, "tp": 175},
                },
                "risk_index_quality": {
                    "brier_score": 0.053, "brier_skill_score": 0.12,
                    "expected_calibration_error": 0.04, "pseudo_r2": 0.15,
                    "spearman_correlation": 0.45, "mae": 0.15, "rmse": 0.22,
                    "spearman_pvalue": 0.001, "calibration_curve": [],
                },
                "early_warning": {
                    "average_lead_time_semesters": 2.5,
                    "recall_at_top_10_pct": 0.80,
                    "recall_at_top_20_pct": 0.92,
                    "detected_before_gpa_below_2_rate": 0.55,
                    "lead_time_distribution": [1, 2, 3, 2, 3],
                },
                "efficiency": {
                    "inference_time_ms_per_student": 15.0,
                    "model_size_kb": 150.0,
                },
            },
        },
        "catboost": {
            "model_key": "catboost",
            "category": "C",
            "label": "CatBoost",
            "is_proposal_model": False,
            "is_sequence_model": False,
            "threshold": 0.05,
            "seed_std": 0.002,
            "validation_roc_auc": 0.81,
            "metrics": {
                "classification": {
                    "roc_auc": 0.81, "pr_auc": 0.20, "f1_score": 0.22,
                    "f2_score": 0.32, "recall": 0.80, "precision": 0.13,
                    "balanced_accuracy": 0.73, "matthews_corrcoef": 0.22,
                    "accuracy": 0.66, "specificity": 0.65, "log_loss": 0.32,
                    "cohen_kappa": 0.16,
                    "confusion_matrix": {"tn": 2050, "fp": 1060, "fn": 40, "tp": 160},
                },
                "risk_index_quality": {
                    "brier_score": 0.055, "brier_skill_score": 0.10,
                    "expected_calibration_error": 0.05, "pseudo_r2": 0.12,
                    "spearman_correlation": 0.42, "mae": 0.16, "rmse": 0.23,
                    "spearman_pvalue": 0.002, "calibration_curve": [],
                },
                "early_warning": {
                    "average_lead_time_semesters": 2.0,
                    "recall_at_top_10_pct": 0.75,
                    "recall_at_top_20_pct": 0.88,
                    "detected_before_gpa_below_2_rate": 0.50,
                    "lead_time_distribution": [1, 2, 2, 3],
                },
                "efficiency": {
                    "inference_time_ms_per_student": 5.0,
                    "model_size_kb": 500.0,
                },
            },
        },
        "gpa_threshold": {
            "model_key": "gpa_threshold",
            "category": "A",
            "label": "GPA Threshold Rule (Reference)",
            "is_proposal_model": False,
            "is_sequence_model": False,
            "threshold": 0.01,
            "seed_std": 0.0,
            "validation_roc_auc": 0.51,
            "metrics": {
                "classification": {
                    "roc_auc": 0.51, "pr_auc": 0.07, "f1_score": 0.12,
                    "f2_score": 0.18, "recall": 1.0, "precision": 0.06,
                    "balanced_accuracy": 0.50, "matthews_corrcoef": 0.02,
                    "accuracy": 0.06, "specificity": 0.0, "log_loss": 0.7,
                    "cohen_kappa": 0.01,
                    "confusion_matrix": {"tn": 0, "fp": 3100, "fn": 0, "tp": 210},
                },
                "risk_index_quality": {
                    "brier_score": 0.058, "brier_skill_score": 0.03,
                    "expected_calibration_error": 0.5, "pseudo_r2": -0.1,
                    "spearman_correlation": 0.1, "mae": 0.4, "rmse": 0.5,
                    "spearman_pvalue": 0.1, "calibration_curve": [],
                },
                "early_warning": {
                    "average_lead_time_semesters": 0.5,
                    "recall_at_top_10_pct": 0.10,
                    "recall_at_top_20_pct": 0.20,
                    "detected_before_gpa_below_2_rate": 0.0,
                    "lead_time_distribution": [0, 1],
                },
                "efficiency": {
                    "inference_time_ms_per_student": 0.1,
                    "model_size_kb": 1.0,
                },
            },
        },
    }

    if overrides:
        for key, vals in overrides.items():
            if key in base_models:
                _deep_update(base_models[key], vals)
            else:
                base_models[key] = vals

    return {
        "models": base_models,
        "serializable_models": base_models,
        "statistical_tests": {
            "paired_bootstrap_seq_vs_boost": {
                "model_a": "gru",
                "model_b": "catboost",
                "roc_auc": {
                    "observed_difference": 0.02,
                    "p_value": 0.03,
                    "statistically_significant": True,
                },
                "pr_auc": {
                    "observed_difference": 0.02,
                    "p_value": 0.04,
                    "statistically_significant": True,
                },
            },
        },
        "key_models": {
            "sequence_champion": "gru",
            "boosting_baseline": "catboost",
            "static_best": None,
            "overall_best": "gru",
        },
        "n_test_students": 750,
        "n_test_samples": 3310,
    }


def _deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


class TestLeaderboardRanking:
    """Tests for leaderboard construction and ranking."""

    def test_ranking_order(self):
        """Model with best metrics should rank #1 in the leaderboard."""
        from backend.src.evaluation.leaderboard import build_leaderboard

        results = _make_mock_evaluation_results()
        lb = build_leaderboard(results)
        df = lb["leaderboard_df"]
        top = df.iloc[0]
        # GRU has better ROC-AUC, PR-AUC, F2 than catboost and gpa_threshold
        assert top["model_key"] == "gru", f"Expected GRU as top-ranked, got {top['model_key']}"

    def test_eligibility_nfr1_constraint(self):
        """Model exceeding 2s inference should be ineligible."""
        from backend.src.evaluation.leaderboard import build_leaderboard

        results = _make_mock_evaluation_results({
            "gru": {"metrics": {"efficiency": {"inference_time_ms_per_student": 3000.0, "model_size_kb": 150.0}}}
        })
        lb = build_leaderboard(results)
        df = lb["leaderboard_df"]
        gru_row = df[df["model_key"] == "gru"].iloc[0]
        assert not gru_row["eligible"], "GRU with 3s latency should be ineligible"

    def test_seed_std_constraint(self):
        """Model with seed_std > 0.05 should be ineligible."""
        from backend.src.evaluation.leaderboard import build_leaderboard

        results = _make_mock_evaluation_results({
            "gru": {"seed_std": 0.08}
        })
        lb = build_leaderboard(results)
        df = lb["leaderboard_df"]
        gru_row = df[df["model_key"] == "gru"].iloc[0]
        assert not gru_row["eligible"], "GRU with seed_std=0.08 should be ineligible"

    def test_champion_is_top_eligible(self):
        """Champion should be the top eligible model."""
        from backend.src.evaluation.leaderboard import build_leaderboard

        results = _make_mock_evaluation_results()
        lb = build_leaderboard(results)
        champion = lb["champion"]
        assert champion is not None
        assert champion["model_key"] == "gru"

    def test_gpa_threshold_never_champion(self):
        """Reference-only baseline must never be selected as champion."""
        from backend.src.evaluation.leaderboard import build_leaderboard

        # Give gpa_threshold impossibly good metrics
        results = _make_mock_evaluation_results({
            "gpa_threshold": {
                "metrics": {
                    "classification": {
                        "roc_auc": 0.99, "pr_auc": 0.99, "f1_score": 0.99,
                        "f2_score": 0.99, "recall": 0.99, "precision": 0.99,
                        "balanced_accuracy": 0.99, "matthews_corrcoef": 0.99,
                        "accuracy": 0.99, "specificity": 0.99, "log_loss": 0.01,
                        "cohen_kappa": 0.99,
                        "confusion_matrix": {"tn": 3000, "fp": 10, "fn": 1, "tp": 300},
                    },
                    "risk_index_quality": {
                        "brier_score": 0.001, "brier_skill_score": 0.99,
                        "expected_calibration_error": 0.001, "pseudo_r2": 0.99,
                        "spearman_correlation": 0.99, "mae": 0.01, "rmse": 0.01,
                        "spearman_pvalue": 0.0001, "calibration_curve": [],
                    },
                    "early_warning": {
                        "average_lead_time_semesters": 5.0,
                        "recall_at_top_10_pct": 1.0,
                        "recall_at_top_20_pct": 1.0,
                        "detected_before_gpa_below_2_rate": 1.0,
                        "lead_time_distribution": [5, 5, 5],
                    },
                    "efficiency": {
                        "inference_time_ms_per_student": 0.01,
                        "model_size_kb": 0.5,
                    },
                },
            },
        })
        lb = build_leaderboard(results)
        champion = lb["champion"]
        assert champion["model_key"] != "gpa_threshold", (
            "gpa_threshold is reference-only and must never be champion"
        )


class TestResearchQuestion:
    """Tests for research question adjudication logic."""

    def test_positive_answer(self):
        """When sequence champion beats boosting significantly -> POSITIVE."""
        from backend.src.evaluation.leaderboard import build_leaderboard

        results = _make_mock_evaluation_results()
        lb = build_leaderboard(results)
        rq = lb["research_question_answer"]
        assert rq["answer"] == "POSITIVE"

    def test_null_answer_when_boost_wins(self):
        """When boosting baseline beats sequence model -> NULL."""
        from backend.src.evaluation.leaderboard import build_leaderboard

        results = _make_mock_evaluation_results({
            "gru": {
                "metrics": {
                    "classification": {
                        "roc_auc": 0.78, "pr_auc": 0.18, "f1_score": 0.20,
                        "f2_score": 0.30, "recall": 0.78, "precision": 0.12,
                        "balanced_accuracy": 0.70, "matthews_corrcoef": 0.18,
                        "accuracy": 0.64, "specificity": 0.62, "log_loss": 0.35,
                        "cohen_kappa": 0.14,
                        "confusion_matrix": {"tn": 1950, "fp": 1150, "fn": 45, "tp": 165},
                    },
                },
            },
        })
        # Also update stat tests to show non-significant
        results["statistical_tests"]["paired_bootstrap_seq_vs_boost"]["roc_auc"]["observed_difference"] = -0.03
        results["statistical_tests"]["paired_bootstrap_seq_vs_boost"]["roc_auc"]["p_value"] = 0.5
        results["statistical_tests"]["paired_bootstrap_seq_vs_boost"]["roc_auc"]["statistically_significant"] = False
        results["statistical_tests"]["paired_bootstrap_seq_vs_boost"]["pr_auc"]["observed_difference"] = -0.02
        results["statistical_tests"]["paired_bootstrap_seq_vs_boost"]["pr_auc"]["p_value"] = 0.4
        results["statistical_tests"]["paired_bootstrap_seq_vs_boost"]["pr_auc"]["statistically_significant"] = False

        lb = build_leaderboard(results)
        rq = lb["research_question_answer"]
        assert rq["answer"] == "NULL", f"Expected NULL when boost wins, got {rq['answer']}"

    def test_synthetic_data_label(self):
        """Research question answer must always include SYNTHETIC data label."""
        from backend.src.evaluation.leaderboard import build_leaderboard

        results = _make_mock_evaluation_results()
        lb = build_leaderboard(results)
        rq = lb["research_question_answer"]
        assert rq["data_source"] == "SYNTHETIC"
        assert "SYNTHETIC" in rq.get("honest_assessment", "")
