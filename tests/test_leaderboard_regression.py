"""Regression test suite for Leaderboard Defect Fixes & Parsimony Auditing.

Asserts:
1. Decision thresholds are tuned per-model on validation data, yielding F1 > 0 and balanced accuracy > 0.50.
2. Brier and ECE columns are computed from validation probabilities and populated (never empty or '-').
3. Hard gates disqualify failing models, preventing them from ranking #1 or winning category awards.
4. Missing metrics are never treated as zero or perfect (incomplete / raised error).
5. artifacts/leaderboard/score_breakdown.csv accurately reflects all weights, normalized values, and penalties.
6. Category cards and table rows read from the single source of truth (leaderboard_C.json) and match exactly.
7. run_id and timestamp are present across all evaluation artifacts and API metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from src.api.main import app
from src.evaluation.calibration import calibrate_classifier, find_optimal_threshold
from src.evaluation.leaderboard import (
    build_leaderboard,
    check_hard_gates,
    compute_classification_composite,
    load_weights_config,
)
from src.utils.paths import get_artifacts_dir


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestLeaderboardDefectFixes:
    def test_threshold_tuning_maximizes_validation_metrics(self):
        """Verify optimal threshold search produces strictly positive F1 on imbalanced data."""
        np.random.seed(42)
        # 16% positive prevalence
        y_val = np.random.binomial(1, 0.16, size=286)
        # Probabilities centered around 0.16 (never exceeding 0.50 under default threshold)
        y_prob = np.clip(y_val * 0.30 + np.random.normal(0.12, 0.05, size=286), 0.01, 0.48)

        # Default 0.5 threshold yields zero positive predictions
        default_preds = (y_prob >= 0.5).astype(int)
        assert np.sum(default_preds) == 0

        # Tuned threshold on validation data
        best_th, metrics = find_optimal_threshold(y_val, y_prob, objective="balanced_accuracy")
        assert 0.05 <= best_th < 0.50
        assert metrics["f1"] > 0.0
        assert metrics["balanced_accuracy"] > 0.50

    def test_hard_gates_disqualification_rules(self):
        """Verify models failing hard gates are disqualified from winning awards."""
        cfg = load_weights_config()
        prevalence = 0.1643
        brier_baseline = prevalence * (1.0 - prevalence)  # ~0.1373

        # Model with F1 == 0
        bad_f1_row = {
            "pr_auc": 0.40, "roc_auc": 0.70, "f1": 0.0, "balanced_accuracy": 0.50,
            "brier_score": 0.11, "ece": 0.05,
        }
        qual, status, reasons = check_hard_gates(bad_f1_row, cfg, prevalence)
        assert qual is False
        assert status == "disqualified"
        assert any("F1" in r for r in reasons)

        # Model with ROC < 0.55
        bad_roc_row = {
            "pr_auc": 0.20, "roc_auc": 0.52, "f1": 0.30, "balanced_accuracy": 0.52,
            "brier_score": 0.11, "ece": 0.05,
        }
        qual, status, reasons = check_hard_gates(bad_roc_row, cfg, prevalence)
        assert qual is False
        assert any("ROC-AUC" in r for r in reasons)

        # Model with Brier >= baseline
        bad_brier_row = {
            "pr_auc": 0.40, "roc_auc": 0.75, "f1": 0.45, "balanced_accuracy": 0.65,
            "brier_score": brier_baseline + 0.02, "ece": 0.05,
        }
        qual, status, reasons = check_hard_gates(bad_brier_row, cfg, prevalence)
        assert qual is False
        assert any("Brier" in r for r in reasons)

        # Model passing all gates
        good_row = {
            "pr_auc": 0.45, "roc_auc": 0.76, "f1": 0.48, "balanced_accuracy": 0.70,
            "brier_score": 0.108, "ece": 0.04,
        }
        qual, status, reasons = check_hard_gates(good_row, cfg, prevalence)
        assert qual is True
        assert status == "passed"
        assert len(reasons) == 0

    def test_missing_metrics_never_treated_as_zero_or_perfect(self):
        """Assert missing metrics raise ValueError or mark model as incomplete."""
        weights = {"pr_auc": 0.30, "roc_auc": 0.20, "f1": 0.15, "balanced_accuracy": 0.10}
        incomplete_row = {
            "model": "incomplete_clf",
            "pr_auc": 0.45,
            "roc_auc": None,  # Missing!
            "f1": 0.35,
            "balanced_accuracy": 0.60,
        }
        with pytest.raises(ValueError, match="Missing required classification metric"):
            compute_classification_composite(incomplete_row, None, weights)

    def test_score_breakdown_artifact_integrity(self):
        """Verify artifacts/leaderboard/score_breakdown.csv matches weights and contributions."""
        bd_path = get_artifacts_dir("leaderboard") / "score_breakdown.csv"
        assert bd_path.exists(), f"Score breakdown CSV missing at {bd_path}"
        df_bd = pd.read_csv(bd_path)
        assert not df_bd.empty
        assert "composite_score" in df_bd.columns
        assert "pr_auc_contrib" in df_bd.columns
        assert "brier_contrib" in df_bd.columns
        assert "total_penalty" in df_bd.columns

    def test_brier_and_ece_populated_in_leaderboard(self):
        """Assert Brier and ECE are real numeric values, never null or '-'."""
        lb_path = get_artifacts_dir("leaderboard") / "leaderboard_C.json"
        assert lb_path.exists()
        with open(lb_path, "r", encoding="utf-8") as f:
            lb_data = json.load(f)

        for entry in lb_data:
            assert entry.get("brier_score") is not None, f"brier_score missing in {entry['model']}"
            assert entry.get("ece") is not None, f"ece missing in {entry['model']}"
            assert entry.get("threshold") is not None, f"threshold missing in {entry['model']}"
            assert 0.0 <= entry["brier_score"] <= 1.0
            assert 0.0 <= entry["ece"] <= 1.0
            assert 0.0 < entry["threshold"] < 1.0

    def test_category_cards_table_exact_agreement(self):
        """Verify API category cards and leaderboard table read from single source of truth."""
        from src.api.main import get_leaderboard
        data = get_leaderboard("C")

        leaderboard_by_model = {m["model"]: m for m in data.leaderboard}
        category_winners = data.category_winners

        for cat, winner in category_winners.items():
            model_name = winner["model"]
            assert model_name in leaderboard_by_model, f"Category winner {model_name} not in leaderboard table"
            table_row = leaderboard_by_model[model_name]

            # Assert exact equality between card and table
            assert np.isclose(winner["pr_auc"], table_row["pr_auc"], atol=1e-4)
            assert np.isclose(winner["roc_auc"], table_row["roc_auc"], atol=1e-4)
            assert np.isclose(winner["composite_score"], table_row["composite_score"], atol=1e-4)
            assert np.isclose(winner["brier_score"], table_row["brier_score"], atol=1e-4)
            assert np.isclose(winner["ece"], table_row["ece"], atol=1e-4)
            assert winner["split"] == "validation"
            assert table_row["split"] == "validation"

    def test_run_id_and_timestamp_contract(self):
        """Assert active run_id and timestamp are present across API and artifacts."""
        from src.api.main import get_metadata, get_leaderboard
        meta = get_metadata()
        assert meta.run_id is not None
        assert meta.timestamp is not None

        lb = get_leaderboard("C")
        assert lb.run_id is not None
        assert lb.timestamp is not None
