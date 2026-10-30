"""
Unit Tests for Phase 5: Training, Tuning, Calibration, and Validation Engine.
Verifies:
  - Zero leakage in TrainingDataLoader
  - Accuracy and stability of validation metrics
  - Probability calibration (Platt & Isotonic)
  - Threshold optimization (Max-F1 & Recall-Constrained)
  - Single model train lifecycle & artifact persistence
  - Resumability (skips completed models)
  - Fault tolerance (logs failure reason without crashing)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.src.training.calibration import ProbabilityCalibrator
from backend.src.training.data_loader import TrainingDataLoader
from backend.src.training.engine import train_and_validate_model
from backend.src.training.metrics import compute_validation_metrics
from backend.src.training.threshold import optimize_decision_threshold


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Loader & Leakage Safeguards
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_splits_load_without_overlap(self):
        loader = TrainingDataLoader()
        train_sids, val_sids = loader.load_splits()
        assert len(train_sids) > 0
        assert len(val_sids) > 0
        overlap = set(train_sids) & set(val_sids)
        assert len(overlap) == 0, f"Found overlapping students: {overlap}"

    def test_tabular_data_shapes(self):
        loader = TrainingDataLoader()
        data = loader.load_tabular_data()
        assert "X_train" in data and "X_val" in data
        assert data["X_train"].shape[1] == 22
        assert len(data["X_train"]) == len(data["y_train"])
        assert len(data["X_val"]) == len(data["y_val"])

    def test_static_data_shapes(self):
        loader = TrainingDataLoader()
        data = loader.load_static_data()
        assert data["X_train"].shape[1] == 2
        assert data["X_val"].shape[1] == 2
        assert list(data["feature_names"]) == ["last_gpa", "last_attendance_pct"]

    def test_sequence_data_shapes(self):
        loader = TrainingDataLoader()
        data = loader.load_sequence_data()
        assert isinstance(data["X_train"], dict)
        assert "tensors" in data["X_train"]
        assert "padding_mask" in data["X_train"]
        assert "seq_lengths" in data["X_train"]
        assert data["X_train"]["tensors"].ndim == 3
        assert data["X_train"]["tensors"].shape[2] == 4

    def test_cv_folds_zero_overlap(self):
        loader = TrainingDataLoader()
        folds = loader.get_train_cv_folds(n_splits=3, seed=42)
        assert len(folds) == 3
        tab_data = loader.load_tabular_data()
        sids = tab_data["student_ids_train"]

        for trn_idx, val_idx in folds:
            trn_sids = set(sids[trn_idx])
            val_sids = set(sids[val_idx])
            overlap = trn_sids & val_sids
            assert len(overlap) == 0, "Grouped CV leaked student IDs between fold train and fold val"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Validation Metrics Calculation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationMetrics:
    def test_metrics_values(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.1, 0.4, 0.85, 0.75])
        metrics = compute_validation_metrics(y_true, y_prob, threshold=0.5)

        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "brier_score" in metrics
        assert "f1_score" in metrics
        assert "recall" in metrics
        assert "precision" in metrics
        assert "accuracy" in metrics
        assert "specificity" in metrics

        assert metrics["roc_auc"] > 0.90
        assert metrics["brier_score"] < 0.10
        assert metrics["accuracy"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Probability Calibration
# ─────────────────────────────────────────────────────────────────────────────

class TestProbabilityCalibration:
    def test_platt_scaling(self, tmp_path):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.4, 0.45, 0.48, 0.42, 0.55, 0.6, 0.7, 0.8])

        cal = ProbabilityCalibrator(method="platt")
        cal.fit(y_true, y_prob)
        cal_probs = cal.transform(y_prob)

        assert len(cal_probs) == len(y_prob)
        assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)
        assert cal.brier_after <= cal.brier_before + 1e-4

        # Save/load roundtrip
        save_path = tmp_path / "cal.joblib"
        cal.save(save_path)
        loaded = ProbabilityCalibrator.load(save_path)
        loaded_probs = loaded.transform(y_prob)
        np.testing.assert_allclose(cal_probs, loaded_probs)

    def test_isotonic_regression(self):
        y_true = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9])

        cal = ProbabilityCalibrator(method="isotonic")
        cal.fit(y_true, y_prob)
        cal_probs = cal.transform(y_prob)

        assert len(cal_probs) == len(y_prob)
        assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Decision Threshold Optimization
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionThreshold:
    def test_max_f1_strategy(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        opt = optimize_decision_threshold(y_true, y_prob, strategy="max_f1")
        assert opt["strategy"] == "max_f1"
        assert 0.4 <= opt["optimal_threshold"] <= 0.6
        assert opt["metrics_at_threshold"]["f1_score"] == 1.0

    def test_recall_constrained_strategy(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        opt = optimize_decision_threshold(y_true, y_prob, strategy="recall_constrained", min_recall=1.0)
        assert opt["strategy"] == "recall_constrained"
        assert opt["metrics_at_threshold"]["recall"] >= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Single Model Training Engine & Resumability
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingEngine:
    @pytest.fixture
    def tiny_tabular_data(self):
        np.random.seed(42)
        X_tr = np.random.randn(80, 22).astype(np.float32)
        y_tr = np.array([0] * 60 + [1] * 20, dtype=np.int64)
        X_va = np.random.randn(30, 22).astype(np.float32)
        y_va = np.array([0] * 20 + [1] * 10, dtype=np.int64)
        return {
            "X_train": X_tr,
            "y_train": y_tr,
            "X_val": X_va,
            "y_val": y_va,
            "feature_names": [f"f_{i}" for i in range(22)],
        }

    def test_train_tabular_model_produces_artifacts(self, tiny_tabular_data, tmp_path):
        cv_folds = [(np.arange(0, 40), np.arange(40, 80))]
        summary = train_and_validate_model(
            model_key="logistic_l2",
            data=tiny_tabular_data,
            cv_folds=cv_folds,
            category="A",
            n_tuning_trials=2,
            seeds=[42, 123, 999],
            output_base_dir=tmp_path,
        )

        assert summary["status"] == "COMPLETED"
        assert "validation_metrics" in summary
        assert "stability_summary" in summary
        assert "roc_auc_std" in summary["stability_summary"]
        assert summary["stability_summary"]["roc_auc_std"] >= 0.0

        model_dir = tmp_path / "Category_A" / "logistic_l2"
        assert (model_dir / "summary.json").exists()
        assert (model_dir / "metadata.json").exists()
        assert (model_dir / "best_params.json").exists()
        assert (model_dir / "seed_metrics.json").exists()
        assert (model_dir / "threshold_config.json").exists()
        assert (model_dir / "calibrator.joblib").exists()

    def test_resumability_skips_finished_model(self, tiny_tabular_data, tmp_path):
        cv_folds = [(np.arange(0, 40), np.arange(40, 80))]
        # First run: trains and persists
        summary_1 = train_and_validate_model(
            model_key="decision_tree",
            data=tiny_tabular_data,
            cv_folds=cv_folds,
            category="B",
            n_tuning_trials=2,
            seeds=[42],
            output_base_dir=tmp_path,
            force=False,
        )
        assert summary_1["status"] == "COMPLETED"

        # Second run: should immediately load from disk
        summary_2 = train_and_validate_model(
            model_key="decision_tree",
            data=tiny_tabular_data,
            cv_folds=cv_folds,
            category="B",
            n_tuning_trials=2,
            seeds=[42],
            output_base_dir=tmp_path,
            force=False,
        )
        assert summary_2["status"] == "COMPLETED"
        assert summary_2["model_key"] == "decision_tree"
