"""Tests for Phase 4: Model Zoo, Training, and Evaluation.

Design goals:
- Each model trains on a tiny synthetic dataset in < 30 s.
- Deterministic output given the same seed.
- No NaN predictions.
- Metric computations are mathematically correct.
- run_all.py quick-mode completes without crashing (limited to a subset).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Suppress known deprecation noise
warnings.filterwarnings("ignore")


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def tiny_classification():
    """200 rows, 10 features, binary target, ~20% prevalence."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10)).astype(np.float32)
    y = (rng.random(200) < 0.20).astype(int)
    return X, y


@pytest.fixture(scope="module")
def tiny_regression():
    """200 rows, 10 features, continuous target 0–100."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10)).astype(np.float32)
    y = rng.uniform(0, 100, 200)
    return X, y


@pytest.fixture(scope="module")
def split_tiny(tiny_classification):
    """80/20 train/val split for tiny classification data."""
    X, y = tiny_classification
    X_train, X_val = X[:160], X[160:]
    y_train, y_val = y[:160], y[160:]
    return X_train, X_val, y_train, y_val


@pytest.fixture(scope="module")
def split_tiny_reg(tiny_regression):
    """80/20 train/val split for tiny regression data."""
    X, y = tiny_regression
    return X[:160], X[160:], y[:160], y[160:]


# ===========================================================================
# 1. Base class
# ===========================================================================

class TestBaseModel:
    def test_base_history_empty(self):
        from src.models.base import TrainingHistory
        h = TrainingHistory.empty()
        assert h.best_epoch == 1
        assert h.stopped_early is False

    def test_history_to_dict(self):
        from src.models.base import TrainingHistory
        h = TrainingHistory(epochs=[1, 2], train_loss=[0.5, 0.4], best_epoch=2)
        d = h.to_dict()
        assert "epochs" in d
        assert d["best_epoch"] == 2

    def test_base_track_validation(self):
        with pytest.raises(AssertionError):
            # XGBoostModel accepts track parameter — 'X' is invalid
            from src.models.tree_ensembles import XGBoostModel
            XGBoostModel(track="X")


# ===========================================================================
# 2. Linear / Shallow Models (Classification)
# ===========================================================================

class TestLinearShallowClassification:

    @pytest.mark.parametrize("penalty", ["l2", "l1", "elasticnet"])
    def test_logistic_regression(self, penalty, split_tiny):
        from src.models.linear_shallow import LogisticRegressionModel
        X_tr, X_va, y_tr, y_va = split_tiny
        m = LogisticRegressionModel(penalty=penalty, seed=42)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_va)
        prob = m.predict_proba(X_va)
        assert pred.shape == (40,)
        assert prob.shape == (40, 2)
        assert not np.any(np.isnan(prob))
        assert set(np.unique(pred)).issubset({0, 1})

    def test_linear_svm(self, split_tiny):
        from src.models.linear_shallow import LinearSVMModel
        X_tr, X_va, y_tr, y_va = split_tiny
        m = LinearSVMModel(seed=42)
        m.fit(X_tr, y_tr)
        prob = m.predict_proba(X_va)
        assert prob.shape == (40, 2)
        assert not np.any(np.isnan(prob))

    def test_gaussian_nb(self, split_tiny):
        from src.models.linear_shallow import GaussianNBModel
        X_tr, X_va, y_tr, y_va = split_tiny
        m = GaussianNBModel(seed=42)
        m.fit(X_tr, y_tr)
        assert m.predict(X_va).shape == (40,)

    def test_knn_classifier(self, split_tiny):
        from src.models.linear_shallow import KNNClassifierModel
        X_tr, X_va, y_tr, y_va = split_tiny
        m = KNNClassifierModel(seed=42)
        m.fit(X_tr, y_tr)
        assert m.predict_proba(X_va).shape == (40, 2)

    def test_decision_tree_classifier(self, split_tiny):
        from src.models.linear_shallow import DecisionTreeClassifierModel
        X_tr, X_va, y_tr, y_va = split_tiny
        m = DecisionTreeClassifierModel(seed=42)
        m.fit(X_tr, y_tr)
        assert not np.any(np.isnan(m.predict_proba(X_va)))

    def test_shallow_mlp_classifier(self, split_tiny):
        from src.models.linear_shallow import ShallowMLPClassifierModel
        X_tr, X_va, y_tr, y_va = split_tiny
        m = ShallowMLPClassifierModel(seed=42)
        m.fit(X_tr, y_tr)
        assert m.predict_proba(X_va).shape == (40, 2)
        h = m.get_history()
        assert len(h.epochs) >= 1


# ===========================================================================
# 3. Linear / Shallow Models (Regression)
# ===========================================================================

class TestLinearShallowRegression:

    @pytest.mark.parametrize("ModelCls,name", [
        ("RidgeModel", "ridge"),
        ("LassoModel", "lasso"),
        ("ElasticNetModel", "elasticnet_reg"),
        ("KNNRegressorModel", "knn_regressor"),
        ("DecisionTreeRegressorModel", "decision_tree_regressor"),
    ])
    def test_regression_models(self, ModelCls, name, split_tiny_reg):
        import importlib
        mod = importlib.import_module("src.models.linear_shallow")
        cls = getattr(mod, ModelCls)
        X_tr, X_va, y_tr, y_va = split_tiny_reg
        m = cls(seed=42)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_va)
        assert pred.shape == (40,)
        assert not np.any(np.isnan(pred))

    def test_shallow_mlp_regressor(self, split_tiny_reg):
        from src.models.linear_shallow import ShallowMLPRegressorModel
        X_tr, X_va, y_tr, y_va = split_tiny_reg
        m = ShallowMLPRegressorModel(seed=42)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_va)
        assert pred.shape == (40,)
        assert not np.any(np.isnan(pred))


# ===========================================================================
# 4. Tree Ensembles
# ===========================================================================

class TestTreeEnsembles:

    @pytest.mark.parametrize("model_name", [
        "random_forest", "extra_trees", "gradient_boosting",
        "hist_gradient_boosting", "xgboost", "lightgbm", "catboost",
    ])
    def test_tree_classification(self, model_name, split_tiny):
        from src.models.tree_ensembles import get_tree_ensemble_models
        X_tr, X_va, y_tr, y_va = split_tiny
        registry = get_tree_ensemble_models("C", seed=42)
        m = registry[model_name]
        # Use small round count for speed; CatBoost uses 'iterations', others use 'n_estimators'
        if model_name == "catboost":
            m.params["iterations"] = 20
        else:
            m.params["n_estimators"] = 20
            m.params["iterations"] = 20  # catboost (ignored by non-catboost)
        m.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
        pred = m.predict(X_va)
        prob = m.predict_proba(X_va)
        assert pred.shape == (40,)
        assert prob.shape == (40, 2)
        assert not np.any(np.isnan(prob))
        h = m.get_history()
        assert len(h.epochs) >= 1

    @pytest.mark.parametrize("model_name", [
        "random_forest", "xgboost", "lightgbm", "catboost",
    ])
    def test_tree_regression(self, model_name, split_tiny_reg):
        from src.models.tree_ensembles import get_tree_ensemble_models
        X_tr, X_va, y_tr, y_va = split_tiny_reg
        registry = get_tree_ensemble_models("R", seed=42)
        m = registry[model_name]
        if model_name == "catboost":
            m.params["iterations"] = 20
        else:
            m.params["n_estimators"] = 20
        m.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
        pred = m.predict(X_va)
        assert pred.shape == (40,)
        assert not np.any(np.isnan(pred))


# ===========================================================================
# 5. Deep Tabular Models
# ===========================================================================

class TestDeepTabular:

    @pytest.mark.parametrize("model_name", ["deep_mlp", "tabular_resnet", "cnn_1d"])
    def test_deep_classification(self, model_name, split_tiny):
        from src.models.deep_tabular import get_deep_tabular_models
        X_tr, X_va, y_tr, y_va = split_tiny
        registry = get_deep_tabular_models("C", seed=42)
        m = registry[model_name]
        m.params["max_epochs"] = 5
        m.params["patience"] = 3
        m.fit(X_tr.astype(np.float32), y_tr, X_val=X_va.astype(np.float32), y_val=y_va)
        prob = m.predict_proba(X_va.astype(np.float32))
        assert prob.shape == (40, 2)
        assert not np.any(np.isnan(prob))
        h = m.get_history()
        assert len(h.epochs) >= 1

    @pytest.mark.parametrize("model_name", ["deep_mlp", "tabular_resnet", "cnn_1d"])
    def test_deep_regression(self, model_name, split_tiny_reg):
        from src.models.deep_tabular import get_deep_tabular_models
        X_tr, X_va, y_tr, y_va = split_tiny_reg
        registry = get_deep_tabular_models("R", seed=42)
        m = registry[model_name]
        m.params["max_epochs"] = 5
        m.params["patience"] = 3
        m.fit(X_tr.astype(np.float32), y_tr.astype(np.float32),
              X_val=X_va.astype(np.float32), y_val=y_va.astype(np.float32))
        pred = m.predict(X_va.astype(np.float32))
        assert pred.shape == (40,)
        assert not np.any(np.isnan(pred))

    def test_tabnet_classification(self, split_tiny):
        """TabNet separate test — slower, so isolated."""
        from src.models.deep_tabular import TabNetModel
        X_tr, X_va, y_tr, y_va = split_tiny
        m = TabNetModel("C", seed=42)
        m.params["max_epochs"] = 5
        m.params["patience"] = 3
        m.fit(X_tr.astype(np.float32), y_tr,
              X_val=X_va.astype(np.float32), y_val=y_va)
        prob = m.predict_proba(X_va.astype(np.float32))
        assert prob.shape == (40, 2)


# ===========================================================================
# 6. Evaluation Metrics
# ===========================================================================

class TestMetrics:

    @pytest.fixture
    def binary_data(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        y_prob = np.clip(rng.beta(2, 5, 200) + y_true * 0.3, 0, 1)
        y_pred = (y_prob >= 0.5).astype(int)
        return y_true, y_pred, y_prob

    @pytest.fixture
    def regression_data(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 100, 200)
        y_pred = y_true + rng.normal(0, 10, 200)
        return y_true, y_pred

    def test_classification_metrics_keys(self, binary_data):
        from src.evaluation.metrics import compute_classification_metrics
        y_true, y_pred, y_prob = binary_data
        m = compute_classification_metrics(y_true, y_pred, y_prob, split="val",
                                           n_bootstrap=50, seed=0)
        required = ["roc_auc", "pr_auc", "f1", "balanced_accuracy", "brier_score",
                    "ece", "recall", "specificity", "mcc", "cohen_kappa", "log_loss"]
        for key in required:
            assert key in m, f"Missing key: {key}"

    def test_bootstrap_ci_present_on_val(self, binary_data):
        from src.evaluation.metrics import compute_classification_metrics
        y_true, y_pred, y_prob = binary_data
        m = compute_classification_metrics(y_true, y_pred, y_prob, split="val",
                                           n_bootstrap=50, seed=0)
        assert "roc_auc_ci_lo" in m
        assert "roc_auc_ci_hi" in m
        assert m["roc_auc_ci_lo"] <= m["roc_auc"] <= m["roc_auc_ci_hi"] + 1e-6

    def test_no_ci_on_train(self, binary_data):
        from src.evaluation.metrics import compute_classification_metrics
        y_true, y_pred, y_prob = binary_data
        m = compute_classification_metrics(y_true, y_pred, y_prob, split="train",
                                           n_bootstrap=50, seed=0)
        assert "roc_auc_ci_lo" not in m

    def test_regression_metrics_keys(self, regression_data):
        from src.evaluation.metrics import compute_regression_metrics
        y_true, y_pred = regression_data
        m = compute_regression_metrics(y_true, y_pred, split="val",
                                       n_features=10, n_bootstrap=50, seed=0)
        required = ["r2", "adj_r2", "mae", "rmse", "smape", "explained_variance",
                    "spearman_rho"]
        for key in required:
            assert key in m, f"Missing key: {key}"

    def test_ece_range(self, binary_data):
        from src.evaluation.metrics import compute_ece
        y_true, _, y_prob = binary_data
        ece = compute_ece(y_true, y_prob)
        assert 0.0 <= ece <= 1.0

    def test_curve_data_shapes(self, binary_data):
        from src.evaluation.metrics import (
            compute_roc_curve_data, compute_pr_curve_data,
            compute_calibration_curve_data,
        )
        y_true, _, y_prob = binary_data
        roc = compute_roc_curve_data(y_true, y_prob)
        assert len(roc["fpr"]) == len(roc["tpr"])

        pr = compute_pr_curve_data(y_true, y_prob)
        assert len(pr["precision"]) == len(pr["recall"])

        cal = compute_calibration_curve_data(y_true, y_prob)
        assert "fraction_of_positives" in cal


# ===========================================================================
# 7. Tune module (quick mode — 2 trials only)
# ===========================================================================

class TestTuner:

    def test_tune_returns_dict(self, tiny_classification):
        from src.training.tune import tune_model
        from src.models.linear_shallow import RidgeModel, LogisticRegressionModel
        X, y = tiny_classification
        params = tune_model(
            model_class=LogisticRegressionModel,
            track="C",
            X_train=X,
            y_train=y,
            model_name="logistic_l2",
            n_trials=2,
            n_inner_folds=2,
            seed=42,
            model_extra_kwargs={"penalty": "l2"},
        )
        assert isinstance(params, dict)

    def test_tune_regression(self, tiny_regression):
        from src.training.tune import tune_model
        from src.models.linear_shallow import RidgeModel
        X, y = tiny_regression
        params = tune_model(
            model_class=RidgeModel,
            track="R",
            X_train=X,
            y_train=y,
            model_name="ridge",
            n_trials=2,
            n_inner_folds=2,
            seed=42,
        )
        assert isinstance(params, dict)


# ===========================================================================
# 8. run_all quick-mode integration (subset of models only)
# ===========================================================================

class TestRunAllIntegration:

    @pytest.mark.slow
    def test_run_all_track_c_quick(self):
        """Run logistic_l2 + xgboost in quick mode — should complete without crashing."""
        from src.training.run_all import run_all
        run_all(
            track="C",
            models=["logistic_l2", "xgboost"],
            do_tune=False,  # no tuning in test
            n_trials=2,
            open_test=False,
            seed=42,
            quick=True,
        )
        # Verify metrics CSV exists and has rows
        metrics_csv = Path(ROOT) / "artifacts" / "metrics" / "all_splits_metrics.csv"
        assert metrics_csv.exists(), "all_splits_metrics.csv not created"
        df = pd.read_csv(metrics_csv)
        assert len(df) >= 2, f"Expected at least 2 rows in metrics CSV, got {len(df)}"
        assert "model" in df.columns
        assert "track" in df.columns
        assert "split" in df.columns

    @pytest.mark.slow
    def test_run_all_track_r_quick(self):
        """Run ridge + random_forest in quick mode — Track R."""
        from src.training.run_all import run_all
        run_all(
            track="R",
            models=["ridge", "random_forest"],
            do_tune=False,
            n_trials=2,
            open_test=False,
            seed=42,
            quick=True,
        )
        metrics_csv = Path(ROOT) / "artifacts" / "metrics" / "all_splits_metrics.csv"
        assert metrics_csv.exists()

    def test_metrics_csv_column_order(self):
        """Verify all_splits_metrics.csv has correct column order."""
        metrics_csv = Path(ROOT) / "artifacts" / "metrics" / "all_splits_metrics.csv"
        if not metrics_csv.exists():
            pytest.skip("Metrics CSV not yet generated — run test_run_all_track_c_quick first")
        df = pd.read_csv(metrics_csv)
        cols = list(df.columns)
        assert cols[0] == "model"
        assert cols[1] == "track"
        assert cols[2] == "split"
        assert cols[3] == "samples"

    def test_history_json_written(self):
        """Check that at least one history JSON was written."""
        history_dir = Path(ROOT) / "artifacts" / "history"
        files = list(history_dir.glob("*.json"))
        if not files:
            pytest.skip("History files not yet generated — run integration test first")
        h = json.loads(files[0].read_text())
        assert "epochs" in h
        assert "train_loss" in h
        assert "best_epoch" in h

    def test_model_artifact_written(self):
        """Check that at least one model artifact exists."""
        model_c_dir = Path(ROOT) / "artifacts" / "models" / "C"
        model_r_dir = Path(ROOT) / "artifacts" / "models" / "R"
        c_files = list(model_c_dir.glob("*"))
        r_files = list(model_r_dir.glob("*"))
        if not (c_files or r_files):
            pytest.skip("Model artifacts not yet generated — run integration test first")
        assert (c_files or r_files), "No model artifacts found"


# ===========================================================================
# 9. Reproducibility
# ===========================================================================

class TestReproducibility:

    def test_logistic_deterministic(self, split_tiny):
        from src.models.linear_shallow import LogisticRegressionModel
        X_tr, X_va, y_tr, _ = split_tiny
        m1 = LogisticRegressionModel("l2", seed=42)
        m2 = LogisticRegressionModel("l2", seed=42)
        m1.fit(X_tr, y_tr)
        m2.fit(X_tr, y_tr)
        np.testing.assert_array_equal(m1.predict(X_va), m2.predict(X_va))

    def test_xgboost_deterministic(self, split_tiny):
        from src.models.tree_ensembles import XGBoostModel
        X_tr, X_va, y_tr, y_va = split_tiny
        m1 = XGBoostModel("C", seed=42)
        m1.params["n_estimators"] = 20
        m2 = XGBoostModel("C", seed=42)
        m2.params["n_estimators"] = 20
        m1.fit(X_tr, y_tr)
        m2.fit(X_tr, y_tr)
        np.testing.assert_array_almost_equal(
            m1.predict_proba(X_va)[:, 1],
            m2.predict_proba(X_va)[:, 1],
        )

    def test_deep_mlp_deterministic(self, split_tiny):
        from src.models.deep_tabular import DeepMLPModel
        X_tr, X_va, y_tr, y_va = split_tiny
        X_tr = X_tr.astype(np.float32)
        X_va = X_va.astype(np.float32)

        def run():
            m = DeepMLPModel("C", seed=42)
            m.params["max_epochs"] = 3
            m.params["patience"] = 2
            m.fit(X_tr, y_tr)
            return m.predict_proba(X_va)[:, 1]

        p1 = run()
        p2 = run()
        np.testing.assert_array_almost_equal(p1, p2, decimal=4)
