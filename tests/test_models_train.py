"""Unit and smoke tests for Model Zoo and Training Engine."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.models.registry import (
    MODEL_CATEGORIES,
    list_registered_models,
    build_model,
)
from src.models.train import compute_metrics, train_single_model
from src.features.build_features import RAW_ITEM_COLS


@pytest.fixture
def tiny_dataset():
    """Create a minimal 30-row dataset for quick smoke-testing of model estimators."""
    rng = np.random.default_rng(42)
    n = 30
    data = {"study_id": [f"TEST_{i:03d}" for i in range(n)]}
    for col in RAW_ITEM_COLS:
        data[col] = rng.integers(1, 6, size=n)

    df = pd.DataFrame(data)
    df["pri_true"] = rng.uniform(20.0, 80.0, size=n)
    df["risk_category"] = rng.choice(["Low", "Moderate", "High"], size=n)
    df["attrition_label"] = rng.choice([0, 1], size=n)
    return df


def test_model_registry_coverage():
    """Verify registry contains models across all required categories."""
    categories = set(MODEL_CATEGORIES.values())
    expected_categories = {"Baseline", "Linear/Shallow", "Ensemble/Bagging", "Boosting", "Meta"}
    assert expected_categories.issubset(categories)

    reg_models = list_registered_models("regression")
    clf_models = list_registered_models("classification")

    assert "dummy" in reg_models
    assert "dummy" in clf_models
    assert "random_forest" in reg_models
    assert "random_forest" in clf_models
    assert "xgboost" in reg_models
    assert "voting_ensemble" in reg_models


def test_smoke_fit_predict_regression(tiny_dataset):
    """Smoke test fitting and predicting on a representative subset of regression models."""
    sample_models = [
        "dummy", "ridge", "decision_tree",
        "random_forest", "hist_gradient_boosting",
        "voting_ensemble",
    ]
    X = tiny_dataset[RAW_ITEM_COLS]
    y = tiny_dataset["pri_true"]

    for m_name in sample_models:
        model = build_model(m_name, track="regression", seed=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()


def test_smoke_fit_predict_classification(tiny_dataset):
    """Smoke test fitting and predicting on a representative subset of classification models."""
    sample_models = [
        "dummy", "logistic_regression", "decision_tree",
        "random_forest", "hist_gradient_boosting",
        "stacking_ensemble",
    ]
    X = tiny_dataset[RAW_ITEM_COLS]
    y = tiny_dataset["risk_category"]

    for m_name in sample_models:
        model = build_model(m_name, track="classification", seed=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({"Low", "Moderate", "High"})


def test_compute_metrics_regression_and_classification():
    """Verify primary and secondary metric evaluations."""
    y_reg_true = np.array([50.0, 60.0, 70.0])
    y_reg_pred = np.array([52.0, 58.0, 71.0])
    m_reg = compute_metrics(y_reg_true, y_reg_pred, track="regression")
    assert "rmse" in m_reg
    assert "mae" in m_reg
    assert "r2" in m_reg
    assert m_reg["rmse"] > 0

    y_clf_true = np.array(["Low", "Moderate", "High", "Low"])
    y_clf_pred = np.array(["Low", "Moderate", "High", "Moderate"])
    m_clf = compute_metrics(y_clf_true, y_clf_pred, track="classification")
    assert "accuracy" in m_clf
    assert "macro_f1" in m_clf
    assert 0.0 <= m_clf["macro_f1"] <= 1.0


def test_train_single_model_quick(tmp_path, tiny_dataset):
    """Smoke test train_single_model in quick mode with artifact saving."""
    train_df = tiny_dataset.head(20).copy()
    val_df = tiny_dataset.tail(10).copy()

    meta = train_single_model(
        model_name="ridge",
        track="regression",
        train_df=train_df,
        val_df=val_df,
        n_trials=1,
        quick=True,
        seed=42,
        output_dir=str(tmp_path),
    )

    assert meta is not None
    assert meta["model_name"] == "ridge"
    assert "validation_metrics" in meta
    assert "overfitting_gap" in meta
    assert meta["overfitting_gap"] >= 0.0
    assert Path(meta["artifact_path"]).is_file()
