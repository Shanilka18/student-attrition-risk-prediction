"""Unit and integration test suite for Phase 6 Explainability Engine."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from src.explain.explainer import detect_explainer_type, build_explainer, explain_dataset
from src.explain.global_explain import compute_global_importance, aggregate_factor_importance
from src.explain.local_explain import explain_student_instance
from src.explain.narrative import generate_student_narrative, DISCLAIMER_TEXT
from src.explain.sanity_checks import (
    check_additivity,
    check_subsample_stability,
    check_ground_truth_recovery,
)


@pytest.fixture
def mock_dataset():
    """Create minimal mock dataset for explainability tests."""
    rng = np.random.default_rng(42)
    n = 60
    feature_names = [
        "factor_stress",
        "factor_motivation",
        "factor_belongingness",
        "factor_self_efficacy",
        "factor_peer_support",
    ]
    X = pd.DataFrame(rng.uniform(1.0, 5.0, size=(n, len(feature_names))), columns=feature_names)
    # y = 0.28*stress + 0.22*(6-motivation) + ...
    y = (
        0.28 * X["factor_stress"]
        + 0.22 * (6.0 - X["factor_motivation"])
        + 0.18 * (6.0 - X["factor_belongingness"])
        + 0.18 * (6.0 - X["factor_self_efficacy"])
        + 0.14 * (6.0 - X["factor_peer_support"])
    ) * 15.0
    return X, y, feature_names


def test_detect_explainer_type():
    """Test explainer type detection across architectures."""
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
    from sklearn.neighbors import KNeighborsClassifier

    assert detect_explainer_type(Ridge()) == "linear"
    assert detect_explainer_type(LogisticRegression()) == "linear"
    assert detect_explainer_type(HistGradientBoostingClassifier()) == "tree"
    assert detect_explainer_type(RandomForestRegressor()) == "tree"
    assert detect_explainer_type(KNeighborsClassifier()) == "kernel"


def test_narrative_generator_rules():
    """Verify deterministic plain-language narrative generation and scientific rules."""
    top_inc = [
        {"feature": "factor_stress", "shap_value": 4.5, "feature_value": 4.2},
        {"feature": "factor_belongingness", "shap_value": 2.1, "feature_value": 1.8},
    ]
    top_dec = [
        {"feature": "factor_peer_support", "shap_value": -3.2, "feature_value": 4.5},
        {"feature": "factor_self_efficacy", "shap_value": -1.8, "feature_value": 3.9},
    ]

    narrative = generate_student_narrative(
        study_id="STUDY_999",
        pri_score=78.4,
        risk_category="High",
        top_risk_increasing=top_inc,
        top_risk_decreasing=top_dec,
    )

    assert narrative["study_id"] == "STUDY_999"
    assert narrative["pri_score"] == 78.4
    assert narrative["risk_category"] == "HIGH"
    assert "contributing indicator" in narrative["full_narrative"]
    assert "not clinical" in narrative["disclaimer"].lower() or "not a clinical" in narrative["disclaimer"].lower()
    assert len(narrative["recommendations"]) >= 1

    # Wording rule: must explicitly say contributing indicators, not direct causes
    assert "contributing indicators" in narrative["full_narrative"]
    assert "rather than deterministic causes" in narrative["full_narrative"]


def test_explain_student_instance_waterfall(mock_dataset):
    """Test local waterfall summation and top driver isolation."""
    X, y, feature_names = mock_dataset
    base_val = 50.0
    # Synthetic SHAP values: sum of shap values
    rng = np.random.default_rng(42)
    shap_vals = rng.normal(0, 2.0, size=X.shape)

    exp = explain_student_instance(
        student_idx=0,
        shap_values=shap_vals,
        X=X,
        feature_names=feature_names,
        base_value=base_val,
        study_id="ST_001",
    )

    assert exp["study_id"] == "ST_001"
    assert exp["student_idx"] == 0
    assert "top_risk_increasing" in exp
    assert "top_risk_decreasing" in exp
    assert len(exp["top_risk_increasing"]) <= 3
    assert len(exp["top_risk_decreasing"]) <= 3

    # Additivity of waterfall
    computed_sum = exp["base_value"] + exp["total_shap_sum"]
    assert np.isclose(computed_sum, exp["predicted_value"], atol=1e-3)


def test_global_importance_and_aggregation():
    """Verify global feature ranking and factor aggregation."""
    feature_names = ["factor_stress", "factor_motivation", "factor_belongingness", "factor_self_efficacy", "factor_peer_support"]
    shap_values = np.array([
        [1.0, -0.8, -0.6, -0.5, -0.3],
        [1.5, -0.7, -0.4, -0.6, -0.2],
        [0.8, -0.9, -0.7, -0.4, -0.4],
    ])

    df_glob = compute_global_importance(shap_values, feature_names)
    assert len(df_glob) == 5
    assert df_glob.iloc[0]["feature"] == "factor_stress"
    assert np.isclose(df_glob["relative_importance_pct"].sum(), 100.0, atol=0.1)

    df_agg = aggregate_factor_importance(df_glob)
    assert len(df_agg) == 5
    assert df_agg.iloc[0]["factor"] == "stress"
    assert np.isclose(df_agg["percentage_contribution"].sum(), 100.0, atol=0.1)


def test_sanity_checks_additivity_and_ground_truth(mock_dataset):
    """Verify additivity check and ground truth recovery check."""
    X, y, feature_names = mock_dataset
    model = Ridge().fit(X, y)

    # For Ridge, linear SHAP is exact: phi_j = w_j * (x_j - E[x_j])
    coef = model.coef_
    mean_X = X.mean().values
    shap_vals = (X.values - mean_X) * coef
    base_val = float(model.predict(mean_X.reshape(1, -1))[0])

    additivity = check_additivity(model, X, shap_vals, base_val, tolerance=1e-2)
    assert additivity["passed"] is True
    assert additivity["status"] == "PASS"

    # Test ground truth check
    factor_df = pd.DataFrame({
        "factor": ["stress", "motivation", "belongingness", "self_efficacy", "peer_support"],
        "mean_abs_shap": [0.28, 0.22, 0.18, 0.17, 0.14],
    })
    gt = check_ground_truth_recovery(factor_df)
    assert gt["passed"] is True
    assert gt["spearman_rank_correlation"] >= 0.90
