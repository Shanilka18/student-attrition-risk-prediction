"""Unit tests for MEI Track B continuous regression metrics, formulas, and estimators."""

import numpy as np
import pytest
import scipy.stats as stats
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from mei.evaluation.regression_metrics import (
    compute_adjusted_r2,
    compute_breusch_pagan_test,
    compute_classification_at_60,
    compute_regression_metrics,
    compute_top_models_wilcoxon,
    compute_track_a_track_b_consistency,
    compute_track_b_leaderboard_score,
)
from mei.models.regression_registry import (
    create_base_regressor,
    get_all_regression_model_specs,
)


def test_regression_metrics_match_sklearn_scipy() -> None:
    """Regression metrics must exactly match direct scikit-learn and scipy calculations."""
    y_true = np.array([15.0, 35.0, 50.0, 65.0, 80.0, 95.0])
    y_pred = np.array([18.0, 30.0, 55.0, 62.0, 78.0, 90.0])

    metrics, clipped, res = compute_regression_metrics(
        y_true, y_pred, n_features=2, compute_kendall=True
    )

    # Check sklearn parity
    expected_r2 = float(r2_score(y_true, y_pred))
    expected_rmse = float(root_mean_squared_error(y_true, y_pred))
    expected_mae = float(mean_absolute_error(y_true, y_pred))
    expected_med_ae = float(median_absolute_error(y_true, y_pred))

    assert np.isclose(metrics.r2, expected_r2, atol=1e-6)
    assert np.isclose(metrics.rmse, expected_rmse, atol=1e-6)
    assert np.isclose(metrics.mae, expected_mae, atol=1e-6)
    assert np.isclose(metrics.median_ae, expected_med_ae, atol=1e-6)

    # Check scipy parity
    p_stat = stats.pearsonr(y_true, y_pred)
    exp_pearson = float(p_stat.statistic if hasattr(p_stat, "statistic") else p_stat[0])
    s_stat = stats.spearmanr(y_true, y_pred)
    exp_spearman = float(
        s_stat.statistic if hasattr(s_stat, "statistic") else s_stat[0]
    )
    k_stat = stats.kendalltau(y_true, y_pred)
    exp_kendall = float(k_stat.statistic if hasattr(k_stat, "statistic") else k_stat[0])

    assert np.isclose(metrics.pearson_r, exp_pearson, atol=1e-6)
    assert np.isclose(metrics.spearman_rho, exp_spearman, atol=1e-6)
    assert np.isclose(metrics.kendall_tau, exp_kendall, atol=1e-6)

    # Residuals check
    assert np.allclose(res, y_true - y_pred)
    assert np.isclose(metrics.residual_mean, float(np.mean(y_true - y_pred)))


def test_adjusted_r2_formula() -> None:
    """Adjusted R² must strictly implement 1 - [(1 - R²) * (n - 1) / (n - p - 1)]."""
    r2 = 0.80
    n = 100
    p = 10

    # Manual calculation
    expected_adj = 1.0 - ((1.0 - r2) * (n - 1) / (n - p - 1))
    calc_adj = compute_adjusted_r2(r2, n_samples=n, n_features=p)

    assert np.isclose(calc_adj, expected_adj, atol=1e-7)
    assert calc_adj < r2  # Penalizes additional features

    # Fallback when degrees of freedom <= 0
    fallback = compute_adjusted_r2(r2, n_samples=10, n_features=12)
    assert fallback == r2


def test_thresholding_at_60_reproduces_labels() -> None:
    """Thresholding at 60 must reproduce exact at-risk classification labels."""
    y_true_score = np.array([20.0, 45.0, 59.9, 60.0, 75.0, 90.0])
    y_pred_score = np.array([25.0, 40.0, 58.0, 65.0, 70.0, 85.0])

    class_res = compute_classification_at_60(y_true_score, y_pred_score, threshold=60.0)

    # True labels: [0, 0, 0, 1, 1, 1]
    # Pred labels: [0, 0, 0, 1, 1, 1] -> Perfect agreement
    assert class_res.accuracy == 1.0
    assert class_res.balanced_accuracy == 1.0
    assert class_res.f1_at_risk == 1.0
    assert class_res.precision == 1.0
    assert class_res.recall == 1.0
    assert class_res.specificity == 1.0
    assert class_res.mcc == 1.0


def test_clipping_audit_and_enforcement() -> None:
    """Raw predictions outside [0, 100] must be audited and clipped strictly."""
    y_true = np.array([10.0, 50.0, 90.0])
    y_pred_raw = np.array([-15.0, 50.0, 115.0])  # Two out of three out of bounds

    metrics, clipped, res = compute_regression_metrics(
        y_true, y_pred_raw, n_features=1, compute_kendall=False
    )

    assert metrics.clipping_count == 2
    assert np.isclose(metrics.clipping_pct, 66.666667, atol=1e-4)
    assert np.all(clipped >= 0.0)
    assert np.all(clipped <= 100.0)
    assert clipped[0] == 0.0
    assert clipped[2] == 100.0


def test_track_b_leaderboard_score_and_penalty() -> None:
    """Leaderboard score must apply 2 * max(0, gap - 0.05) penalty on R² gap."""
    # Case 1: Good fit (gap <= 0.05)
    score1, gap1, pen1, st1 = compute_track_b_leaderboard_score(
        r2_train=0.82, r2_val=0.80, spearman_val=0.85
    )
    assert gap1 == pytest.approx(0.02)
    assert pen1 == 0.0
    assert score1 == pytest.approx((0.80 + 0.85) / 2.0)
    assert st1 == "Good fit"

    # Case 2: Slight overfit (0.05 < gap <= 0.10)
    score2, gap2, pen2, st2 = compute_track_b_leaderboard_score(
        r2_train=0.88, r2_val=0.80, spearman_val=0.85
    )
    assert gap2 == pytest.approx(0.08)
    assert pen2 == pytest.approx(2.0 * (0.08 - 0.05))  # 0.06 penalty
    assert score2 == pytest.approx((0.80 + 0.85) / 2.0 - 0.06)
    assert st2 == "Slight overfit"

    # Case 3: Overfit (gap > 0.10)
    score3, gap3, pen3, st3 = compute_track_b_leaderboard_score(
        r2_train=0.96, r2_val=0.80, spearman_val=0.85
    )
    assert gap3 == pytest.approx(0.16)
    assert pen3 == pytest.approx(2.0 * (0.16 - 0.05))  # 0.22 penalty
    assert st3 == "Overfit"


def test_breusch_pagan_heteroscedasticity() -> None:
    """Breusch-Pagan test must compute LM statistic and p-value in valid domain."""
    residuals = np.random.normal(0, 5, 100)
    fitted = np.linspace(20, 80, 100)

    bp = compute_breusch_pagan_test(residuals, fitted)
    assert "lm_statistic" in bp
    assert "lm_pvalue" in bp
    assert 0.0 <= bp["lm_pvalue"] <= 1.0
    assert isinstance(bp["is_heteroscedastic"], bool)


def test_wilcoxon_holm_correction() -> None:
    """Multi-model Wilcoxon test must execute paired tests and apply Holm multiplicity correction."""
    np.random.seed(42)
    fold_scores = {
        "model_a": [0.80 + np.random.normal(0, 0.01) for _ in range(15)],
        "model_b": [0.78 + np.random.normal(0, 0.01) for _ in range(15)],
        "model_c": [0.70 + np.random.normal(0, 0.01) for _ in range(15)],
    }

    res = compute_top_models_wilcoxon(fold_scores, alpha=0.05)
    assert res["n_comparisons"] == 3
    assert len(res["pairwise_results"]) == 3
    for p in res["pairwise_results"]:
        assert 0.0 <= p["p_value_holm"] <= 1.0


def test_regression_model_spec_registry_and_factory() -> None:
    """Estimator factory must build valid regressors for candidate specifications."""
    specs = get_all_regression_model_specs()
    assert len(specs) >= 28

    # Test baseline
    est_dummy = create_base_regressor(specs["mean_regressor"])
    assert hasattr(est_dummy, "fit")
    assert hasattr(est_dummy, "predict")

    # Test linear
    est_ridge = create_base_regressor(specs["ridge_regressor"])
    assert hasattr(est_ridge, "fit")

    # Test classical
    est_knn = create_base_regressor(specs["knn_regressor"])
    assert hasattr(est_knn, "fit")

    # Test boosting
    est_rf = create_base_regressor(specs["random_forest_regressor"])
    assert hasattr(est_rf, "fit")


def test_track_a_track_b_consistency() -> None:
    """Consistency checker must measure rank concordance between probability and intent score."""
    scores = np.array([20.0, 35.0, 55.0, 65.0, 75.0, 85.0])
    probs = np.array(
        [0.15, 0.30, 0.45, 0.65, 0.75, 0.90]
    )  # Both agree on <50/60 and >=50/60

    res = compute_track_a_track_b_consistency(probs, scores)
    assert np.isclose(res["spearman_rho"], 1.0, atol=1e-5)
    assert res["classification_agreement_pct"] == 100.0
