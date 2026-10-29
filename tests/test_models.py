"""Unit tests for MEI classification benchmarking engine and evaluation metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mei.evaluation.metrics import (
    compute_leaderboard_score,
    compute_probabilistic_metrics_and_curves,
    compute_threshold_metrics,
)
from mei.features.diagnostics import audit_leakage
from mei.models.engine import (
    evaluate_single_model,
    evaluate_stacking_ensemble,
    evaluate_voting_ensemble,
)
from mei.models.registry import (
    ModelSpec,
    RuleLoginImitationClassifier,
    get_all_model_specs,
)


def test_metrics_sklearn_parity() -> None:
    """Assert our evaluation metrics match scikit-learn standard implementations on toy data."""
    rng = np.random.RandomState(42)
    n = 200
    y_true = rng.binomial(1, 0.3, size=n)
    # Generate realistic correlated probabilities
    y_prob = np.clip(y_true * 0.6 + rng.uniform(0.1, 0.4, size=n), 0.01, 0.99)
    y_pred = (y_prob >= 0.5).astype(int)

    # 1. Threshold metrics at 0.5
    tm = compute_threshold_metrics(y_true, y_pred, threshold=0.5)

    assert np.isclose(tm.accuracy, accuracy_score(y_true, y_pred), atol=1e-5)
    assert np.isclose(
        tm.balanced_accuracy, balanced_accuracy_score(y_true, y_pred), atol=1e-5
    )
    assert np.isclose(
        tm.precision, precision_score(y_true, y_pred, zero_division=0), atol=1e-5
    )
    assert np.isclose(
        tm.recall, recall_score(y_true, y_pred, zero_division=0), atol=1e-5
    )
    assert np.isclose(tm.f1, f1_score(y_true, y_pred, zero_division=0), atol=1e-5)
    assert np.isclose(tm.mcc, matthews_corrcoef(y_true, y_pred), atol=1e-5)
    assert np.isclose(tm.cohen_kappa, cohen_kappa_score(y_true, y_pred), atol=1e-5)

    # 2. Probabilistic metrics
    pm, curves = compute_probabilistic_metrics_and_curves(y_true, y_prob)

    assert np.isclose(pm.auc_roc, roc_auc_score(y_true, y_prob), atol=1e-5)
    assert np.isclose(pm.brier_score, brier_score_loss(y_true, y_prob), atol=1e-5)
    assert np.isclose(pm.log_loss, log_loss(y_true, y_prob), atol=1e-5)
    assert 0.0 <= pm.ece <= 1.0


def test_leaderboard_score_and_penalty() -> None:
    """Test leaderboard score formula, overfit penalties, and category transitions."""
    # Case 1: Good fit (gap <= 0.05)
    score_gf, gap_gf, pen_gf, status_gf = compute_leaderboard_score(
        val_f1=0.80,
        val_auc_roc=0.90,
        val_balanced_acc=0.85,
        val_mcc=0.70,
        train_f1=0.83,  # gap = 0.03 <= 0.05
    )
    expected_mcc_norm = (0.70 + 1.0) / 2.0  # 0.85
    expected_base = (0.80 + 0.90 + 0.85 + expected_mcc_norm) / 4.0  # 0.85
    assert np.isclose(gap_gf, 0.03, atol=1e-4)
    assert pen_gf == 0.0
    assert np.isclose(score_gf, expected_base, atol=1e-4)
    assert status_gf == "Good fit"

    # Case 2: Slight overfit (0.05 < gap <= 0.10)
    score_so, gap_so, pen_so, status_so = compute_leaderboard_score(
        val_f1=0.80,
        val_auc_roc=0.90,
        val_balanced_acc=0.85,
        val_mcc=0.70,
        train_f1=0.88,  # gap = 0.08
    )
    assert np.isclose(gap_so, 0.08, atol=1e-4)
    expected_pen_so = 2.0 * (0.08 - 0.05)  # 0.06
    assert np.isclose(pen_so, expected_pen_so, atol=1e-4)
    assert np.isclose(score_so, expected_base - expected_pen_so, atol=1e-4)
    assert status_so == "Slight overfit"

    # Case 3: Overfit (gap > 0.10)
    score_of, gap_of, pen_of, status_of = compute_leaderboard_score(
        val_f1=0.75,
        val_auc_roc=0.85,
        val_balanced_acc=0.80,
        val_mcc=0.60,
        train_f1=0.92,  # gap = 0.17
    )
    assert np.isclose(gap_of, 0.17, atol=1e-4)
    assert status_of == "Overfit"
    expected_pen_of = 2.0 * (0.17 - 0.05)  # 0.24
    assert np.isclose(pen_of, expected_pen_of, atol=1e-4)


def test_rule_login_imitation_classifier() -> None:
    """Test the RuleLoginImitationClassifier baseline behaves predictably."""
    clf = RuleLoginImitationClassifier(threshold=2.0)
    # X has 4 columns (resource_access p1..p4)
    # Row 0: mean = 1.5 -> at risk (1)
    # Row 1: mean = 3.5 -> not at risk (0)
    X = np.array(
        [
            [1.0, 2.0, 1.0, 2.0],
            [3.0, 4.0, 3.0, 4.0],
        ]
    )
    y = np.array([1, 0])

    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds[0] == 1
    assert preds[1] == 0

    probs = clf.predict_proba(X)
    assert probs.shape == (2, 2)
    assert probs[0, 1] > 0.5
    assert probs[1, 1] < 0.5


def test_model_reproducibility() -> None:
    """Test that running evaluate_single_model twice with the same seed yields identical scores."""
    rng = np.random.RandomState(42)
    n = 120
    p = 10
    X = rng.randn(n, p)
    y = rng.binomial(1, 0.3, size=n)

    # 3 folds
    cv_folds = [
        {"fold": 0, "train_idx": list(range(0, 80)), "val_idx": list(range(80, 120))},
        {"fold": 1, "train_idx": list(range(40, 120)), "val_idx": list(range(0, 40))},
        {
            "fold": 2,
            "train_idx": list(range(0, 40)) + list(range(80, 120)),
            "val_idx": list(range(40, 80)),
        },
    ]

    spec = ModelSpec(
        id="decision_tree",
        display_name="Decision Tree",
        plain_name="Decision Tree",
        category="classical",
        task="classification",
        feature_set="agg",
        needs_scaling=False,
        supports_class_weight=True,
        n_tuning_trials=0,
        time_budget_seconds=10,
        fixed_params={"max_depth": 3, "min_samples_leaf": 2},
    )

    res1 = evaluate_single_model(
        spec, {"agg": X}, y, cv_folds=cv_folds, seed=42, enable_tuning=False
    )
    res2 = evaluate_single_model(
        spec, {"agg": X}, y, cv_folds=cv_folds, seed=42, enable_tuning=False
    )

    score1 = res1["leaderboard_metrics"]["leaderboard_score"]
    score2 = res2["leaderboard_metrics"]["leaderboard_score"]
    assert np.isclose(score1, score2, atol=1e-7)

    # OOF probabilities must match identically
    np.testing.assert_allclose(
        res1["oof_predictions"], res2["oof_predictions"], atol=1e-7
    )


def test_calibrator_and_scaler_fit_rows_match_each_training_fold(monkeypatch) -> None:
    """Inspect fitted calibration and scaling inputs for each outer CV fold."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler

    import mei.models.engine as engine

    rng = np.random.RandomState(7)
    X = rng.normal(size=(120, 8))
    y = np.tile([0, 1], 60)
    cv_folds = [
        {"fold": 0, "train_idx": list(range(60)), "val_idx": list(range(60, 120))},
        {"fold": 1, "train_idx": list(range(60, 120)), "val_idx": list(range(60))},
    ]
    fit_counts = {"calibrator": [], "scaler": []}

    original_calibrator_fit = CalibratedClassifierCV.fit
    original_scaler_fit = StandardScaler.fit

    def record_calibrator_fit(estimator, X_fit, y_fit, *args, **kwargs):
        fit_counts["calibrator"].append(len(X_fit))
        return original_calibrator_fit(estimator, X_fit, y_fit, *args, **kwargs)

    def record_scaler_fit(scaler, X_fit, y_fit=None, **kwargs):
        fit_counts["scaler"].append(len(X_fit))
        return original_scaler_fit(scaler, X_fit, y_fit, **kwargs)

    monkeypatch.setattr(CalibratedClassifierCV, "fit", record_calibrator_fit)
    monkeypatch.setattr(StandardScaler, "fit", record_scaler_fit)
    monkeypatch.setattr(engine, "compute_learning_curve_data", lambda *a, **k: {})
    monkeypatch.setattr(engine, "compute_validation_curve_data", lambda *a, **k: {})
    monkeypatch.setattr(
        engine, "compute_boosting_iteration_losses", lambda *a, **k: None
    )

    spec = get_all_model_specs()["ridge_calibrated"]
    evaluate_single_model(
        spec,
        {"agg": X},
        y,
        cv_folds=cv_folds,
        seed=7,
        enable_tuning=False,
    )

    assert fit_counts["calibrator"] == [60, 60, 120]
    assert fit_counts["scaler"] == [60, 60, 120]


def test_stacking_oof_canary_leakage_audit() -> None:
    """Test that canary leakage is caught by audit and stacking uses OOF properly."""
    rng = np.random.RandomState(42)
    n = 100
    y = rng.binomial(1, 0.3, size=n)

    # Create synthetic feature dataframe with a canary feature equal to label
    df = pd.DataFrame(
        {
            "feat_legit": rng.randn(n),
            "canary_label_leak": y.astype(float),  # Exactly the target
            "at_risk": y,
        }
    )

    # The audit must flag the canary feature (> 0.95 correlation)
    audit_res = audit_leakage(
        df,
        feature_names=["feat_legit", "canary_label_leak"],
        label_col="at_risk",
        threshold=0.95,
    )
    assert audit_res["status"] == "FAIL"
    assert len(audit_res["high_correlation_features"]) > 0
    assert audit_res["high_correlation_features"][0][0] == "canary_label_leak"

    # Now verify stacking ensemble runs on mock candidate models
    mock_candidates = [
        {
            "spec": {
                "id": f"model_{i}",
                "display_name": f"Model {i}",
                "category": "classical",
            },
            "oof_predictions": rng.uniform(0.1, 0.9, size=n).tolist(),
            "leaderboard_metrics": {
                "leaderboard_score": 0.8 - i * 0.05,
                "val_f1_optimal": 0.75,
                "val_auc_roc": 0.85,
                "train_f1_mean": 0.80,
            },
            "fold_metrics": [{"fold": 0, "train_f1": 0.8, "val_f1": 0.75}],
        }
        for i in range(5)
    ]
    cv_folds = [
        {"fold": 0, "train_idx": list(range(0, 50)), "val_idx": list(range(50, 100))},
        {"fold": 1, "train_idx": list(range(50, 100)), "val_idx": list(range(0, 50))},
    ]

    stack_res = evaluate_stacking_ensemble(
        mock_candidates, y, cv_folds=cv_folds, seed=42
    )
    assert stack_res["spec"]["id"] == "stacking"
    assert len(stack_res["oof_predictions"]) == n
    assert 0.0 <= stack_res["leaderboard_metrics"]["leaderboard_score"] <= 1.0


def test_voting_soft_ensemble() -> None:
    """Test soft voting ensemble averages constituent probabilities correctly."""
    n = 50
    y = np.random.binomial(1, 0.3, size=n)
    p1 = np.full(n, 0.2)
    p2 = np.full(n, 0.4)
    p3 = np.full(n, 0.6)

    candidates = [
        {
            "spec": {"id": f"m{i}", "display_name": f"M{i}", "category": "boosting"},
            "oof_predictions": [p1, p2, p3][i].tolist(),
            "leaderboard_metrics": {
                "leaderboard_score": 0.8 - i * 0.05,
                "val_f1_optimal": 0.75,
                "val_auc_roc": 0.85,
                "train_f1_mean": 0.80,
            },
            "fold_metrics": [{"fold": 0, "train_f1": 0.8, "val_f1": 0.75}],
        }
        for i in range(3)
    ]

    voting_res = evaluate_voting_ensemble(candidates, y)
    assert voting_res["spec"]["id"] == "voting_soft"
    # Average of 0.2, 0.4, 0.6 is 0.4
    np.testing.assert_allclose(voting_res["oof_predictions"], 0.4, atol=1e-5)


def test_tune_model_and_registry() -> None:
    """Test tuner optimization on quick mock search space and estimator creation across families."""
    from mei.models.registry import create_base_estimator
    from mei.models.tuner import tune_model

    all_specs = get_all_model_specs()
    assert len(all_specs) >= 25

    # Check estimator creation across diverse model families
    for m_id in [
        "logistic_regression_l2",
        "svm_rbf",
        "random_forest",
        "lightgbm",
        "catboost",
    ]:
        spec = all_specs[m_id]
        est = create_base_estimator(spec, hyperparams=None, random_seed=42)
        assert est is not None

    # Test tuning logic with 2 trials on decision tree
    dt_spec = all_specs["decision_tree"]
    rng = np.random.RandomState(42)
    X = rng.randn(60, 5)
    y = rng.binomial(1, 0.4, size=60)
    best_params, meta = tune_model(dt_spec, X, y, cv_folds=3, seed=42, max_trials=2)
    assert isinstance(best_params, dict)
    assert meta["n_trials_completed"] == 2


def test_leaderboard_assembly_and_figures(tmp_path: Path) -> None:
    """Test leaderboard assembly, ranking, and figure rendering functions on mock results."""
    from mei.models.engine import assemble_and_save_leaderboard, save_model_artifacts
    from mei.models.figures import (
        plot_generalization_gap,
        plot_leaderboard_bars,
        plot_metric_heatmap,
    )

    rng = np.random.RandomState(42)
    mock_results = []
    for i, cat in enumerate(["baseline", "linear", "classical", "boosting", "hybrid"]):
        res = {
            "spec": {
                "id": f"model_{cat}",
                "display_name": f"Model {cat.capitalize()}",
                "plain_name": f"Plain {cat}",
                "category": cat,
                "task": "classification",
                "feature_set": "agg",
                "needs_scaling": False,
                "supports_class_weight": True,
                "twin_model_id": None,
            },
            "hyperparameters": {"C": 1.0},
            "tuning_summary": {"n_trials_completed": 0},
            "leaderboard_metrics": {
                "leaderboard_score": 0.85 - i * 0.05,
                "generalization_gap": 0.02 + i * 0.02,
                "overfit_penalty": 0.0,
                "overfit_status": "Good fit" if i < 2 else "Slight overfit",
                "train_f1_mean": 0.82,
                "train_auc_mean": 0.88,
                "val_f1_at_optimal": 0.80 - i * 0.04,
                "val_f1_at_t05": 0.78 - i * 0.04,
                "val_auc_roc": 0.86 - i * 0.03,
                "val_pr_auc": 0.82 - i * 0.03,
                "val_balanced_accuracy": 0.80,
                "val_mcc": 0.65,
                "val_accuracy": 0.85,
                "val_precision": 0.80,
                "val_recall": 0.80,
                "val_specificity": 0.88,
                "val_macro_f1": 0.82,
                "val_cohen_kappa": 0.66,
                "val_brier_score": 0.12,
                "val_log_loss": 0.35,
                "val_ece": 0.04,
                "optimal_threshold": 0.48,
                "training_time_seconds": 2.5,
                "inference_time_ms_per_1000": 3.0,
                "serialized_size_bytes": 10000,
            },
            "fold_metrics": [{"fold": 0, "train_f1": 0.82, "val_f1": 0.80}],
            "confusion_matrix_raw": [[70, 10], [5, 15]],
            "confusion_matrix_norm": [[0.875, 0.125], [0.25, 0.75]],
            "curves": {
                "roc_fpr": [0.0, 0.2, 1.0],
                "roc_tpr": [0.0, 0.8, 1.0],
                "pr_precision": [0.8, 0.7, 0.3],
                "pr_recall": [0.0, 0.8, 1.0],
                "cal_prob_true": [0.1, 0.5, 0.9],
                "cal_prob_pred": [0.12, 0.48, 0.88],
            },
            "oof_predictions": rng.uniform(0.1, 0.9, size=100).tolist(),
            "final_pipeline": None,
        }
        mock_results.append(res)
        save_model_artifacts(res, tmp_path / "models")

    # Assemble leaderboard
    lb_df, per_cat = assemble_and_save_leaderboard(mock_results, tmp_path)
    assert len(lb_df) == 5
    assert lb_df.iloc[0]["rank"] == 1
    assert "baseline" in per_cat
    assert "boosting" in per_cat

    # Render figures
    fig_dir = tmp_path / "figures"
    plot_leaderboard_bars(lb_df, [fig_dir])
    plot_generalization_gap(lb_df, [fig_dir])
    plot_metric_heatmap(lb_df, [fig_dir])

    assert (fig_dir / "class_01_leaderboard_bars.png").exists()
    assert (fig_dir / "class_02_generalization_gap.png").exists()
    assert (fig_dir / "class_03_metric_heatmap.png").exists()
