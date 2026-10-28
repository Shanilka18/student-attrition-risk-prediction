"""Tests for Feature Transformers, Splits, Leakage Guards, and Sanity Checks."""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from mei.features.diagnostics import audit_leakage
from mei.features.pipeline import (
    build_feature_pipeline,
    compute_cohort_class_weights,
    get_canonical_28_columns,
)
from mei.features.splits import (
    create_cv_folds,
    create_outer_split,
    load_test_indices,
    load_train_indices,
    load_val_indices,
    save_cv_folds,
    save_splits,
)
from mei.features.transformers import (
    CANONICAL_INDICATORS,
    PERIODS,
    EngagementFeatures,
    EngagementImputer,
    PeriodReshaper,
)


def _make_dummy_28_df(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic dummy survey features for testing."""
    rng = np.random.RandomState(seed)
    cols = get_canonical_28_columns()
    data = rng.randint(1, 6, size=(n_samples, 28)).astype(float)
    df = pd.DataFrame(data, columns=cols)
    df["student_id"] = [f"STU_{i:04d}" for i in range(n_samples)]
    df["at_risk"] = rng.choice([0, 1], size=n_samples, p=[0.75, 0.25])
    return df


def test_period_reshaper_shape_and_inverse() -> None:
    """PeriodReshaper must transform (n, 28) <-> (n, 4, 7) losslessly."""
    df = _make_dummy_28_df(n_samples=20)
    reshaper = PeriodReshaper()

    tensor = reshaper.fit_transform(df)
    assert tensor.ndim == 3
    assert tensor.shape == (20, 4, 7)

    # Verify period 1 value of first indicator matches df
    col_p1_ind0 = f"{CANONICAL_INDICATORS[0]}_{PERIODS[0]}"
    np.testing.assert_array_equal(tensor[:, 0, 0], df[col_p1_ind0].values)

    # Invert back to tabular
    inverted_df = reshaper.inverse_transform(tensor)
    assert inverted_df.shape == (20, 28)
    for col in get_canonical_28_columns():
        np.testing.assert_array_equal(inverted_df[col].values, df[col].values)


def test_engagement_imputer_median_and_missing_indicators() -> None:
    """EngagementImputer must learn medians and flag high-missingness columns."""
    df = _make_dummy_28_df(n_samples=100)

    # Inject missing values: col 0 has 10% missing, col 1 has 1% missing
    col_high = get_canonical_28_columns()[0]
    col_low = get_canonical_28_columns()[1]
    df.loc[:9, col_high] = np.nan
    df.loc[:0, col_low] = np.nan

    imputer = EngagementImputer(method="median", missing_threshold=0.03)
    out_df = imputer.fit_transform(df[get_canonical_28_columns()])

    assert not out_df[get_canonical_28_columns()].isna().any().any()
    assert f"missing_{col_high}" in out_df.columns
    assert f"missing_{col_low}" not in out_df.columns
    assert out_df[f"missing_{col_high}"].sum() == 10.0


def test_engagement_imputer_student_interpolation() -> None:
    """Student interpolation must interpolate across periods linearly."""
    df = _make_dummy_28_df(n_samples=10)
    # Student 0, indicator 0: p1=1, p2=NaN, p3=3, p4=4
    p1 = f"{CANONICAL_INDICATORS[0]}_p1"
    p2 = f"{CANONICAL_INDICATORS[0]}_p2"
    p3 = f"{CANONICAL_INDICATORS[0]}_p3"
    p4 = f"{CANONICAL_INDICATORS[0]}_p4"

    df.loc[0, p1] = 1.0
    df.loc[0, p2] = np.nan
    df.loc[0, p3] = 3.0
    df.loc[0, p4] = 4.0

    imputer = EngagementImputer(method="student_interpolation")
    out_df = imputer.fit_transform(df[get_canonical_28_columns()])

    assert not out_df.isna().any().any()
    # Interpolated value between 1 and 3 should be 2.0
    assert out_df.loc[0, p2] == 2.0


def test_engagement_features_dimensions_and_math() -> None:
    """EngagementFeatures must calculate 56 indicator dynamics + 12 composites."""
    df = _make_dummy_28_df(n_samples=25)

    feat_eng = EngagementFeatures(include_raw=True)
    out_df = feat_eng.fit_transform(df[get_canonical_28_columns()])

    # 28 raw + 56 indicator + 12 composite = 96 features
    assert out_df.shape == (25, 96)

    # Test without raw
    feat_eng_no_raw = EngagementFeatures(include_raw=False)
    out_no_raw = feat_eng_no_raw.fit_transform(df[get_canonical_28_columns()])
    assert out_no_raw.shape == (25, 68)

    # Verify composite bounded [0, 100]
    for p in PERIODS:
        assert (out_df[f"composite_{p}"] >= 0.0).all()
        assert (out_df[f"composite_{p}"] <= 100.0).all()

    assert (out_df["composite_mean"] >= 0.0).all()
    assert (out_df["composite_mean"] <= 100.0).all()


def test_splits_outer_partition_and_disjointness(tmp_path: Path) -> None:
    """Outer split must be 70/15/15, strictly disjoint, and stratified."""
    df = _make_dummy_28_df(n_samples=1000)

    tr_idx, va_idx, te_idx = create_outer_split(
        df, stratify_col="at_risk", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    assert len(tr_idx) == 700
    assert len(va_idx) == 150
    assert len(te_idx) == 150

    # Strict disjointness
    assert len(set(tr_idx).intersection(set(va_idx))) == 0
    assert len(set(tr_idx).intersection(set(te_idx))) == 0
    assert len(set(va_idx).intersection(set(te_idx))) == 0

    # Save and reload train and val
    _ = save_splits(
        splits_dir=tmp_path,
        df=df,
        train_idx=tr_idx,
        val_idx=va_idx,
        test_idx=te_idx,
    )
    loaded_tr = load_train_indices(tmp_path)
    loaded_va = load_val_indices(tmp_path)
    np.testing.assert_array_equal(tr_idx, loaded_tr)
    np.testing.assert_array_equal(va_idx, loaded_va)


def test_guard_blocks_reading_test_indices(tmp_path: Path) -> None:
    """load_test_indices must raise PermissionError if unauthorized."""
    df = _make_dummy_28_df(n_samples=100)
    tr_idx, va_idx, te_idx = create_outer_split(df)
    save_splits(tmp_path, df, tr_idx, va_idx, te_idx)

    # 1. Unauthorized caller must raise PermissionError
    def unauthorized_caller() -> None:
        load_test_indices(tmp_path)

    with pytest.raises(PermissionError) as exc_info:
        unauthorized_caller()

    assert "Target Leakage Guard Violation" in str(exc_info.value)

    # 2. Designated Phase 9 caller must succeed
    def evaluate_final_on_test() -> np.ndarray:
        return load_test_indices(tmp_path)

    res_auth = evaluate_final_on_test()
    np.testing.assert_array_equal(te_idx, res_auth)

    # 3. Explicit override flag must succeed
    res_override = load_test_indices(tmp_path, allow_override=True)
    np.testing.assert_array_equal(te_idx, res_override)


def test_cv_folds_repeated_kfold_count(tmp_path: Path) -> None:
    """CV scheme must produce 15 folds for N >= 500 and 25 folds for N < 500."""
    df_large = _make_dummy_28_df(n_samples=600)
    folds_large = create_cv_folds(df_large, n_splits=5, n_repeats=3)
    assert len(folds_large) == 15

    # Check N < 500 switches to 25 folds
    df_small = _make_dummy_28_df(n_samples=400)
    folds_small = create_cv_folds(df_small, n_splits=5, n_repeats=3)
    assert len(folds_small) == 25

    # Test serialization
    save_cv_folds(tmp_path, folds_large)
    from mei.features.splits import load_cv_folds

    reloaded_folds = load_cv_folds(tmp_path)
    assert len(reloaded_folds) == 15
    np.testing.assert_array_equal(folds_large[0][0], reloaded_folds[0][0])


def test_pipeline_no_data_leakage_on_test_transform() -> None:
    """Pipeline fitted on train must NOT refit learned parameters on test transform."""
    df_train = _make_dummy_28_df(n_samples=200, seed=1)
    df_test = _make_dummy_28_df(n_samples=50, seed=2)

    pipe = build_feature_pipeline(feature_set="agg", imputer_method="median")
    pipe.fit(df_train[get_canonical_28_columns()])

    # Record learned medians
    learned_medians = dict(pipe.named_steps["imputer"].medians_)

    # Transform test set with different distribution
    pipe.transform(df_test[get_canonical_28_columns()])

    # Verify learned medians did not change
    assert pipe.named_steps["imputer"].medians_ == learned_medians


def test_audit_leakage_guard() -> None:
    """audit_leakage must flag outcome columns and features with |r| > 0.95."""
    df = _make_dummy_28_df(n_samples=100)
    clean_features = get_canonical_28_columns()

    # Clean audit must pass
    res_clean = audit_leakage(df, feature_names=clean_features)
    assert res_clean["status"] == "PASS"

    # Injecting forbidden outcome column must fail
    res_forbidden = audit_leakage(
        df, feature_names=clean_features + ["attrition_intent_score"]
    )
    assert res_forbidden["status"] == "FAIL"
    assert "attrition_intent_score" in res_forbidden["forbidden_columns_found"]

    # Injecting column with r = 0.99 must fail
    df["leaked_col"] = df["at_risk"] * 10.0 + np.random.normal(0, 0.01, size=len(df))
    res_corr = audit_leakage(df, feature_names=clean_features + ["leaked_col"])
    assert res_corr["status"] == "FAIL"


def test_shuffled_label_sanity_test() -> None:
    """When labels are randomly shuffled, CV AUC must be approx 0.50 (+-0.08)."""
    df = _make_dummy_28_df(n_samples=400, seed=42)
    raw_28 = get_canonical_28_columns()

    # Shuffle labels randomly
    rng = np.random.RandomState(99)
    y_shuffled = rng.permutation(df["at_risk"].values)

    pipe = build_feature_pipeline(feature_set="agg", scale=True)
    X = pipe.fit_transform(df[raw_28])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs: List[float] = []

    for tr_idx, va_idx in skf.split(X, y_shuffled):
        clf = LogisticRegression(class_weight="balanced", random_state=42)
        clf.fit(X[tr_idx], y_shuffled[tr_idx])
        p = clf.predict_proba(X[va_idx])[:, 1]
        aucs.append(roc_auc_score(y_shuffled[va_idx], p))

    mean_auc = float(np.mean(aucs))
    # Should be around 0.50
    assert 0.42 <= mean_auc <= 0.58, (
        f"Shuffled label sanity test failed: AUC = {mean_auc:.4f} outside [0.42, 0.58]"
    )


def test_class_weights_computation() -> None:
    """Cohort class weights must balance binary classes inversely to prevalence."""
    y = np.array([0] * 75 + [1] * 25)
    weights = compute_cohort_class_weights(y)

    assert 0 in weights and 1 in weights
    # w0 = 100 / (2 * 75) = 2/3 ≈ 0.667; w1 = 100 / (2 * 25) = 2.0
    assert np.isclose(weights[0], 100.0 / (2 * 75), atol=1e-3)
    assert np.isclose(weights[1], 100.0 / (2 * 25), atol=1e-3)


def test_column_selector_and_constant_filter() -> None:
    """ColumnSelector and ConstantNearConstantFilter must filter correctly."""
    from mei.features.pipeline import ColumnSelector
    from mei.features.transformers import ConstantNearConstantFilter

    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [5.0, 5.0, 5.0],  # Constant
            "c": [10.0, 20.0, 30.0],
        }
    )

    selector = ColumnSelector(["a", "b"])
    sel_df = selector.fit_transform(df)
    assert list(sel_df.columns) == ["a", "b"]

    flt = ConstantNearConstantFilter()
    clean_df = flt.fit_transform(df)
    assert "b" in flt.dropped_columns_
    assert "b" not in clean_df.columns
    assert list(clean_df.columns) == ["a", "c"]


def test_diagnostics_and_vif_reports() -> None:
    """Diagnostics and VIF reporting must compute scores without crash."""
    from mei.features.diagnostics import compute_feature_diagnostics, compute_vif_report

    df = _make_dummy_28_df(n_samples=50)
    cols = get_canonical_28_columns()[:8]

    diag_df = compute_feature_diagnostics(df, feature_cols=cols, label_col="at_risk")
    assert len(diag_df) == 8
    assert "mutual_info" in diag_df.columns
    assert "cohens_d" in diag_df.columns
    assert "spearman_corr" in diag_df.columns

    vif_df = compute_vif_report(df[cols])
    assert "VIF" in vif_df.columns
    assert len(vif_df) == 8


def test_extract_features_and_figure_rendering(tmp_path: Path) -> None:
    """extract_features and figure rendering must export files cleanly."""
    from mei.features.diagnostics import (
        compute_feature_diagnostics,
        render_feature_diagnostic_figures,
    )
    from mei.features.pipeline import extract_features

    df = _make_dummy_28_df(n_samples=60)
    feats_agg, pipe = extract_features(df, feature_set="agg")
    assert feats_agg.shape[0] == 60
    assert feats_agg.shape[1] >= 96

    # Transform without refitting
    df_new = _make_dummy_28_df(n_samples=10, seed=999)
    feats_new, _ = extract_features(df_new, feature_set="agg", fitted_pipeline=pipe)
    assert feats_new.shape[0] == 10

    # Test rendering figures
    diag_df = compute_feature_diagnostics(
        df.assign(**feats_agg),
        feature_cols=list(feats_agg.columns[:10]),
        label_col="at_risk",
    )
    dummy_bench = {
        "lr_raw28": {
            "auc_mean": 0.86,
            "auc_ci95": 0.01,
            "f1_mean": 0.65,
            "f1_ci95": 0.01,
        },
        "lr_agg": {
            "auc_mean": 0.87,
            "auc_ci95": 0.01,
            "f1_mean": 0.66,
            "f1_ci95": 0.01,
        },
        "lgbm_raw28": {
            "auc_mean": 0.86,
            "auc_ci95": 0.01,
            "f1_mean": 0.66,
            "f1_ci95": 0.01,
        },
        "lgbm_agg": {
            "auc_mean": 0.87,
            "auc_ci95": 0.01,
            "f1_mean": 0.66,
            "f1_ci95": 0.01,
        },
    }

    fig_files = render_feature_diagnostic_figures(
        df=df,
        feature_df=feats_agg,
        diagnostics_df=diag_df,
        benchmark_summary=dummy_bench,
        figures_dir=tmp_path / "figs",
    )
    assert len(fig_files) == 8  # 4 PNG + 4 SVG
    for f in fig_files:
        assert Path(f).exists()


def test_cv_benchmark_execution() -> None:
    """run_cv_benchmark must evaluate 5 models across given folds without crash."""
    from mei.features.diagnostics import run_cv_benchmark

    df = _make_dummy_28_df(n_samples=40)
    # Define 2 small test folds
    folds = [
        (np.arange(20), np.arange(20, 40)),
        (np.arange(20, 40), np.arange(20)),
    ]

    res = run_cv_benchmark(df, folds=folds, label_col="at_risk", seed=42)
    assert "lr_raw28" in res
    assert "lr_agg" in res
    assert "lgbm_raw28" in res
    assert "lgbm_agg" in res
    assert "lr_agg_interp" in res
    assert "auc_mean" in res["lr_raw28"]


def test_pipeline_feature_sets_and_smote() -> None:
    """build_feature_pipeline must handle single_signal, sequence, smote, and errors."""
    from mei.features.pipeline import build_feature_pipeline

    df = _make_dummy_28_df(n_samples=30)
    raw_cols = get_canonical_28_columns()

    # 1. single_signal
    p_single = build_feature_pipeline(feature_set="single_signal")
    res_single = p_single.fit_transform(df[raw_cols])
    assert res_single.shape == (30, 4)

    # 2. sequence
    p_seq = build_feature_pipeline(feature_set="sequence")
    res_seq = p_seq.fit_transform(df[raw_cols])
    assert res_seq.shape == (30, 4, 7)

    # 3. SMOTE
    p_smote = build_feature_pipeline(feature_set="raw28", add_smote=True)
    X_res, y_res = p_smote.fit_resample(df[raw_cols], df["at_risk"].values)
    assert len(X_res) >= len(df)
    assert len(y_res) >= len(df)

    # 4. Unknown feature set
    with pytest.raises(ValueError):
        build_feature_pipeline(feature_set="nonexistent")
