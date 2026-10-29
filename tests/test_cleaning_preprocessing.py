"""Automated test suite verifying data cleaning, leakage prevention, in-fold preprocessing, and permutation integrity."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.cleaning.cleaner import SurveyCleaner
from src.data_gen.generate import SocioeconomicDGP, SurveyCorruptionInjector
from src.preprocessing.pipeline import (
    FORBIDDEN_COLUMNS,
    LeakageProofPreprocessor,
    get_scaled_pipeline,
    get_tree_pipeline,
    validate_feature_boundary,
)
from src.preprocessing.resampling import apply_smote_in_fold
from src.preprocessing.split import (
    SealedTestAccessViolation,
    create_grouped_stratified_split,
    get_sealed_test_set,
)


@pytest.fixture
def sample_raw_and_clean_data():
    """Generate synthetic sample for testing cleaning and preprocessing."""
    dgp = SocioeconomicDGP(seed=42)
    df_truth, _ = dgp.generate(n=600)
    injector = SurveyCorruptionInjector(seed=42)
    df_raw = injector.corrupt(df_truth)
    return df_raw, df_truth


def test_survey_cleaner_operations(sample_raw_and_clean_data):
    """Verify that cleaner removes duplicates, maps sentinels to NaN, clips bounds, and reconciles logic."""
    df_raw, _ = sample_raw_and_clean_data
    cleaner = SurveyCleaner()
    df_cleaned, summary = cleaner.clean(df_raw)

    # 1. Duplicates removed
    assert len(df_cleaned) < len(df_raw), "Duplicates were not removed"
    assert summary["duplicates_removed"] > 0

    # 2. Sentinels mapped to NaN
    income_vals = df_cleaned["household_income_bracket"].dropna().astype(str)
    assert not any(s in ["Prefer not to say", "N/A", "na"] for s in income_vals)

    # 3. Boundary clipping
    assert (df_cleaned["commute_time_min"] >= 8.0).all()
    assert (df_cleaned["commute_time_min"] <= 240.0).all()
    assert (df_cleaned["work_hours_per_week"] <= 50.0).all()

    # 4. Logical contradiction resolution
    unemployed_hours = df_cleaned[df_cleaned["part_time_employment"] == 0][
        "work_hours_per_week"
    ]
    assert (
        unemployed_hours == 0.0
    ).all(), "Unemployed students still have non-zero work hours!"


def test_sealed_test_set_security():
    """Verify that accessing the test set raises SealedTestAccessViolation unless explicitly allowed."""
    with pytest.raises(SealedTestAccessViolation):
        get_sealed_test_set(
            allow_test_access=False, caller_description="Unauthorized routine"
        )


def test_preprocessor_refuses_fit_without_infold_flag(sample_raw_and_clean_data):
    """Verify that pipeline raises ValueError if fit is attempted without is_train_fold=True."""
    _, df_truth = sample_raw_and_clean_data
    preprocessor = get_tree_pipeline()

    # Must raise error when is_train_fold=False
    with pytest.raises(ValueError, match="DATA LEAKAGE PREVENTION VIOLATION"):
        preprocessor.fit(df_truth, is_train_fold=False)


def test_no_nan_after_transformation(sample_raw_and_clean_data):
    """Verify that transformed feature matrices contain zero NaNs for both tree and scaled views."""
    df_raw, _ = sample_raw_and_clean_data
    cleaner = SurveyCleaner()
    df_cleaned, _ = cleaner.clean(df_raw)

    # Partition
    train_df = df_cleaned.iloc[:400]
    test_df = df_cleaned.iloc[400:]

    for view in ["tree", "scaled"]:
        preprocessor = LeakageProofPreprocessor(view_type=view)
        X_train_trans = preprocessor.fit_transform(train_df, is_train_fold=True)
        X_test_trans = preprocessor.transform(test_df)

        assert not np.isnan(
            X_train_trans
        ).any(), f"NaN found in {view} train transformation!"
        assert not np.isnan(
            X_test_trans
        ).any(), f"NaN found in {view} test transformation!"
        assert X_train_trans.shape[1] == X_test_trans.shape[1]


def test_feature_boundary_enforcement(sample_raw_and_clean_data):
    """Verify that validate_feature_boundary drops forbidden outcomes."""
    _, df_truth = sample_raw_and_clean_data
    assert "attrition" in df_truth.columns

    df_filtered = validate_feature_boundary(df_truth)
    for forbidden in FORBIDDEN_COLUMNS:
        assert (
            forbidden not in df_filtered.columns
        ), f"Forbidden column '{forbidden}' was not dropped!"


def test_permutation_leakage_test(sample_raw_and_clean_data):
    """Permutation test: Training a classifier on shuffled labels MUST yield validation ROC-AUC ≈ 0.50.

    Proves no pipeline stage leaks the label into the features.
    """
    _, df_truth = sample_raw_and_clean_data
    split = create_grouped_stratified_split(df_truth, seed=42)

    train_df = df_truth.iloc[split["train_indices"]].copy()
    val_df = df_truth.iloc[split["val_indices"]].copy()

    # Fit preprocessor strictly on train
    preprocessor = get_scaled_pipeline()
    X_train = preprocessor.fit_transform(train_df, is_train_fold=True)
    X_val = preprocessor.transform(val_df)

    # Average across 5 random permutations to assess expected null distribution
    aucs = []
    for s in range(5):
        rng = np.random.default_rng(s)
        y_train_shuffled = rng.permutation(train_df["attrition"].values)
        clf = LogisticRegression(random_state=42, max_iter=200)
        clf.fit(X_train, y_train_shuffled)
        val_preds = clf.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(val_df["attrition"].values, val_preds))

    mean_auc = float(np.mean(aucs))
    # Shuffled label performance must be near random chance ~0.50
    assert 0.40 <= mean_auc <= 0.60, (
        f"Permutation test failed: Mean shuffled label ROC-AUC is {mean_auc:.4f} (expected ~0.50). "
        "Potential label leakage detected!"
    )


def test_infold_smote_resampling():
    """Verify that in-fold SMOTE balances the minority class to target ratio."""
    rng = np.random.default_rng(42)
    # Create imbalanced synthetic dataset (100 negative, 20 positive)
    X = rng.normal(0, 1, size=(120, 5))
    y = np.array([0] * 100 + [1] * 20)

    _, y_res = apply_smote_in_fold(X, y, target_ratio=0.50, seed=42)

    assert len(y_res) > len(y)
    n_pos = int((y_res == 1).sum())
    n_neg = int((y_res == 0).sum())
    ratio = n_pos / n_neg
    assert abs(ratio - 0.50) < 0.05, f"Expected ratio ~0.50, got {ratio:.2f}"
