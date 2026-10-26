"""Unit and anti-leakage tests for preprocessing, reliability, and splitting."""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.splits import create_stratified_splits, load_split, SealedTestAccessError
from src.data.preprocess import (
    clean_dataset,
    compare_imputation_strategies,
    fit_and_save_preprocessor,
    load_preprocessor,
    PsychometricFeatureTransformer,
)
from src.features.reliability import (
    calculate_cronbach_alpha,
    calculate_mcdonald_omega,
    calculate_item_analysis,
    run_full_reliability_analysis,
)
from src.features.build_features import (
    RAW_ITEM_COLS,
    FACTOR_COLS,
    INTERACTION_COLS,
    compute_factor_scores,
    compute_risk_oriented_factors,
    compute_interaction_features,
    extract_feature_set,
    extract_targets,
    engineer_all_features,
)


@pytest.fixture
def synthetic_sample_df() -> pd.DataFrame:
    """Generate a controlled 50-row DataFrame with known psychometric structure."""
    rng = np.random.default_rng(42)
    n = 60
    data = {"study_id": [f"ID_{i:03d}" for i in range(n)]}
    for col in RAW_ITEM_COLS:
        data[col] = rng.integers(1, 6, size=n)

    # Inject one straight-liner
    for col in RAW_ITEM_COLS:
        data[col][0] = 4

    df = pd.DataFrame(data)
    df["risk_category"] = rng.choice(["Low", "Moderate", "High"], size=n)
    df["pri_true"] = rng.uniform(10.0, 90.0, size=n)
    df["attrition_label"] = rng.choice([0, 1], p=[0.75, 0.25], size=n)
    return df


def test_sealed_test_set_access_guard(tmp_path, synthetic_sample_df):
    """Ensure accessing test set without explicit allow_test=True raises SealedTestAccessError."""
    create_stratified_splits(
        synthetic_sample_df,
        stratify_col="risk_category",
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        seed=42,
        output_dir=str(tmp_path),
    )

    # 1. Attempting unauthorized access MUST fail
    with pytest.raises(SealedTestAccessError, match="TEST SET ACCESS VIOLATION"):
        load_split("test", processed_dir=str(tmp_path), allow_test=False)

    # 2. Authorized access for final verification must succeed
    test_df = load_split("test", processed_dir=str(tmp_path), allow_test=True)
    assert isinstance(test_df, pd.DataFrame)
    assert len(test_df) > 0


def test_zero_data_leakage_imputer_and_scaler(tmp_path, synthetic_sample_df):
    """Strict verification: imputer and scaler statistics equal train stats ONLY."""
    train_df, val_df, _ = create_stratified_splits(
        synthetic_sample_df,
        stratify_col="risk_category",
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        seed=42,
        output_dir=str(tmp_path),
    )

    # Fit preprocessor on train_df ONLY
    model_path = str(tmp_path / "preprocessor.joblib")
    preprocessor = fit_and_save_preprocessor(
        train_df,
        output_path=model_path,
        imputation_strategy="median",
        include_scaler=False,
    )

    # Verify imputer statistics exactly equal training medians
    expected_train_medians = train_df[RAW_ITEM_COLS].median().values
    fitted_imputer_stats = preprocessor.named_steps["imputer"].statistics_
    np.testing.assert_array_equal(fitted_imputer_stats, expected_train_medians)

    # Verify that fitting scaler on train produces train mean/scale, NOT val mean/scale
    _, fitted_scaler = engineer_all_features(train_df, fit_scaler=True)
    cols_scaled = FACTOR_COLS + [f"risk_{c.replace('factor_', '')}" for c in FACTOR_COLS] + INTERACTION_COLS
    expected_train_means = compute_interaction_features(train_df)[cols_scaled].mean().values

    np.testing.assert_allclose(fitted_scaler.mean_, expected_train_means, rtol=1e-5)


def test_straight_liner_removal(synthetic_sample_df):
    """Verify that respondents with zero variance across all 15 items are correctly isolated and dropped."""
    clean_df, audit = clean_dataset(synthetic_sample_df, drop_straight_liners=True, validate=False)
    assert audit["straight_liners_removed"] >= 1
    assert "ID_000" not in clean_df["study_id"].values
    assert len(clean_df) == len(synthetic_sample_df) - audit["straight_liners_removed"]


def test_cronbach_alpha_calculation():
    """Verify Cronbach's alpha on perfectly correlated vs. random items."""
    # Perfectly correlated items -> alpha == 1.0
    perfect_df = pd.DataFrame({
        "Q1": [1, 2, 3, 4, 5],
        "Q2": [1, 2, 3, 4, 5],
        "Q3": [1, 2, 3, 4, 5],
    })
    alpha_perfect = calculate_cronbach_alpha(perfect_df)
    assert np.isclose(alpha_perfect, 1.0, atol=1e-4)

    # Low correlation items
    noisy_df = pd.DataFrame({
        "Q1": [1, 5, 2, 4, 3],
        "Q2": [5, 1, 4, 2, 3],
        "Q3": [2, 3, 1, 5, 4],
    })
    alpha_noisy = calculate_cronbach_alpha(noisy_df)
    assert alpha_noisy < 0.50


def test_mcdonald_omega_and_item_analysis():
    """Verify McDonald's omega computation and item-total correlations."""
    df = pd.DataFrame({
        "S1": [1, 2, 3, 4, 5, 2, 3, 4, 5, 1],
        "S2": [1, 2, 3, 4, 5, 1, 3, 4, 4, 2],
        "S3": [2, 2, 3, 5, 5, 2, 3, 4, 5, 1],
    })
    omega = calculate_mcdonald_omega(df)
    assert 0.0 <= omega <= 1.0

    analysis = calculate_item_analysis(df, ["S1", "S2", "S3"])
    assert "S1" in analysis
    assert "alpha_if_deleted" in analysis["S1"]
    assert "corrected_item_total_r" in analysis["S1"]
    assert analysis["S1"]["corrected_item_total_r"] > 0.0


def test_feature_engineering_sets(synthetic_sample_df):
    """Verify that extract_feature_set correctly subsets feature dimensions."""
    engineered_df, _ = engineer_all_features(synthetic_sample_df, fit_scaler=False)

    raw_set = extract_feature_set(engineered_df, "raw_items")
    assert raw_set.shape == (len(synthetic_sample_df), 15)

    factor_set = extract_feature_set(engineered_df, "factor_scores")
    assert factor_set.shape == (len(synthetic_sample_df), 5)

    interaction_set = extract_feature_set(engineered_df, "interactions")
    assert interaction_set.shape == (len(synthetic_sample_df), 5)

    combined_set = extract_feature_set(engineered_df, "combined")
    assert combined_set.shape == (len(synthetic_sample_df), 25)


def test_targets_extraction(synthetic_sample_df):
    """Verify target extraction returns continuous PRI, categorical risk, and binary attrition."""
    targets = extract_targets(synthetic_sample_df)
    assert "pri_score" in targets
    assert "risk_category" in targets
    assert "attrition_label" in targets

    assert targets["pri_score"].between(0.0, 100.0).all()
    assert set(targets["risk_category"].dropna().unique()).issubset({"Low", "Moderate", "High"})
    assert set(targets["attrition_label"].dropna().unique()).issubset({0, 1})


def test_joblib_preprocessor_reproducibility(tmp_path, synthetic_sample_df):
    """Verify loaded preprocessor produces identical output on unobserved records."""
    model_path = str(tmp_path / "preprocessor.joblib")
    preprocessor = fit_and_save_preprocessor(
        synthetic_sample_df,
        output_path=model_path,
        imputation_strategy="median",
        include_scaler=False,
    )
    loaded = load_preprocessor(model_path)

    new_data = synthetic_sample_df[RAW_ITEM_COLS].head(10).copy()
    out1 = preprocessor.transform(new_data)
    out2 = loaded.transform(new_data)

    pd.testing.assert_frame_equal(out1, out2)
