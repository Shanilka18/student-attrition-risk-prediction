"""Unit and statistical validation tests for the synthetic data generator."""

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data_gen.generate import SocioeconomicDGP, SurveyCorruptionInjector


def test_generation_sample_size_and_columns():
    """Verify clean dataset generates requested sample size with expected columns."""
    dgp = SocioeconomicDGP(seed=42)
    df, _ = dgp.generate(n=1000)

    assert len(df) == 1000, f"Expected 1000 rows, got {len(df)}"
    assert "participant_id" in df.columns
    assert "attrition" in df.columns
    assert "latent_risk_score" in df.columns
    assert "intention_to_withdraw" in df.columns

    # Verify ID formatting: P-00001
    assert df["participant_id"].str.match(r"^P-\d{5}$").all()


def test_attrition_prevalence_bounds():
    """Verify class imbalance: attrition prevalence must strictly be in 12% - 20% range."""
    dgp = SocioeconomicDGP(seed=42)
    df, _ = dgp.generate(n=2000)

    prevalence = df["attrition"].mean()
    assert (
        0.12 <= prevalence <= 0.20
    ), f"Attrition prevalence {prevalence:.4f} outside required bounds [0.12, 0.20]"


def test_no_label_leakage():
    """Verify no single feature has unrealistic predictive power (target ROC-AUC 0.72 - 0.88).

    Guarantees no feature leaks ground-truth label directly.
    """
    dgp = SocioeconomicDGP(seed=42)
    df, _ = dgp.generate(n=2000)
    y = df["attrition"].values

    candidate_features = [
        "commute_time_min",
        "commute_cost_lkr_week",
        "financial_stress",
        "work_hours_per_week",
    ]

    for col in candidate_features:
        x = df[col].values
        # Compute single feature ROC-AUC
        auc = roc_auc_score(y, x)
        auc_directed = max(auc, 1.0 - auc)
        assert (
            auc_directed <= 0.88
        ), f"Feature '{col}' has excessive ROC-AUC ({auc_directed:.4f} > 0.88). Label leakage detected!"


def test_derived_exposures_logic():
    """Verify deterministic logic of the four planted policy exposures."""
    dgp = SocioeconomicDGP(seed=42)
    df, _ = dgp.generate(n=1000)

    # 1. financial_hardship: income <= 2 AND stress >= 4
    expected_fh = (
        (df["household_income_bracket"] <= 2) & (df["financial_stress"] >= 4)
    ).astype(int)
    pd.testing.assert_series_equal(
        df["financial_hardship"], expected_fh, check_names=False
    )

    # 2. long_commute: commute_time_min > 120
    expected_lc = (df["commute_time_min"] > 120.0).astype(int)
    pd.testing.assert_series_equal(df["long_commute"], expected_lc, check_names=False)

    # 3. inadequate_digital_access: reliability <= 2 OR device in ('phone only', 'none')
    expected_ida = (
        (df["internet_reliability"] <= 2)
        | df["device_type"].isin(["phone only", "none"])
    ).astype(int)
    pd.testing.assert_series_equal(
        df["inadequate_digital_access"], expected_ida, check_names=False
    )

    # 4. heavy_part_time_work: work_hours_per_week > 15
    expected_hpt = (df["work_hours_per_week"] > 15.0).astype(int)
    pd.testing.assert_series_equal(
        df["heavy_part_time_work"], expected_hpt, check_names=False
    )


def test_planted_effects_ground_truth_structure():
    """Verify ground truth JSON contains all planted ATEs and heterogeneity."""
    dgp = SocioeconomicDGP(seed=42)
    _, gt = dgp.generate(n=2000)

    assert "planted_causal_effects" in gt
    pce = gt["planted_causal_effects"]

    assert pce["financial_hardship"]["theoretical_ate_risk_difference"] == 0.10
    assert pce["long_commute"]["theoretical_ate_risk_difference"] == 0.06
    assert pce["inadequate_digital_access"]["theoretical_ate_risk_difference"] == 0.07
    assert pce["heavy_part_time_work"]["theoretical_ate_risk_difference"] == 0.05

    # Check planted heterogeneity
    het = pce["financial_hardship"]["planted_heterogeneity"]
    assert het["senior_rural_cate"] == 0.16
    assert het["junior_urban_cate"] == 0.07


def test_mess_injection_in_raw_dataset():
    """Verify that corruptions, missing values, duplicates, and impossible values exist in raw data."""
    dgp = SocioeconomicDGP(seed=42)
    df_clean, _ = dgp.generate(n=1000)

    injector = SurveyCorruptionInjector(seed=42)
    df_raw = injector.corrupt(df_clean)

    # 1. Row count should be greater due to appended duplicates
    assert len(df_raw) > len(df_clean), "Duplicates were not injected into raw dataset"

    # 2. Missing values exist
    has_nulls = df_raw.isna().sum().sum() > 0
    assert has_nulls, "Missing values were not injected into raw dataset"

    # 3. Sentinels exist in income or commute cost
    raw_income_vals = df_raw["household_income_bracket"].dropna().astype(str).values
    has_sentinel = any(s in ["Prefer not to say", "N/A"] for s in raw_income_vals)
    assert (
        has_sentinel
    ), "Sentinel values ('Prefer not to say' / 'N/A') missing from income column"

    # 4. Impossible values exist in commute time (-5 or 900)
    commute_raw = pd.to_numeric(df_raw["commute_time_min"], errors="coerce")
    has_negative_commute = (commute_raw < 0).any()
    has_extreme_commute = (commute_raw >= 800).any()
    assert (
        has_negative_commute or has_extreme_commute
    ), "Impossible commute times were not injected"

    # 5. Impossible work hours (> 80)
    hours_raw = pd.to_numeric(df_raw["work_hours_per_week"], errors="coerce")
    has_extreme_hours = (hours_raw > 80).any()
    assert has_extreme_hours, "Extreme work hours (> 80) were not injected"

    # 6. Logical contradiction: unemployed but work hours > 0
    contradictions = df_raw[(df_raw["part_time_employment"] == 0) & (hours_raw > 0)]
    assert (
        len(contradictions) > 0
    ), "Logical contradiction (unemployed with work hours) not injected"


def test_seed_reproducibility():
    """Verify exact bitwise reproducibility across runs with identical seed."""
    dgp1 = SocioeconomicDGP(seed=42)
    df1, gt1 = dgp1.generate(n=500)

    dgp2 = SocioeconomicDGP(seed=42)
    df2, gt2 = dgp2.generate(n=500)

    pd.testing.assert_frame_equal(df1, df2)
    assert gt1 == gt2
