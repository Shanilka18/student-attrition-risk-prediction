"""Unit tests for synthetic survey data generation and leakage guards."""

import pandas as pd

from mei.config import DATA_DIR
from mei.generator import SyntheticSurveyGenerator
from mei.io.schema import load_schema


def test_generator_seed_reproducibility() -> None:
    """Same random seed must produce byte-identical cohorts and targets."""
    gen1 = SyntheticSurveyGenerator(seed=42)
    raw1, hidden1, stats1 = gen1.generate_all(calibrate=False)

    gen2 = SyntheticSurveyGenerator(seed=42)
    raw2, hidden2, stats2 = gen2.generate_all(calibrate=False)

    pd.testing.assert_frame_equal(raw1, raw2)
    pd.testing.assert_frame_equal(hidden1, hidden2)
    assert stats1.metrics == stats2.metrics


def test_sample_cohort_dimensions_and_prevalence() -> None:
    """N students and at-risk prevalence must fall within configured bounds."""
    hidden_path = DATA_DIR / "raw" / "ground_truth_hidden.csv"
    assert hidden_path.exists(), "Ground truth hidden file does not exist"

    df_hidden = pd.read_csv(hidden_path)
    assert len(df_hidden) == 2400
    assert "at_risk" in df_hidden.columns
    assert "attrition_intent_score" in df_hidden.columns

    prevalence = df_hidden["at_risk"].mean()
    assert 0.24 <= prevalence <= 0.32, (
        f"Prevalence {prevalence:.3f} outside [0.24, 0.32]"
    )

    # Score range check
    assert df_hidden["attrition_intent_score"].min() >= 0.0
    assert df_hidden["attrition_intent_score"].max() <= 100.0


def test_raw_survey_export_format() -> None:
    """Raw Google Forms export must contain 35 canonical columns + duplicates."""
    raw_csv_path = DATA_DIR / "raw" / "survey_raw_sample.csv"
    raw_xlsx_path = DATA_DIR / "raw" / "survey_raw_sample.xlsx"

    assert raw_csv_path.exists()
    assert raw_xlsx_path.exists()

    df_raw = pd.read_csv(raw_csv_path)
    # 2400 students + 12 duplicates = 2412 rows
    assert len(df_raw) == 2412
    assert df_raw.shape[1] == 35

    # Check that long headers contain brackets for period indicators
    period_cols = [c for c in df_raw.columns if "[" in c and "Period" in c]
    assert len(period_cols) >= 28, (
        f"Expected >= 28 period indicator columns, found {len(period_cols)}"
    )

    # Check presence of administrative mess
    # 1. Duplicates
    id_col = [c for c in df_raw.columns if "ID" in c or "id" in c][0]
    duplicate_ids = df_raw[id_col].duplicated()
    assert duplicate_ids.sum() == 12

    # 2. Consent refusals
    consent_col = [c for c in df_raw.columns if "consent" in c.lower()][0]
    no_consent_count = (df_raw[consent_col] == "I do not consent").sum()
    assert no_consent_count > 0, "No-consent submissions missing"

    # 3. Missingness
    assert df_raw.isna().sum().sum() > 0, "Missing values not injected"


def test_marginal_distributions_rule() -> None:
    """No indicator should have >45% in any single ordinal response category."""
    hidden_path = DATA_DIR / "raw" / "ground_truth_hidden.csv"
    assert hidden_path.exists()

    gen = SyntheticSurveyGenerator(seed=42)
    _, _, latent_e = gen.generate_cohort()
    codes, col_names = gen.generate_indicator_codes(latent_e)

    for idx, col in enumerate(col_names):
        col_codes = codes[:, idx]
        counts = pd.Series(col_codes).value_counts(normalize=True)
        max_pct = counts.max()
        assert max_pct < 0.45, (
            f"Indicator {col} has category with {max_pct:.2%} >= 45% ceiling"
        )


def test_no_target_leakage_in_evaluation_samples() -> None:
    """Rule G3: outcome columns must NEVER leak into sample upload files."""
    clean_path = DATA_DIR / "samples" / "sample_upload_clean.xlsx"
    messy_path = DATA_DIR / "samples" / "sample_upload_messy.csv"

    assert clean_path.exists()
    assert messy_path.exists()

    df_clean = pd.read_excel(clean_path)
    df_messy = pd.read_csv(messy_path)

    forbidden_outcomes = [
        "intent_leave",
        "intent_defer",
        "intent_not_continue",
        "attrition_intent_score",
        "at_risk",
        "latent_risk",
    ]

    for col in df_clean.columns:
        for forbidden in forbidden_outcomes:
            assert forbidden.lower() not in col.lower(), (
                f"Target leakage in clean sample: '{col}' matches '{forbidden}'"
            )

    for col in df_messy.columns:
        for forbidden in forbidden_outcomes:
            assert forbidden.lower() not in col.lower(), (
                f"Target leakage in messy sample: '{col}' matches '{forbidden}'"
            )


def test_ground_truth_hidden_from_feature_schema() -> None:
    """Ground truth hidden file must never be referenced as a feature source."""
    schema = load_schema()
    feature_names = set(schema.get_feature_names())

    hidden_columns = {
        "archetype",
        "e_1",
        "e_2",
        "e_3",
        "e_4",
        "latent_risk",
        "attrition_intent_score",
        "at_risk",
    }

    overlap = feature_names.intersection(hidden_columns)
    assert len(overlap) == 0, f"Ground truth leaked into feature schema: {overlap}"


def test_messy_sample_upload_characteristics() -> None:
    """Messy upload must contain blanks, out-of-range values, and extra col."""
    messy_path = DATA_DIR / "samples" / "sample_upload_messy.csv"
    df_messy = pd.read_csv(messy_path)

    assert len(df_messy) == 200
    assert "campus_branch" in df_messy.columns, (
        "Extra unknown column 'campus_branch' missing"
    )
    assert df_messy.isna().sum().sum() > 0, "Missing blanks missing from messy upload"

    # Check out-of-range values injected in first few rows
    vals = df_messy.values.flatten()
    has_99 = any(v == 99 or v == "99" for v in vals)
    has_unknown = any(str(v).lower() == "unknown" for v in vals)
    assert has_99 or has_unknown, "Out-of-range test values missing from messy upload"
