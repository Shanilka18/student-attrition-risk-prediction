"""Unit and regression tests for realistic synthetic data generator."""

import pytest
import numpy as np
import pandas as pd

from src.data.generate_synthetic import (
    generate_synthetic_dataset,
    compute_cronbach_alpha,
    create_dirty_dataset,
    load_synthetic_config,
    FACTOR_ITEMS,
)
from src.data.schema import validate_dataframe


@pytest.fixture(scope="module")
def synth_config():
    """Load configuration for synthetic generator tests."""
    return load_synthetic_config()


def test_seed_reproducibility(synth_config):
    """Ensure two independent runs with identical seeds generate exact identical data."""
    df1, alphas1 = generate_synthetic_dataset(n_samples=100, config=synth_config, seed=100)
    df2, alphas2 = generate_synthetic_dataset(n_samples=100, config=synth_config, seed=100)

    pd.testing.assert_frame_equal(df1, df2)
    assert alphas1 == alphas2


def test_value_ranges_and_constraints(synth_config):
    """Verify ranges of all generated survey items, PRI, and outcome variables."""
    df, _ = generate_synthetic_dataset(
        n_samples=300,
        config=synth_config,
        seed=42,
        include_missing=False,
        include_straight_lining=False,
    )

    # 1. Check all 15 survey items bounded [1, 5]
    all_items = [it for item_group in FACTOR_ITEMS.values() for it in item_group]
    for item in all_items:
        assert df[item].min() >= 1
        assert df[item].max() <= 5
        assert np.issubdtype(df[item].dtype, np.integer)

    # 2. Check PRI_true range [0.0, 100.0]
    assert df["pri_true"].min() >= 0.0
    assert df["pri_true"].max() <= 100.0

    # 3. Check risk categories
    valid_categories = {"Low", "Moderate", "High"}
    assert set(df["risk_category"].unique()).issubset(valid_categories)

    # 4. Check attrition_label is binary {0, 1}
    assert set(df["attrition_label"].unique()).issubset({0, 1})

    # 5. Check dropout_intention is in {1, 2, 3, 4, 5}
    assert df["dropout_intention"].min() >= 1
    assert df["dropout_intention"].max() <= 5

    # 6. Check synthetic flag
    assert df["is_synthetic"].all() == True


def test_achieved_cronbach_alpha_tolerance(synth_config):
    """Verify achieved Cronbach's alpha per factor falls within target range [0.70, 0.88]."""
    df, alphas = generate_synthetic_dataset(
        n_samples=1500,
        config=synth_config,
        seed=42,
        include_missing=False,
        include_straight_lining=False,
    )

    target_min, target_max = synth_config["items"]["target_cronbach_range"]

    for factor_name, alpha_val in alphas.items():
        assert target_min <= alpha_val <= target_max, (
            f"Cronbach's alpha for {factor_name} ({alpha_val:.3f}) outside target [{target_min}, {target_max}]"
        )


def test_attrition_class_balance_tolerance(synth_config):
    """Verify attrition base rate is within expected probabilistic window (~18% - 30%)."""
    df, _ = generate_synthetic_dataset(
        n_samples=2000,
        config=synth_config,
        seed=42,
        include_missing=False,
    )
    base_rate = df["attrition_label"].mean()
    min_rate, max_rate = synth_config["outcomes"]["attrition"]["target_base_rate_range"]

    # Allow slight statistical variance (+/- 3% margin)
    assert (min_rate - 0.03) <= base_rate <= (max_rate + 0.05), (
        f"Attrition rate {base_rate:.3f} outside expected bounds [{min_rate}, {max_rate}]"
    )


def test_dirty_dataset_fails_validation_and_clean_passes(synth_config):
    """Verify that dirty dataset triggers validation errors, while clean dataset passes."""
    clean_df, _ = generate_synthetic_dataset(
        n_samples=100,
        config=synth_config,
        seed=42,
        include_missing=False,
        include_straight_lining=False,
    )
    clean_report = validate_dataframe(clean_df)
    assert clean_report.passed is True
    assert len(clean_report.errors) == 0

    dirty_df = create_dirty_dataset(clean_df, seed=42)
    dirty_report = validate_dataframe(dirty_df)
    assert dirty_report.passed is False
    assert len(dirty_report.errors) > 0
    # Must catch out-of-range, non-integer, and duplicate ID
    assert len(dirty_report.out_of_range_items) > 0
    assert len(dirty_report.non_integer_items) > 0
    assert len(dirty_report.duplicate_study_ids) > 0
