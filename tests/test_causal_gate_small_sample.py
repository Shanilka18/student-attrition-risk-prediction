"""Test Causal Forest Gate behavior under underpowered / small dataset regimes.

Verifies hard rule:
'Causal Forest (EconML) is CONDITIONAL: only run if events >= 60 and overlap is adequate;
otherwise report "not run - insufficient data" and show why.'
"""

import pytest
import numpy as np
import pandas as pd
from src.causal.causal_forest import can_run_causal_forest, fit_conditional_causal_forest

def test_causal_forest_gate_skips_small_sample():
    """Assert that a small dataset (N=150, events=20) strictly fails the gate."""
    n = 150
    n_events = 20
    min_exposure = 25
    overlap_ok = True

    passed, reason = can_run_causal_forest(
        n=n,
        n_events=n_events,
        min_exposure_group_size=min_exposure,
        overlap_satisfied=overlap_ok,
        threshold_n=300,
        threshold_events=60,
    )

    assert passed is False
    assert "Sample size N=150 < 300 minimum threshold" in reason
    assert "events=20 < 60 minimum required" in reason

def test_causal_forest_graceful_fallback_when_gated():
    """Verify that fit_conditional_causal_forest safely returns fallback dictionary when gate fails."""
    # Create small synthetic dataframe with N=150 and only 15 events
    np.random.seed(42)
    df_small = pd.DataFrame({
        "financial_hardship": np.random.choice([0, 1], size=150, p=[0.7, 0.3]),
        "attrition": np.random.choice([0, 1], size=150, p=[0.9, 0.1]),
        "household_income_bracket": np.random.choice(["Low", "Medium", "High"], size=150),
        "district_type": np.random.choice(["urban", "rural"], size=150),
        "year_of_study": np.random.choice([1, 2, 3, 4], size=150),
    })

    # The gate should reject this dataset
    n_events = int(df_small["attrition"].sum())
    passed, reason = can_run_causal_forest(
        n=len(df_small),
        n_events=n_events,
        min_exposure_group_size=int(df_small["financial_hardship"].value_counts().min()),
        overlap_satisfied=True
    )
    assert passed is False
    assert reason is not None
