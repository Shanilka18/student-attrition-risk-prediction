"""Tests for feature leakage checks and diagnostic artifact generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.feature_diagnostics import (
    leakage_audit,
    run_feature_diagnostics,
)


def test_leakage_audit_flags_outcome_columns_and_high_correlation():
    target = pd.Series([0, 0, 1, 1], name="at_risk")
    features = pd.DataFrame(
        {
            "signal": [0.0, 0.1, 0.9, 1.0],
            "at_risk": target,
        }
    )

    audit = leakage_audit(features, target, source_columns=list(features.columns))

    assert not audit["passed"]
    assert audit["outcome_columns_present_in_features"] == ["at_risk"]
    assert audit["high_label_correlations"]["signal"] > 0.95


def test_diagnostics_write_effect_mi_and_spearman_outputs(tmp_path):
    rng = np.random.default_rng(42)
    target = pd.Series(np.tile([0, 1], 40), name="at_risk")
    features = pd.DataFrame(
        {
            "pressure": target.to_numpy() + rng.normal(0, 0.5, len(target)),
            "access": rng.normal(size=len(target)),
            "binary_flag": rng.integers(0, 2, size=len(target)),
        }
    )

    result = run_feature_diagnostics(
        features,
        target,
        source_columns=list(features.columns) + ["at_risk"],
        output_dir=tmp_path,
    )

    assert result["leakage_audit"]["passed"]
    assert len(result["effects"]) == len(features.columns)
    assert len(result["mutual_information"]) == len(features.columns)
    assert len(result["figures"]) == 3
    assert all(path.is_file() for path in result["figures"])
    assert (tmp_path / "feature_spearman_heatmap.png").is_file()
    assert np.isfinite(result["effects"]["Effect size"]).all()
