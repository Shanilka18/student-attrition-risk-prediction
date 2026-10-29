"""Tests for profile ablation statistics and profile figure outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.profiles.evaluation import paired_wilcoxon_tests
from src.profiles.figures import profile_statistics, save_profile_figures
from src.profiles.selection import DEFAULT_INDEX_COLUMNS


def test_paired_wilcoxon_uses_matching_fold_metrics():
    rows = []
    for model in ("logistic_regression", "lightgbm"):
        for feature_set, offset in (
            ("agg", 0.0),
            ("agg_profiles", 0.1),
            ("profiles_only", -0.1),
        ):
            for fold in range(1, 6):
                rows.append(
                    {
                        "fold": fold,
                        "model": model,
                        "feature_set": feature_set,
                        "f1": 0.5 + offset + fold / 100,
                        "roc_auc": 0.6 + offset + fold / 100,
                        "mcc": 0.2 + offset + fold / 100,
                    }
                )

    tests = paired_wilcoxon_tests(pd.DataFrame(rows), expected_folds=5)

    assert len(tests) == 12
    assert all(result["status"] == "available" for result in tests)
    assert all(result["paired_folds"] == 5 for result in tests)
    assert all(0 <= result["p_value"] <= 1 for result in tests)


def test_profile_statistics_and_all_requested_figures(tmp_path):
    rng = np.random.default_rng(42)
    profile_ids = np.tile([0, 1], 40)
    values = np.column_stack(
        [
            np.clip(
                np.where(profile_ids == 0, 0.25, 0.75) + rng.normal(0, 0.03, 80),
                0,
                1,
            )
            for _ in DEFAULT_INDEX_COLUMNS
        ]
    )
    data = pd.DataFrame(values, columns=DEFAULT_INDEX_COLUMNS)
    data["at_risk"] = np.where(profile_ids == 0, 0, 1)
    index_means = data.groupby(profile_ids)[list(DEFAULT_INDEX_COLUMNS)].mean()
    statistics = profile_statistics(
        data,
        profile_ids,
        ("Few pressures", "Mainly financial strain"),
    )
    candidates = [
        {
            "algorithm": algorithm,
            "k": k,
            "stability_ari_mean": 0.7 + k / 100,
            "silhouette": 0.2 + k / 100,
        }
        for algorithm in ("kmeans", "gmm_full")
        for k in (2, 3, 4)
    ]
    standardized = np.column_stack([values, values.mean(axis=1)])

    figures = save_profile_figures(
        standardized_training=standardized,
        training_labels=profile_ids,
        index_means=index_means,
        profile_statistics=statistics,
        candidate_metrics=candidates,
        profile_names=("Few pressures", "Mainly financial strain"),
        output_dir=tmp_path,
        palette={},
    )

    assert statistics["students"].tolist() == [40, 40]
    assert statistics.loc[1, "at_risk_rate"] == 1
    assert len(figures) == 6
    assert all(path.is_file() for path in figures)
