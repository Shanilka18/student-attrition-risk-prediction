"""Tests for fold-local profile fitting and emitted profile features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.profiles.transformer import ProfileFeatureTransformer


def _index_frame(size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centers = np.asarray(
        [
            [0.2, 0.25, 0.3, 0.2],
            [0.8, 0.75, 0.7, 0.8],
        ]
    )
    labels = np.arange(size) % 2
    indices = np.clip(
        centers[labels] + rng.normal(0, 0.03, size=(size, 4)),
        0.0,
        1.0,
    )
    frame = pd.DataFrame(
        indices,
        columns=[
            "financial_stress_index",
            "commute_burden_index",
            "digital_access_gap_index",
            "study_environment_pressure_index",
        ],
    )
    frame["overall_pressure_index"] = frame.mean(axis=1)
    return frame


def test_transformer_fits_only_training_rows_and_emits_profile_features(tmp_path):
    config_path = tmp_path / "profile_selection.yaml"
    config_path.write_text(
        """
profile_selection:
  k_values: [2]
  algorithms: [kmeans]
  silhouette_minimum: 0.15
  bootstrap_resamples: 4
  bootstrap_fraction: 0.8
  random_seed: 42
""".strip(),
        encoding="utf-8",
    )
    training = _index_frame(80, 5)
    validation = _index_frame(20, 7) + 10.0
    transformer = ProfileFeatureTransformer(selection_config_path=str(config_path)).fit(
        training
    )
    learned_mean = transformer.scaler_.mean_.copy()
    learned_centers = transformer.clusterer_.centers.copy()
    learned_candidate_count = len(transformer.selection_summary_["candidate_metrics"])

    transformed = transformer.transform(validation)

    assert len(transformer.training_labels_) == len(training)
    assert transformer.profile_names_ == (
        "Few pressures",
        "Several pressures at once",
    )
    assert learned_candidate_count == 1
    assert transformed.filter(regex=r"^profile_id_").shape[1] == 2
    assert "profile_distance_to_center" in transformed
    assert "profile_high_pressure_index_count" in transformed
    np.testing.assert_array_equal(transformer.scaler_.mean_, learned_mean)
    np.testing.assert_array_equal(transformer.clusterer_.centers, learned_centers)


def test_gmm_transformer_emits_soft_membership_probabilities(tmp_path):
    config_path = tmp_path / "profile_selection.yaml"
    config_path.write_text(
        """
profile_selection:
  k_values: [2]
  algorithms: [gmm_full]
  silhouette_minimum: 0.15
  bootstrap_resamples: 3
  bootstrap_fraction: 0.8
  random_seed: 42
""".strip(),
        encoding="utf-8",
    )
    training = _index_frame(80, 12)
    transformer = ProfileFeatureTransformer(selection_config_path=str(config_path)).fit(
        training
    )

    transformed = transformer.transform(training.iloc[:10])
    probability_columns = [
        column
        for column in transformed.columns
        if column.startswith("profile_probability_")
    ]

    assert transformer.clusterer_.algorithm == "gmm_full"
    assert len(probability_columns) == 2
    np.testing.assert_allclose(
        transformed[probability_columns].sum(axis=1),
        np.ones(10),
    )
