"""Tests for end-to-end fold-fitted feature-set pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.preprocessing.codec import EnvFinCodec
from src.preprocessing.feature_transformers import build_feature_pipeline
from src.profiles.transformer import ProfileFeatureTransformer


@pytest.fixture
def codec():
    return EnvFinCodec()


def _survey_frame(codec, size=240, seed=4):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            item: rng.choice(
                [level["code"] for level in metadata["levels"]],
                size=size,
            )
            for item, metadata in codec.items.items()
        }
    )
    online = rng.random(size) < 0.1
    for item in ("commute_time", "commute_spend", "transport_mode"):
        frame.loc[online, item] = codec.items[item]["structural_na"]["code"]
    frame["is_online"] = online
    frame["pns_cost_difficulty"] = rng.random(size) < 0.05
    frame["flag_commute_partial_na"] = False
    frame["flag_constant"] = True
    frame.loc[rng.random(size) < 0.08, "cost_difficulty"] = np.nan
    frame["student_id"] = [f"S{index:05d}" for index in range(size)]
    frame["at_risk"] = rng.integers(0, 2, size=size)
    frame["attrition_intent_score"] = rng.integers(0, 101, size=size)
    return frame


def test_feature_sets_have_expected_boundaries_and_shapes(codec, monkeypatch):
    monkeypatch.setattr(
        ProfileFeatureTransformer,
        "_selection_settings",
        lambda self: {
            "k_values": [2],
            "algorithms": ["kmeans"],
            "silhouette_minimum": -1.0,
            "bootstrap_resamples": 3,
        },
    )
    frame = _survey_frame(codec)
    expected_features = {
        "raw": lambda columns: any(name.endswith("__pressure") for name in columns)
        and "financial_stress_index" not in columns,
        "agg": lambda columns: "financial_stress_index" in columns
        and "travel_spend_to_income_ratio" in columns,
        "agg_profiles": lambda columns: sum(
            name.startswith("profile_id_") for name in columns
        )
        >= 2,
        "profiles_only": lambda columns: all(
            name.startswith(
                (
                    "profile_id_",
                    "profile_probability_",
                    "profile_distance",
                    "profile_high_pressure_index_count",
                )
            )
            for name in columns
        ),
        "single_signal": lambda columns: list(columns) == ["income_band__pressure"],
    }

    for feature_set, check in expected_features.items():
        pipeline = build_feature_pipeline(feature_set, "lightgbm", codec=codec)
        transformed = pipeline.fit_transform(frame)
        assert check(transformed.columns)
        assert transformed.shape[0] == len(frame)
        assert not {"student_id", "at_risk", "attrition_intent_score"}.intersection(
            transformed.columns
        )
        assert not transformed.isna().any().any()


def test_fitted_statistics_and_scaler_are_unchanged_by_validation_data(codec):
    frame = _survey_frame(codec)
    training = frame.iloc[:180].copy()
    validation = frame.iloc[180:].copy()
    validation["income_band"] = max(
        level["code"] for level in codec.items["income_band"]["levels"]
    )
    validation["cost_difficulty"] = np.nan

    pipeline = build_feature_pipeline("agg", "logistic_regression", codec=codec).fit(
        training
    )
    learned_imputation = pipeline.named_steps["imputer"].statistics_.copy()
    learned_scaler_mean = pipeline.named_steps["scale"].scaler_.mean_.copy()
    validation_features = pipeline.transform(validation)

    assert validation_features.shape[0] == len(validation)
    assert pipeline.named_steps["imputer"].statistics_ == learned_imputation
    np.testing.assert_array_equal(
        pipeline.named_steps["scale"].scaler_.mean_,
        learned_scaler_mean,
    )
    assert "flag_constant" in pipeline.named_steps["near_constant"].removed_features_


def test_registry_scales_only_configured_model_families(codec):
    frame = _survey_frame(codec)
    tree = build_feature_pipeline("raw", "lightgbm", codec=codec)
    linear = build_feature_pipeline("raw", "logistic_regression", codec=codec)

    assert "scale" not in tree.named_steps
    assert "scale" in linear.named_steps
    assert clone(linear).fit(frame).transform(frame).shape[0] == len(frame)


def test_shuffled_label_sanity_is_near_chance(codec):
    frame = _survey_frame(codec, size=1500, seed=31)
    labels = np.random.default_rng(202).permutation(frame["at_risk"].to_numpy())
    model = Pipeline(
        [
            (
                "features",
                build_feature_pipeline("raw", "logistic_regression", codec=codec),
            ),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=500)),
        ]
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=18)
    probabilities = cross_val_predict(
        model,
        frame,
        labels,
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    assert 0.45 <= roc_auc_score(labels, probabilities) <= 0.55
