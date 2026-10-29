"""Tests for the configuration-driven Environmental & Financial sample core."""

import pandas as pd
import pytest

from src.data_gen.sample_survey import (
    EnvironmentalFinancialSampleGenerator,
    calibrate_signal_strength,
    load_sample_configuration,
)


@pytest.fixture(scope="module")
def sample_generator():
    settings, schema, mappings = load_sample_configuration()
    return EnvironmentalFinancialSampleGenerator(settings, schema, mappings)


def test_core_generates_canonical_feature_columns_and_isolated_truth(sample_generator):
    settings = sample_generator.settings
    raw, hidden = sample_generator.generate(settings["sizes"]["full"])
    features = sample_generator.canonical_feature_frame(raw)

    assert len(raw) == settings["sizes"]["full"]
    assert list(features.columns) == list(sample_generator.schema["items"])
    assert list(hidden.columns) == ["type", "F", "A", "D", "H", "latent_risk"]
    assert not set(hidden.columns).intersection(features.columns)
    assert all(
        sample_generator.mappings[column]["google_forms_header"] in raw.columns
        for column in features.columns
    )


def test_core_generation_is_reproducible(sample_generator):
    left_raw, left_truth = sample_generator.generate(300, seed=42)
    right_raw, right_truth = sample_generator.generate(300, seed=42)

    pd.testing.assert_frame_equal(left_raw, right_raw)
    pd.testing.assert_frame_equal(left_truth, right_truth)


def test_online_commute_responses_are_consistent(sample_generator):
    raw, _ = sample_generator.generate(1200)
    features = sample_generator.canonical_feature_frame(raw)
    online = features["commute_time"].str.startswith("Not applicable")

    assert online.mean() == pytest.approx(
        sample_generator.settings["response_generation"]["online_rate"], abs=0.035
    )
    assert features.loc[online, "commute_spend"].str.startswith("Not applicable").all()
    assert features.loc[online, "transport_mode"].str.startswith("Not applicable").all()
    assert (
        features.loc[
            online, ["commute_time", "commute_spend", "transport_mode"]
        ].nunique()
        == 1
    ).all()


def test_nonbinary_item_levels_are_not_dominated(sample_generator):
    raw, _ = sample_generator.generate(1200)
    features = sample_generator.canonical_feature_frame(raw)

    for item, metadata in sample_generator.schema["items"].items():
        if metadata["type"] == "binary":
            continue
        largest_level = features[item].value_counts(normalize=True).max()
        assert largest_level <= 0.50, f"{item} level share is {largest_level:.3f}"


def test_response_quality_injects_configured_export_mess(sample_generator):
    settings = sample_generator.settings
    raw, _ = sample_generator.generate(settings["sizes"]["full"])
    messy, achieved = sample_generator.add_response_quality_effects(raw, seed=42)
    mappings = sample_generator.mappings

    assert len(messy) == len(raw) + round(
        len(raw) * settings["response_quality"]["duplicate_rate"]
    )
    assert messy[mappings["consent"]["google_forms_header"]].eq(
        "No / I decline"
    ).sum() == round(len(raw) * settings["response_quality"]["no_consent_rate"])
    assert messy[mappings["attention_check"]["google_forms_header"]].eq(
        settings["response_quality"]["attention_fail_response"]
    ).sum() == round(len(raw) * settings["response_quality"]["attention_fail_rate"])
    assert messy[mappings["cost_difficulty"]["google_forms_header"]].eq(
        sample_generator.schema["items"]["cost_difficulty"]["prefer_not_to_say"][
            "label"
        ]
    ).sum() >= round(len(raw) * settings["response_quality"]["prefer_not_to_say_rate"])
    assert messy.isna().sum().sum() > 0
    assert messy[mappings["student_id"]["google_forms_header"]].duplicated().any()
    assert achieved["duplicates"] == round(
        len(raw) * settings["response_quality"]["duplicate_rate"]
    )
    assert achieved["typo_cells"] == round(
        len(raw)
        * len(sample_generator.schema["items"])
        * settings["response_quality"]["typo_rate"]
    )


def test_response_quality_is_reproducible(sample_generator):
    raw, _ = sample_generator.generate(300, seed=42)
    left, left_metrics = sample_generator.add_response_quality_effects(raw, seed=42)
    right, right_metrics = sample_generator.add_response_quality_effects(raw, seed=42)

    pd.testing.assert_frame_equal(left, right)
    assert left_metrics == right_metrics


def test_five_fold_calibration_meets_prevalence_and_auc_gates(sample_generator):
    settings = sample_generator.settings
    result = calibrate_signal_strength(sample_generator)
    metrics = result["metrics"]
    auc_lower, auc_upper = settings["outcomes"]["lightgbm_auc_range"]
    prevalence_lower, prevalence_upper = settings["outcomes"]["prevalence_range"]

    assert (
        len(result["history"])
        <= settings["outcomes"]["signal_strength"]["maximum_iterations"]
    )
    assert auc_lower <= metrics["lightgbm_auc"] <= auc_upper
    assert prevalence_lower <= metrics["at_risk_prevalence"] <= prevalence_upper
    assert (
        metrics["logistic_auc"]
        <= metrics["lightgbm_auc"] + settings["outcomes"]["max_logistic_auc_delta"]
    )
