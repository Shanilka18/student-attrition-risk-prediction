"""Tests for persisted synthetic survey artifacts."""

import hashlib

import pandas as pd

from src.data_gen.sample_survey import generate_sample_data, load_sample_configuration


def _artifact_hashes(result, root):
    paths = [*result["outputs"].values(), *result["figures"]]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def test_sample_outputs_are_complete_valid_and_reproducible(tmp_path):
    settings, schema, mappings = load_sample_configuration()
    first = generate_sample_data(tmp_path)
    first_hashes = _artifact_hashes(first, tmp_path)

    raw = pd.read_csv(first["outputs"]["raw_full"])
    small = pd.read_csv(first["outputs"]["raw_small"])
    hidden = pd.read_csv(first["outputs"]["hidden_truth"])
    clean_upload = pd.read_excel(first["outputs"]["clean_upload"])
    messy_upload = pd.read_csv(first["outputs"]["messy_upload"])
    at_risk_header = mappings["at_risk"]["google_forms_header"]
    score_header = mappings["attrition_intent_score"]["google_forms_header"]

    assert len(raw) >= settings["sizes"]["full"]
    assert len(small) >= settings["sizes"]["small"]
    assert len(hidden) == settings["sizes"]["full"]
    assert list(hidden.columns) == ["type", "F", "A", "D", "H", "latent_risk"]
    assert len(clean_upload) == settings["sizes"]["upload"]
    assert list(clean_upload.columns) == list(schema["items"])
    assert len(messy_upload) == settings["sizes"]["upload"]
    assert settings["messy_upload"]["unknown_column_name"] in messy_upload.columns
    assert not any(
        name in clean_upload.columns
        for name in ("attrition_intent_score", "at_risk", at_risk_header)
    )
    expected_at_risk = raw[score_header].ge(settings["outcomes"]["at_risk_threshold"])
    actual_at_risk = raw[at_risk_header].eq("At-Risk of Withdrawal")
    assert expected_at_risk.equals(actual_at_risk)
    assert (
        actual_at_risk.iloc[: settings["sizes"]["full"]].mean()
        == first["calibration"]["metrics"]["at_risk_prevalence"]
    )
    assert 0.24 <= actual_at_risk.mean() <= 0.32
    assert len(first["figures"]) == 6
    assert all(path.stat().st_size > 0 for path in first["figures"])

    second = generate_sample_data(tmp_path)
    assert first_hashes == _artifact_hashes(second, tmp_path)
