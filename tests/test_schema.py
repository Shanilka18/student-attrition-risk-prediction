"""Unit tests for schema loading, mapping, codec, and leakage prevention."""

from pathlib import Path

import pandas as pd

from mei.io.codec import SurveyCodec
from mei.io.schema import (
    load_labels,
    load_mapping,
    load_schema,
    validate_all_configs,
)


def test_schema_loads() -> None:
    """Survey schema must load and contain 35 cols (28 features, 3 outcomes)."""
    schema = load_schema()
    assert schema.num_columns == 35
    assert len(schema.columns) == 35
    assert schema.num_features == 28
    assert len(schema.get_feature_names()) == 28
    assert schema.num_outcomes == 3
    assert len(schema.get_outcome_names()) == 3


def test_mapping_coverage() -> None:
    """Mapping config must contain accepted headers for every canonical column."""
    schema = load_schema()
    mapping = load_mapping()

    canonical_cols = {col.name for col in schema.columns}
    mapped_cols = set(mapping.columns.keys())

    assert canonical_cols == mapped_cols, (
        f"Mapping mismatch. Missing in mapping: {canonical_cols - mapped_cols}, "
        f"Extra in mapping: {mapped_cols - canonical_cols}"
    )

    # Every mapping list must contain >= 2 variants (including canonical name)
    for col_name, variants in mapping.columns.items():
        assert len(variants) >= 2, f"Column {col_name} has insufficient variants"
        assert col_name in variants, f"Column {col_name} missing canonical name"


def test_no_target_leakage_roles() -> None:
    """Rule G3: outcome columns must NEVER be marked as role='feature'."""
    schema = load_schema()
    feature_names = set(schema.get_feature_names())
    outcome_names = set(schema.get_outcome_names())

    overlap = feature_names.intersection(outcome_names)
    assert len(overlap) == 0, f"Target leakage detected! Overlap: {overlap}"

    for col in schema.columns:
        if col.role == "outcome":
            assert col.name in ["intent_leave", "intent_defer", "intent_not_continue"]
            assert col.role != "feature"
            assert col.name not in feature_names


def test_codec_round_trip() -> None:
    """Codec must round-trip numeric code to canonical text and back to code."""
    schema = load_schema()
    codec = SurveyCodec(schema)

    for col in schema.columns:
        if not col.text_labels:
            continue
        for expected_code, expected_label in col.text_labels.items():
            int_code = int(expected_code)

            # Code -> Label
            decoded_label = codec.decode(col.name, int_code, strict=True)
            assert decoded_label == expected_label, (
                f"Decode mismatch for {col.name}[{int_code}]: "
                f"got '{decoded_label}', expected '{expected_label}'"
            )

            # Label -> Code
            encoded_code = codec.encode(col.name, decoded_label, strict=True)
            assert encoded_code == int_code, (
                f"Encode mismatch for {col.name}['{decoded_label}']: "
                f"got {encoded_code}, expected {int_code}"
            )


def test_codec_tolerance_variations() -> None:
    """Codec must tolerate case variation, whitespace, and unicode dashes."""
    codec = SurveyCodec()

    # Resource access: 1-2 days (with en-dash, hyphen, different casing)
    assert codec.encode("resource_access_p1", "1–2 days") == 2
    assert codec.encode("resource_access_p1", "1-2 days") == 2
    assert codec.encode("resource_access_p1", "  1-2 DAYS  ") == 2
    assert codec.encode("resource_access_p1", "1 to 2 days") == 2

    # Study time: <2 hours
    assert codec.encode("study_time_p2", "<2 hours") == 1
    assert codec.encode("study_time_p2", " <2 HOURS ") == 1

    # Frequency options: Never, Rarely, Sometimes, Often, Always
    assert codec.encode("quiz_practice_p3", "sometimes") == 3
    assert codec.encode("quiz_practice_p3", "SOMETIMES") == 3
    assert codec.encode("quiz_practice_p3", " Sometimes ") == 3

    # Likert agreement: Strongly disagree, Agree, etc.
    assert codec.encode("study_consistency_p4", "strongly agree") == 5
    assert codec.encode("intent_leave", "STRONGLY DISAGREE") == 1


def test_fuzzy_mapping_suggestion_behavior() -> None:
    """Fuzzy matching suggests on noisy headers but never auto-accepts."""
    mapping = load_mapping()
    assert mapping.fuzzy_matching.auto_accept is False

    # Test slightly altered header for resource access p1
    noisy = "How many days a week did u open online learning materials? [Period 1]"
    suggestions = mapping.suggest_mapping(noisy, threshold=70.0)

    assert len(suggestions) > 0
    top_col, top_score = suggestions[0]
    assert "resource_access_p1" in top_col
    assert top_score >= 70.0


def test_labels_config_integrity() -> None:
    """Labels configuration must specify valid score formula and threshold."""
    labels = load_labels()
    assert labels.targets["classification"]["threshold"] == 60.0
    assert "attrition_intent_score" in labels.targets["regression"]["name"]
    assert len(labels.outcome_items) == 3
    assert set(labels.outcome_items.keys()) == {
        "intent_leave",
        "intent_defer",
        "intent_not_continue",
    }
    assert "low" in labels.mei_index["bands"]
    assert "moderate" in labels.mei_index["bands"]
    assert "high" in labels.mei_index["bands"]


def test_validate_all_configs_pass() -> None:
    """validate_all_configs() must return status VALID."""
    summary = validate_all_configs()
    assert summary["status"] == "VALID"
    assert summary["num_columns"] == 35
    assert summary["num_features"] == 28
    assert summary["num_outcomes"] == 3


def test_templates_integrity() -> None:
    """Template files must match the 35 canonical columns."""
    schema = load_schema()
    canonical_cols = [c.name for c in schema.columns]

    csv_path = Path("data/templates/survey_template.csv")
    xlsx_path = Path("data/templates/survey_template.xlsx")

    assert csv_path.exists()
    assert xlsx_path.exists()

    df_csv = pd.read_csv(csv_path)
    assert list(df_csv.columns) == canonical_cols

    df_xlsx = pd.read_excel(xlsx_path, sheet_name="Survey Data")
    assert list(df_xlsx.columns) == canonical_cols

    df_dict = pd.read_excel(xlsx_path, sheet_name="Data Dictionary")
    assert len(df_dict) == 35
    assert "Column Name" in df_dict.columns
    assert "Allowed Codes & Text Options" in df_dict.columns
