"""Tests for Environmental & Financial Schema, Codec, and Header Mapping (EF-01).

Validates:
- Schema YAML structure, canonical item count (17), and pressure directions.
- Strict isolation of outcome columns (never role=feature).
- Bidirectional label-to-code round-trip integrity across all items.
- Formatting tolerance (case, spacing, en-dashes, LKR prefixes, thousand commas).
- Mapping coverage across all canonical columns.
- Derived indices reference valid schema items only.
- Template files exist and conform to schema.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from src.preprocessing.codec import EnvFinCodec


@pytest.fixture(scope="module")
def codec():
    return EnvFinCodec(
        schema_path="config/schema_envfin.yaml",
        mapping_path="config/mapping_envfin.yaml",
    )


def test_schema_loads(codec):
    """Verify schema loads and contains all required top-level sections."""
    assert "component" in codec.schema
    assert "items" in codec.schema
    assert "outcomes" in codec.schema
    assert "shared_meta" in codec.schema
    assert "derived_indices" in codec.schema
    assert len(codec.items) == 17, f"Expected 17 items, got {len(codec.items)}"


def test_every_item_has_pressure_direction(codec):
    """Confirm every canonical item specifies a valid pressure direction and role=feature."""
    valid_directions = {"higher", "reversed", "nominal_mapped"}
    for name, meta in codec.items.items():
        assert "pressure_direction" in meta, f"Item '{name}' missing 'pressure_direction'"
        assert meta["pressure_direction"] in valid_directions, (
            f"Item '{name}' has invalid direction: {meta['pressure_direction']}"
        )
        assert meta.get("role") == "feature", f"Item '{name}' must have role='feature'"
        assert "n_levels" in meta, f"Item '{name}' missing 'n_levels'"
        assert len(meta.get("levels", [])) == meta["n_levels"], (
            f"Item '{name}' declared {meta['n_levels']} levels but provided {len(meta.get('levels', []))}"
        )


def test_outcome_columns_never_features(codec):
    """Ensure outcome and metadata columns are strictly quarantined from role=feature."""
    for out_name, out_meta in codec.outcomes.items():
        assert out_meta.get("role") == "outcome", f"Outcome '{out_name}' must have role='outcome'"
        assert out_meta.get("role") != "feature", f"Outcome '{out_name}' must NOT have role='feature'"

    for meta_name, meta_def in codec.shared_meta.items():
        assert meta_def.get("role") in ["metadata", "quality_control"], (
            f"Meta '{meta_name}' has unexpected role: {meta_def.get('role')}"
        )
        assert meta_def.get("role") != "feature"


def test_codec_round_trips_every_label(codec):
    """Verify that every standard option label round-trips: decode(encode(label)) == label."""
    for name, meta in codec.items.items():
        for lvl in meta.get("levels", []):
            label = lvl["label"]
            code = lvl["code"]

            # Label -> Code
            encoded_code = codec.encode(name, label)
            assert encoded_code == code, (
                f"Mismatch encoding '{name}' -> '{label}': expected {code}, got {encoded_code}"
            )

            # Code -> Label
            decoded_label = codec.decode(name, encoded_code)
            assert decoded_label == label, (
                f"Mismatch decoding '{name}' -> code {encoded_code}: expected '{label}', got '{decoded_label}'"
            )


def test_codec_tolerance_formatting(codec):
    """Test tolerance for case, extra spaces, en-dashes, LKR prefixes, and commas."""
    # Income band with LKR and hyphen
    code1 = codec.encode("income_band", "lkr 25000 - 50000")
    assert code1 == 1

    # Commute time with extra spaces
    code2 = codec.encode("commute_time", "  30  -  59 minutes  ")
    assert code2 == 1

    # Money worry with uppercase
    code3 = codec.encode("money_worry", "STRONGLY AGREE")
    assert code3 == 4

    # Missed due to barriers with mixed case
    code4 = codec.encode("missed_due_to_barriers", "several times")
    assert code4 == 2


def test_structural_na_and_prefer_not_to_say(codec):
    """Verify structural N/A and Prefer Not to Say special handling."""
    # Commute time structural N/A
    sna_code = codec.encode("commute_time", "Not applicable (I study fully online)")
    assert sna_code == -2
    assert codec.get_pressure("commute_time", sna_code) == 0.0

    # Shorthand N/A
    assert codec.encode("commute_time", "n/a") == -2

    # Cost difficulty Prefer Not to Say
    pns_code = codec.encode("cost_difficulty", "Prefer not to say")
    assert pns_code == -1
    assert np.isnan(codec.get_pressure("cost_difficulty", pns_code))


def test_mapping_covers_every_canonical_column(codec):
    """Confirm mapping configuration covers every single canonical item."""
    mappings = codec.mapping.get("canonical_mappings", {})
    for item_name in codec.items.keys():
        assert item_name in mappings, f"Mapping configuration missing canonical item '{item_name}'"
        assert "google_forms_header" in mappings[item_name], (
            f"Item '{item_name}' missing 'google_forms_header'"
        )
        assert len(mappings[item_name].get("accepted_variants", [])) > 0, (
            f"Item '{item_name}' must have accepted variants"
        )


def test_indices_reference_only_existing_items(codec):
    """Verify that all 4 derived indices reference valid schema items only."""
    assert len(codec.derived_indices) == 4, f"Expected 4 indices, got {len(codec.derived_indices)}"
    expected_indices = {
        "financial_stress",
        "commute_burden",
        "digital_access_gap",
        "study_environment_pressure",
    }
    assert set(codec.derived_indices.keys()) == expected_indices

    for idx_name, idx_def in codec.derived_indices.items():
        referenced_items = idx_def.get("items", [])
        assert len(referenced_items) > 0, f"Index '{idx_name}' has no referenced items"
        for ref in referenced_items:
            assert ref in codec.items, f"Index '{idx_name}' references non-existent item '{ref}'"


def test_compute_derived_indices(codec):
    """Test automated calculation of the 4 derived indices from dataframe rows."""
    df_sample = pd.DataFrame([
        {
            "income_band": "Less than LKR 25,000",        # reversed: 1.0
            "aid_status": "No",                            # 1.0
            "family_support_duty": "Yes",                  # 1.0
            "cost_difficulty": "Always",                   # 1.0
            "financial_shock": "Yes, a major one",         # 1.0
            "money_worry": "Strongly agree",               # 1.0
            "job_hours": "More than 30",                   # 1.0
            "commute_time": "More than 120 minutes",       # 1.0
            "commute_spend": "More than LKR 10,000",       # 1.0
            "transport_mode": "hired_vehicle",             # 0.8
            "home_internet": "No home internet access",    # 1.0
            "main_device": "Smartphone only",              # 0.9
            "mobile_data_plan": "No",                      # 1.0
            "quiet_study_place": "No",                     # 1.0
            "living_situation": "Rented accommodation",    # 0.7
        },
        {
            "income_band": "More than LKR 200,000",        # reversed: 0.0
            "aid_status": "Yes, full coverage",            # 0.0
            "family_support_duty": "No",                   # 0.0
            "cost_difficulty": "Never",                    # 0.0
            "financial_shock": "No",                       # 0.0
            "money_worry": "Strongly disagree",            # 0.0
            "job_hours": "None",                           # 0.0
            "commute_time": "Not applicable (I study fully online)", # 0.0
            "commute_spend": "Not applicable",             # 0.0
            "transport_mode": "not_applicable",            # 0.0
            "home_internet": "Reliable, high-speed",       # 0.0
            "main_device": "Personal laptop / computer",   # 0.1
            "mobile_data_plan": "Yes, sufficient for coursework", # 0.0
            "quiet_study_place": "Yes, always",            # 0.0
            "living_situation": "Living with family",      # 0.2
        }
    ])

    df_out = codec.compute_indices(df_sample)
    assert "financial_stress_index" in df_out.columns
    assert "commute_burden_index" in df_out.columns
    assert "digital_access_gap_index" in df_out.columns
    assert "study_environment_pressure_index" in df_out.columns

    # High pressure student: values should be close to 1.0
    assert df_out["financial_stress_index"].iloc[0] == 1.0
    assert df_out["commute_burden_index"].iloc[0] > 0.8
    assert df_out["digital_access_gap_index"].iloc[0] > 0.8

    # Low pressure / online student: values should be low / 0.0
    assert df_out["financial_stress_index"].iloc[1] == 0.0
    assert df_out["commute_burden_index"].iloc[1] == 0.0
    assert df_out["digital_access_gap_index"].iloc[1] < 0.1


def test_template_reopens_correctly():
    """Verify that template CSV and Excel files reopen and contain all 17 items."""
    csv_path = Path("data/templates/envfin_template.csv")
    xlsx_path = Path("data/templates/envfin_template.xlsx")

    assert csv_path.exists(), "envfin_template.csv must exist"
    assert xlsx_path.exists(), "envfin_template.xlsx must exist"

    df_csv = pd.read_csv(csv_path)
    df_excel_main = pd.read_excel(xlsx_path, sheet_name="Template")
    df_excel_dict = pd.read_excel(xlsx_path, sheet_name="Data Dictionary")

    assert len(df_csv) >= 2
    assert len(df_excel_main) >= 2
    assert len(df_excel_dict) == 24  # 17 items + 4 meta/QC + 3 outcomes

    # Verify all 17 items are columns
    codec = EnvFinCodec()
    for item in codec.items.keys():
        assert item in df_csv.columns, f"CSV template missing item '{item}'"
        assert item in df_excel_main.columns, f"Excel template missing item '{item}'"
