"""Focused tests for the reusable Environmental & Financial cleaner."""

import json
import sys

import pandas as pd
import pytest

from src.cleaning.cleaner import EnvironmentalFinancialCleaner


@pytest.fixture
def cleaner():
    return EnvironmentalFinancialCleaner()


def canonical_row(cleaner, student_id="P-00001"):
    row = {}
    mappings = cleaner.codec.mapping["canonical_mappings"]
    for item, metadata in cleaner.codec.items.items():
        row[mappings[item]["google_forms_header"]] = metadata["levels"][0]["label"]
    row[mappings["student_id"]["google_forms_header"]] = student_id
    row[mappings["timestamp"]["google_forms_header"]] = "2026-01-01 09:00:00"
    row[mappings["consent"]["google_forms_header"]] = "Yes / I agree"
    row[mappings["attention_check"]["google_forms_header"]] = "Agree"
    row[mappings["attrition_intent_score"]["google_forms_header"]] = 20
    row[mappings["at_risk"]["google_forms_header"]] = "Persisting / Low Risk"
    return row


def test_read_detects_header_row_and_normalizes_text(cleaner, tmp_path):
    content = (
        "Synthetic export\nTimestamp,income_band,,\n" "2026-01-01,  LKR 25 – 50  ,,\n\n"
    ).encode("utf-8-sig")
    path = tmp_path / "header.csv"
    path.write_bytes(content)

    frame, log = cleaner.read_input(path)

    assert log["rule"] == "read_normalize"
    assert "Timestamp" in frame.columns
    assert len(frame) == 1
    assert frame.iloc[0, 1] == "LKR 25 - 50"


def test_read_xlsx_header_and_data(cleaner, tmp_path):
    row = canonical_row(cleaner)
    path = tmp_path / "input.xlsx"
    pd.DataFrame([row]).to_excel(path, index=False)

    frame, log = cleaner.read_input(path)

    assert len(frame) == 1
    assert log["action"].find("binary workbook") >= 0


def test_header_mapping_normalizes_names_and_only_suggests_fuzzy(cleaner):
    row = canonical_row(cleaner)
    income_header = cleaner.codec.mapping["canonical_mappings"]["income_band"][
        "google_forms_header"
    ]
    row["household income"] = row.pop(income_header)
    row["houshold incom band"] = "Less than LKR 25,000"
    frame, mapping = cleaner.map_headers(pd.DataFrame([row]))

    assert "income_band" in frame.columns
    assert "houshold incom band" in frame.columns
    assert mapping["missing_required"] == []
    assert mapping["suggestions"][0]["accepted"] is False
    assert mapping["suggestions"][0]["confidence"] > 0
    assert mapping["log"]["rule"] == "map_headers"


def test_consent_and_attention_filters_are_configurable(cleaner):
    rows = [canonical_row(cleaner, f"P-{i:05d}") for i in range(1, 4)]
    names = cleaner.codec.mapping["canonical_mappings"]
    rows[1][names["consent"]["google_forms_header"]] = "No / I decline"
    rows[2][names["attention_check"]["google_forms_header"]] = "Disagree"
    frame, _ = cleaner.map_headers(pd.DataFrame(rows))
    frame, consent_log = cleaner.filter_consent(frame, "training")
    frame, attention_log = cleaner.filter_attention(frame, "training")

    assert len(frame) == 1
    assert consent_log["rows"] == 1
    assert attention_log["rows"] == 1
    assert cleaner.row_reasons[1] == ["consent_filter"]
    assert cleaner.row_reasons[2] == ["attention_filter"]


def test_value_fuzzy_threshold_and_item_range_are_enforced(cleaner):
    row = canonical_row(cleaner)
    columns = cleaner.codec.mapping["canonical_mappings"]
    row[columns["cost_difficulty"]["google_forms_header"]] = "Neverr"
    frame, _ = cleaner.map_headers(pd.DataFrame([row]))
    frame, metrics = cleaner.encode_values(frame)
    assert frame.loc[0, "cost_difficulty"] == 0
    assert metrics["invalid_values"]["cost_difficulty"] == 0

    frame.loc[0, "income_band"] = 99
    frame, range_metrics = cleaner.check_ranges(frame)
    assert pd.isna(frame.loc[0, "income_band"])
    assert range_metrics["invalid_range_values"]["income_band"] == 1
    assert "out_of_range:income_band" in cleaner.row_reasons[0]


def test_pns_online_and_job_hours_special_cases(cleaner):
    row = canonical_row(cleaner)
    mappings = cleaner.codec.mapping["canonical_mappings"]
    for item in ("commute_time", "commute_spend", "transport_mode"):
        special = cleaner.codec.items[item]["structural_na"]
        row[mappings[item]["google_forms_header"]] = special["label"]
    row[mappings["cost_difficulty"]["google_forms_header"]] = cleaner.codec.items[
        "cost_difficulty"
    ]["prefer_not_to_say"]["label"]
    row[mappings["job_hours"]["google_forms_header"]] = "21–30"
    frame, _ = cleaner.map_headers(pd.DataFrame([row]))
    frame, _ = cleaner.encode_values(frame)
    frame, special_metrics = cleaner.special_answers(frame)

    assert frame.loc[0, "is_online"]
    assert frame.loc[0, "commute_time"] == -2
    assert pd.isna(frame.loc[0, "cost_difficulty"])
    assert frame.loc[0, "pns_cost_difficulty"]
    assert frame.loc[0, "job_hours"] == 0
    assert frame.loc[0, "flag_unemployed_with_hours"]
    assert special_metrics["prefer_not_to_say"]["cost_difficulty"] == 1


def test_commute_consistency_flags_keep_rows_and_ignore_valid_income_cost_pair(cleaner):
    row = canonical_row(cleaner)
    mappings = cleaner.codec.mapping["canonical_mappings"]
    row[mappings["income_band"]["google_forms_header"]] = "Less than LKR 25,000"
    row[mappings["cost_difficulty"]["google_forms_header"]] = "Never"
    for item in ("commute_time", "commute_spend"):
        row[mappings[item]["google_forms_header"]] = cleaner.codec.items[item][
            "structural_na"
        ]["label"]
    frame, _ = cleaner.map_headers(pd.DataFrame([row]))
    frame, _ = cleaner.encode_values(frame)
    frame, _ = cleaner.special_answers(frame)
    frame, counts = cleaner.consistency_flags(frame)

    assert len(frame) == 1
    assert frame.loc[0, "flag_commute_partial_na"]
    assert frame.loc[0, "flag_commute_physical_mode"]
    assert counts["flag_commute_partial_na"] == 1
    assert "flag_commute_partial_na" in cleaner.row_reasons[0]


def test_duplicate_ids_keep_last_submission(cleaner):
    rows = [canonical_row(cleaner) for _ in range(2)]
    score_header = cleaner.codec.mapping["canonical_mappings"][
        "attrition_intent_score"
    ]["google_forms_header"]
    rows[0][score_header] = 15
    rows[1][score_header] = 85
    frame, _ = cleaner.map_headers(pd.DataFrame(rows))
    frame, _ = cleaner.deduplicate(frame, "training")

    assert len(frame) == 1
    assert frame.iloc[0]["attrition_intent_score"] == 85
    assert cleaner.row_reasons[0] == ["duplicate_student_id"]


def test_straight_liner_detection_can_drop_or_preserve_by_mode(cleaner):
    rows = [canonical_row(cleaner, "P-00001"), canonical_row(cleaner, "P-00002")]
    mappings = cleaner.codec.mapping["canonical_mappings"]
    for index, item in enumerate(cleaner.items):
        codes = [level["code"] for level in cleaner.codec.items[item]["levels"]]
        rows[1][mappings[item]["google_forms_header"]] = cleaner.codec.decode(
            item, codes[-1] if index % 2 else codes[0]
        )
    frame, _ = cleaner.map_headers(pd.DataFrame(rows))
    frame, _ = cleaner.encode_values(frame)
    training, metrics = cleaner.detect_careless(frame.copy(), "training")
    scoring, _ = cleaner.detect_careless(frame.copy(), "scoring")

    assert metrics["straight_liners"] == 1
    assert len(training) == 1
    assert len(scoring) == 2
    assert scoring.iloc[0]["flag_straight_liner"]


def test_missing_data_drops_over_threshold_only_for_training(cleaner):
    rows = [canonical_row(cleaner, "P-00001")]
    frame, _ = cleaner.map_headers(pd.DataFrame(rows))
    frame, _ = cleaner.encode_values(frame)
    frame.loc[0, cleaner.items[:5]] = pd.NA
    training, train_metrics = cleaner.missing_data(frame.copy(), "training")
    scoring, score_metrics = cleaner.missing_data(frame.copy(), "scoring")

    assert training.empty
    assert len(scoring) == 1
    assert train_metrics["dropped"] == 1
    assert score_metrics["dropped"] == 0
    assert "more_than_25_percent_features_missing" in cleaner.row_reasons[0]
    assert pd.isna(scoring.loc[0, "income_band"])


def test_outcomes_reports_and_writes_parquet(cleaner, tmp_path):
    frame, _ = cleaner.map_headers(pd.DataFrame([canonical_row(cleaner, "P-00001")]))
    frame, _ = cleaner.encode_values(frame)
    score = frame["attrition_intent_score"]
    frame["attrition_intent_score"] = score
    frame, metrics = cleaner.build_outcomes(frame, "training")

    assert frame.loc[0, "at_risk"] == 0
    assert metrics["outcome_source"] == "attrition_intent_score"
    result = cleaner.clean(pd.DataFrame([canonical_row(cleaner)]), mode="training")
    saved = cleaner.write_result(result, tmp_path)
    assert saved.parquet_path.is_file()
    assert saved.report_path.is_file()
    assert json.loads(saved.report_path.read_text(encoding="utf-8"))["no_imputation"]
    assert pd.read_parquet(saved.parquet_path).shape == saved.data.shape


def test_training_without_outcomes_is_red_and_scoring_keeps_rows(cleaner):
    row = canonical_row(cleaner)
    mapping = cleaner.codec.mapping["canonical_mappings"]
    row.pop(mapping["attrition_intent_score"]["google_forms_header"])
    row.pop(mapping["at_risk"]["google_forms_header"])

    training = cleaner.clean(pd.DataFrame([row]), mode="training")
    scoring = cleaner.clean(pd.DataFrame([row]), mode="scoring")

    assert training.report["traffic_light"]["status"] == "red"
    assert scoring.report["rows_out"] == 1
    assert scoring.data["at_risk"].isna().all()
    assert scoring.report["stage_log"][-1]["rule"] == "outcome_handling"


def test_scoring_mode_preserves_rows_and_marks_missing_data(cleaner):
    frame = pd.DataFrame([{"household income": "low income", "legacy_extra": 1}])
    result = cleaner.clean(frame, mode="scoring")

    assert len(result.data) == 1
    assert result.report["traffic_light"]["status"] == "red"
    assert result.report["mapping"]["suggestions"]
    assert result.report["mapping"]["missing_required"]
    assert result.data.loc[0, "row_reasons"]


def test_generated_export_removal_counts_match_quality_injection(cleaner):
    result = cleaner.clean("data/raw/survey_raw_sample.csv", mode="training")
    counts = result.report["per_rule_counts"]

    assert result.report["rows_in"] == 1206
    assert counts["consent_filter"] == 12
    assert counts["attention_filter"] == 24
    assert counts["duplicate_student_id"] == 6
    assert 20 <= counts["straight_liner"] <= 80
    assert result.report["traffic_light"]["status"] == "green"
    assert result.report["rows_in"] - result.report["rows_out"] == 98


def test_cli_clean_writes_outputs_and_prints_suggestions(tmp_path, monkeypatch, capsys):
    from mei.cli import main

    input_path = tmp_path / "messy.csv"
    pd.DataFrame(
        [
            {
                "houshold incom band": "Less than LKR 25,000",
                "legacy_extra": "test",
            }
        ]
    ).to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m mei.cli",
            "clean",
            "--input",
            str(input_path),
            "--mode",
            "scoring",
            "--output-root",
            str(tmp_path / "outputs"),
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "Header suggestion:" in output
    assert "not auto-accepted" in output
    assert "Cleaning status: red" in output
    output_dir = tmp_path / "outputs" / "data" / "interim"
    assert list(output_dir.glob("clean_*.parquet"))
    assert list(output_dir.glob("clean_*.json"))
