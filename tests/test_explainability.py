"""Unit and validation tests for the MEI explainability subsystem."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mei.explain.engine import (
    get_active_run_dir,
    get_champion_explainer,
)
from mei.explain.mapping import get_feature_group_mapping
from mei.explain.reasons import (
    convert_shap_to_mei_points,
    generate_student_reasons,
    load_reason_templates,
)
from mei.registry.score import score_dataframe


@pytest.fixture(scope="module")
def active_run_dir() -> Path:
    """Fixture providing active run directory."""
    return get_active_run_dir()


@pytest.fixture(scope="module")
def champion_features(active_run_dir: Path) -> list:
    """Fixture providing frozen champion feature names."""
    with open(
        active_run_dir / "champion" / "features.json", "r", encoding="utf-8"
    ) as f:
        return json.load(f)


def test_reason_templates_exist_for_every_model_feature(champion_features: list):
    """Test that every single model input feature has a reason template.

    Must define plain_name, risk_increasing, and risk_decreasing text.
    Fails if even one feature is missing.
    """
    templates = load_reason_templates()
    missing_features = []
    incomplete_templates = []

    for feat in champion_features:
        if feat not in templates:
            missing_features.append(feat)
        else:
            tmpl = templates[feat]
            if (
                not tmpl.get("risk_increasing")
                or not tmpl.get("risk_decreasing")
                or not tmpl.get("plain_name")
            ):
                incomplete_templates.append(feat)

    assert len(missing_features) == 0, (
        f"Missing reason templates for features: {missing_features}"
    )
    assert len(incomplete_templates) == 0, (
        f"Incomplete reason templates for features: {incomplete_templates}"
    )


def test_feature_group_mapping_coverage(champion_features: list):
    """Test that all 100 features map to a valid indicator, period, and family."""
    mapping = get_feature_group_mapping(champion_features)
    assert len(mapping) == len(champion_features)

    valid_families = {"raw_answers", "aggregates_drift", "state_features"}
    for feat, meta in mapping.items():
        assert meta["family"] in valid_families, (
            f"Invalid family for {feat}: {meta['family']}"
        )
        assert meta["indicator"] != "", f"Empty indicator for {feat}"
        assert meta["period"] != "", f"Empty period for {feat}"
        assert meta["plain_name"] != "", f"Empty plain name for {feat}"


def test_shap_values_sum_to_model_output(active_run_dir: Path, champion_features: list):
    """Test that local SHAP values satisfy the efficiency property: sum(shap) == f(x) - E[f(x)]."""
    explainer, pipeline, calibrator, feat_names, baseline_p = get_champion_explainer(
        active_run_dir, background_samples=15
    )
    bg_df = pd.read_parquet(
        active_run_dir / "champion" / "shap_background_200.parquet"
    )[feat_names]
    test_sample = bg_df.iloc[:3].values

    shap_vals = explainer.shap_values(
        test_sample, nsamples=100, l1_reg="num_features(10)"
    )
    preds = pipeline.predict_proba(test_sample)[:, 1]

    sums = np.sum(shap_vals, axis=1)
    expected_deltas = preds - explainer.expected_value

    np.testing.assert_allclose(
        sums, expected_deltas, atol=1e-4, err_msg="SHAP efficiency property violated!"
    )


def test_points_add_up_to_mei_delta():
    """Test that MEI points conversion sums to round(MEI - baseline) within exact tolerance."""
    np.random.seed(42)
    shap_vals = np.array([0.15, -0.08, 0.04, 0.12, -0.03, 0.05])
    mei_score = 65.0
    baseline_mei = 40.0

    points = convert_shap_to_mei_points(
        shap_vals, mei_score=mei_score, baseline_mei=baseline_mei
    )
    total_pts = int(np.sum(points))
    target_delta = round(mei_score - baseline_mei)

    assert total_pts == target_delta, (
        f"Points sum {total_pts} does not match delta {target_delta}"
    )


def test_explanations_are_deterministic(champion_features: list):
    """Test that repeated calls with the same input produce identical reasons."""
    templates = load_reason_templates()
    shap_vals = np.array(
        [0.12, -0.05, 0.08, 0.02] + [0.0] * (len(champion_features) - 4)
    )
    vals = np.array([1.5, 3.2, 0.8, 2.1] + [1.0] * (len(champion_features) - 4))

    reasons1 = generate_student_reasons(
        vals,
        shap_vals,
        champion_features,
        mei_score=60.0,
        baseline_mei=42.0,
        templates=templates,
    )
    reasons2 = generate_student_reasons(
        vals,
        shap_vals,
        champion_features,
        mei_score=60.0,
        baseline_mei=42.0,
        templates=templates,
    )

    assert len(reasons1) == len(reasons2)
    for r1, r2 in zip(reasons1, reasons2):
        assert r1["text"] == r2["text"]
        assert r1["points"] == r2["points"]
        assert r1["direction"] == r2["direction"]


def test_score_dataframe_with_explain(active_run_dir: Path):
    """Test that score_dataframe with explain=True generates valid top_reasons."""
    sample_file = Path("data/samples/sample_upload_clean.xlsx")
    if not sample_file.exists():
        sample_file = Path("data/raw/survey_raw_sample.csv")

    if sample_file.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(sample_file).head(3)
    else:
        df = pd.read_csv(sample_file).head(3)

    scored_df = score_dataframe(df, run_dir=active_run_dir, explain=True)

    assert "top_reasons" in scored_df.columns
    assert len(scored_df) == len(df)
    for reason in scored_df["top_reasons"]:
        assert isinstance(reason, str)
        assert len(reason) > 10
        assert "pts" in reason
