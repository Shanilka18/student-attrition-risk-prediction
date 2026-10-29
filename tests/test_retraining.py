"""Comprehensive test suite for Phase 16 retraining pipeline and small-data guards."""

import threading

import numpy as np
import pandas as pd
import pytest

from mei.config import ARTIFACTS_DIR, RUNS_DIR
from mei.registry.retrain import (
    SmallDataGuards,
    activate_pipeline_run,
    compare_pipeline_runs,
    get_retrain_regression_specs,
    get_retrain_specs,
    run_full_pipeline,
    validate_retrain_dataset,
)


def make_test_survey_df(n_students: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic survey questionnaire responses with all 35 canonical columns."""
    rng = np.random.default_rng(seed)
    data = {
        "student_id": [f"STUDENT_TEST_{i:04d}" for i in range(n_students)],
        "timestamp": ["2026-10-28 10:00:00"] * n_students,
        "consent": [1] * n_students,
    }
    indicators = [
        "resource_access",
        "study_time",
        "review_time",
        "study_consistency",
        "participation",
        "assignment_engagement",
        "quiz_practice",
    ]
    for ind in indicators:
        for p in range(1, 5):
            data[f"{ind}_p{p}"] = rng.integers(1, 6, size=n_students)

    data["attention_check"] = [3] * n_students
    # 3 outcome questions (A1, A2, A3)
    data["intent_leave"] = rng.integers(1, 6, size=n_students)
    data["intent_defer"] = rng.integers(1, 6, size=n_students)
    data["intent_not_continue"] = rng.integers(1, 6, size=n_students)
    return pd.DataFrame(data)


def test_small_data_guards_refusal_under_150():
    """Gives refusal when cohort sample size is below minimum threshold of 150 students."""
    guards = SmallDataGuards.evaluate(
        n_samples=100,
        pos_rate_pct=22.0,
        has_outcome=True,
        missing_outcome_cols=[],
    )
    assert guards["can_retrain"] is False
    assert guards["traffic_light"] == "RED"
    assert "at least 150 students" in guards["refusal_reason"]


def test_small_data_guards_missing_outcomes():
    """Gives refusal when outcome questions are missing."""
    guards = SmallDataGuards.evaluate(
        n_samples=300,
        pos_rate_pct=0.0,
        has_outcome=False,
        missing_outcome_cols=["intent_leave", "intent_defer"],
    )
    assert guards["can_retrain"] is False
    assert guards["traffic_light"] == "RED"
    assert "intent_leave" in guards["refusal_reason"]


def test_small_data_guards_small_cohort_configuration():
    """Enforces 5x5 repeated CV and disables deep Transformer/Stacking when 150 <= N < 500."""
    guards = SmallDataGuards.evaluate(
        n_samples=250,
        pos_rate_pct=25.0,
        has_outcome=True,
        missing_outcome_cols=[],
    )
    assert guards["can_retrain"] is True
    assert guards["is_small_cohort"] is True
    assert guards["cv_scheme"] == "5x5_repeated"
    assert guards["cv_folds_count"] == 25
    assert "tiny_transformer" in guards["disabled_models"]
    assert "stacking_classifier" in guards["disabled_models"]
    assert any("fewer than 500 students" in w for w in guards["warnings"])


def test_small_data_guards_imbalance_directive():
    """Activates class weights and alerts when positive rate is under 15%."""
    guards = SmallDataGuards.evaluate(
        n_samples=600,
        pos_rate_pct=11.5,
        has_outcome=True,
        missing_outcome_cols=[],
    )
    assert guards["is_imbalanced"] is True
    assert guards["use_class_weights"] is True
    assert any("class imbalance" in w.lower() for w in guards["warnings"])


def test_quick_retraining_selects_registered_specs():
    """Quick retraining filters model spec objects rather than registry keys."""
    guards = {"disabled_models": ["gru", "gru_regressor"]}

    classification = get_retrain_specs("quick", guards)
    regression = get_retrain_regression_specs("quick", guards)

    assert {spec.id for spec in classification} >= {
        "majority_dummy",
        "logistic_regression_l2",
        "random_forest",
    }
    assert {spec.id for spec in regression} >= {"ridge_regressor", "lasso_regressor"}
    assert "gru" not in {spec.id for spec in classification}
    assert "gru_regressor" not in {spec.id for spec in regression}


def test_validation_endpoint_logic(tmp_path):
    """Validates real spreadsheet on disk and returns correct summary stats."""
    sample_df = make_test_survey_df(n_students=180, seed=123)
    test_file = tmp_path / "survey_validation_test.csv"
    sample_df.to_csv(test_file, index=False)

    val_res = validate_retrain_dataset(test_file)
    assert val_res["total_rows"] == 180
    assert val_res["can_retrain"] is True
    assert val_res["small_data_guard"]["is_small_cohort"] is True
    assert val_res["class_balance"]["at_risk_count"] > 0
    assert val_res["estimated_time"]["quick_minutes"] > 0


def test_cooperative_cancellation(tmp_path):
    """Verifies that cancel event cleanly interrupts pipeline execution."""
    sample_df = make_test_survey_df(n_students=160, seed=456)
    test_file = tmp_path / "survey_cancel_test.csv"
    sample_df.to_csv(test_file, index=False)

    cancel_event = threading.Event()
    cancel_event.set()  # Pre-cancelled

    with pytest.raises(InterruptedError, match="cancelled by user"):
        run_full_pipeline(
            dataset_path=test_file,
            mode="quick",
            cancel_event=cancel_event,
        )


def test_quick_retraining_and_governance(tmp_path):
    """Executes quick retraining on small cohort, verifies run isolation, cards, and activation."""
    active_file = ARTIFACTS_DIR / "active_run.txt"
    initial_active_run = (
        active_file.read_text(encoding="utf-8").strip()
        if active_file.exists()
        else "none"
    )

    # 160 students (small cohort, triggers 5x5 repeated CV & disabled transformer)
    cohort_df = make_test_survey_df(n_students=160, seed=789)
    test_file = tmp_path / "quick_retrain_cohort.xlsx"
    cohort_df.to_excel(test_file, index=False)

    progress_updates = []

    def on_progress(p: float, msg: str):
        progress_updates.append((p, msg))

    # Run quick pipeline
    result = run_full_pipeline(
        dataset_path=test_file,
        mode="quick",
        seed=42,
        update_fn=on_progress,
    )

    new_run_id = result["run_id"]
    run_dir = RUNS_DIR / new_run_id

    # 1. Output verification
    assert run_dir.exists()
    assert (run_dir / "champion" / "model_card.json").exists()
    assert (run_dir / "champion" / "model_card.md").exists()
    assert (run_dir / "data_card.md").exists()
    assert (run_dir / "leaderboard_classification.csv").exists()
    assert len(progress_updates) >= 5

    # 2. Run isolation: active_run.txt must NOT have changed during training
    current_active = (
        active_file.read_text(encoding="utf-8").strip()
        if active_file.exists()
        else "none"
    )
    assert (
        current_active == initial_active_run
    ), "Active run was mutated before explicit activation!"

    # 3. Side-by-side run comparison
    if initial_active_run != "none" and (RUNS_DIR / initial_active_run).exists():
        comp = compare_pipeline_runs(initial_active_run, new_run_id)
        assert "run_a" in comp
        assert "run_b" in comp
        assert "comparison" in comp
        assert "verdict_summary" in comp["comparison"]

    # 4. Model activation and audit log verification
    act_res = activate_pipeline_run(new_run_id, actor="Test Suite Verification")
    assert act_res["status"] == "activated"
    assert active_file.read_text(encoding="utf-8").strip() == new_run_id

    audit_log = ARTIFACTS_DIR / "audit.log"
    assert audit_log.exists()
    audit_text = audit_log.read_text(encoding="utf-8")
    assert f"NEW_RUN={new_run_id}" in audit_text
    assert "Test Suite Verification" in audit_text

    # 5. Clean rollback to original active run
    if initial_active_run != "none":
        activate_pipeline_run(initial_active_run, actor="Test Suite Rollback")
        assert active_file.read_text(encoding="utf-8").strip() == initial_active_run
