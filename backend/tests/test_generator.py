"""
Tests for Latent Student Generator, Schema Conformance, and Naive Baseline.
"""

import pytest
import numpy as np
import pandas as pd
from backend.src.data.generator import LatentStudentGenerator
from backend.src.data.task import create_prefix_task_samples
from backend.src.evaluation.naive_baseline import evaluate_naive_gpa_cutoff


def test_generator_determinism():
    """Verify generator produces bitwise identical dataframes given the same seed."""
    gen1 = LatentStudentGenerator(seed=42, preset="realistic")
    df_long1, df_out1, _ = gen1.generate_cohort(n_students=200, max_semesters=4, missing_rate=0.0)

    gen2 = LatentStudentGenerator(seed=42, preset="realistic")
    df_long2, df_out2, _ = gen2.generate_cohort(n_students=200, max_semesters=4, missing_rate=0.0)

    pd.testing.assert_frame_equal(df_long1, df_long2)
    pd.testing.assert_frame_equal(df_out1, df_out2)


def test_no_label_or_latent_leakage():
    """Verify that academic_long contains ONLY observable features and zero outcome or hidden trait leakage."""
    gen = LatentStudentGenerator(seed=123, preset="realistic")
    df_long, _, _ = gen.generate_cohort(n_students=100)

    expected_cols = {
        "student_id", "cohort_year", "semester", "gpa", "exam_marks",
        "attendance_pct", "assignment_delay_days", "late_assignments_count"
    }
    assert set(df_long.columns) == expected_cols

    # Ensure no hidden traits leaked into columns
    forbidden_terms = ["attrit", "label", "target", "ability", "conscientiousness", "resilience", "hazard", "latent"]
    for col in df_long.columns:
        for term in forbidden_terms:
            assert term not in col.lower(), f"Forbidden leakage term '{term}' found in column '{col}'"


def test_valid_feature_ranges():
    """Verify that generated numeric features strictly adhere to physical and academic boundaries."""
    gen = LatentStudentGenerator(seed=999, preset="realistic")
    df_long, df_outcomes, _ = gen.generate_cohort(n_students=300)

    valid_gpa = df_long["gpa"].dropna()
    assert (valid_gpa >= 0.0).all() and (valid_gpa <= 4.0).all()

    valid_exam = df_long["exam_marks"].dropna()
    assert (valid_exam >= 0.0).all() and (valid_exam <= 100.0).all()

    valid_att = df_long["attendance_pct"].dropna()
    assert (valid_att >= 0.0).all() and (valid_att <= 100.0).all()

    valid_delay = df_long["assignment_delay_days"].dropna()
    assert (valid_delay >= 0.0).all()

    valid_late = df_long["late_assignments_count"].dropna()
    assert (valid_late >= 0).all()

    # Outcomes validity
    assert set(df_outcomes["attrited"].unique()).issubset({0, 1})
    attr_rate = df_outcomes["attrited"].mean()
    assert 0.12 <= attr_rate <= 0.30, f"Attrition rate {attr_rate} outside realistic bounds [12%, 30%]"


def test_prefix_task_samples():
    """Verify prefix task samples correctly structure window horizon and retain student_id."""
    gen = LatentStudentGenerator(seed=42, preset="realistic")
    df_long, df_outcomes, _ = gen.generate_cohort(n_students=200)

    df_task = create_prefix_task_samples(df_long, df_outcomes, min_window=2, max_window=4)
    assert "student_id" in df_task.columns
    assert "window_t" in df_task.columns
    assert "target_attrite_next_2_sems" in df_task.columns
    assert set(df_task["target_attrite_next_2_sems"].unique()).issubset({0, 1})

    # Grouped integrity: Each student appears with valid windows
    assert df_task["student_id"].nunique() <= 200


def test_naive_baseline_mediocre_recall():
    """Verify naive GPA < 2.0 cut-off yields mediocre recall, validating research need."""
    gen = LatentStudentGenerator(seed=42, preset="realistic")
    df_long, df_outcomes, _ = gen.generate_cohort(n_students=500)
    df_task = create_prefix_task_samples(df_long, df_outcomes, min_window=2, max_window=5)

    results = evaluate_naive_gpa_cutoff(df_long, df_task, gpa_threshold=2.0)
    assert results["recall"] < 0.65, f"Naive recall {results['recall']} unexpectedly high; must be mediocre"
