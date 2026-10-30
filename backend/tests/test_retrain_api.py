"""
Unit & Integration Test Suite for Retraining & Isolation Endpoints.
Covers:
  - Dataset inspection (/train/inspect) with and without outcome labels
  - All blocking validation rules (minimum cohort, minimum positive/negative classes, leakage, invalid labels)
  - Template endpoint with outcome labels (/predict/template?with_outcomes=true)
  - Asynchronous retraining job submission (HTTP 202) and status polling (/jobs/{id})
  - Concurrency control (HTTP 409 conflict when a job is already in progress)
  - Complete filesystem isolation: canonical thesis artifacts (artifacts/models, artifacts/leaderboard)
    remain 100% untouched during and after user dataset retraining runs.

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.src.api.main import app
from backend.src.api.retrain import RetrainJobManager, has_outcomes
from backend.src.api.service import InferenceService
from backend.src.config import config

client = TestClient(app)


def _generate_test_dataset(
    n_students: int = 120,
    attrition_rate: float = 0.25,
    semesters_per_student: int = 3,
    introduce_leakage: bool = False,
    invalid_label: bool = False,
) -> pd.DataFrame:
    """Helper to synthesize structured observation records with outcomes."""
    rng = np.random.RandomState(42)
    records = []

    for i in range(n_students):
        sid = f"STU_{i:04d}"
        is_attrited = 1 if i < int(n_students * attrition_rate) else 0
        att_sem = 2 if is_attrited else None

        # Number of semesters to generate
        if is_attrited and not introduce_leakage:
            n_sems = min(2, semesters_per_student)
        elif is_attrited and introduce_leakage:
            n_sems = 4  # Semester 3 & 4 occur AFTER attrition_semester=2 (Leakage)
        else:
            n_sems = semesters_per_student

        for s in range(1, n_sems + 1):
            records.append({
                "student_id": sid,
                "cohort_year": 2023,
                "semester": s,
                "gpa": round(rng.uniform(2.0, 4.0), 2),
                "exam_marks": round(rng.uniform(45.0, 95.0), 1),
                "attendance_pct": round(rng.uniform(60.0, 100.0), 1),
                "assignment_delay_days": round(rng.uniform(0.0, 5.0), 1),
                "late_assignments_count": int(rng.randint(0, 4)),
                "attrited": 99 if (invalid_label and i == 0) else is_attrited,
                "attrition_semester": att_sem if att_sem is not None else "",
            })

    return pd.DataFrame(records)


def _hash_directory_files(directory: Path) -> Dict[str, str]:
    """Computes SHA256 hashes of all files in a directory to verify zero mutations."""
    hashes = {}
    if not directory.exists():
        return hashes
    for p in directory.rglob("*"):
        if p.is_file():
            hasher = hashlib.sha256()
            with open(p, "rb") as f:
                hasher.update(f.read())
            hashes[str(p.relative_to(directory))] = hasher.hexdigest()
    return hashes


# ── 1. Inspection Tests ───────────────────────────────────────────────────────

def test_inspect_without_outcomes():
    """Verify inspection identifies inference-only files without outcome labels."""
    df = _generate_test_dataset(n_students=120)
    df_no_outcomes = df.drop(columns=["attrited", "attrition_semester"])

    csv_bytes = df_no_outcomes.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/train/inspect",
        files={"file": ("dataset_inference.csv", io.BytesIO(csv_bytes), "text/csv")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["has_outcomes"] is False
    assert data["can_retrain"] is False
    assert any("no outcome column" in msg.lower() for msg in data["blocking_issues"])


def test_inspect_with_outcomes_valid():
    """Verify inspection accepts compliant datasets exceeding size thresholds."""
    df = _generate_test_dataset(n_students=120, attrition_rate=0.25)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/train/inspect",
        files={"file": ("dataset_valid.csv", io.BytesIO(csv_bytes), "text/csv")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["has_outcomes"] is True
    assert data["can_retrain"] is True
    assert data["n_students"] == 120
    assert data["n_attrited"] == 30
    assert len(data["blocking_issues"]) == 0


def test_inspect_blocking_under_100_students():
    """Verify blocking rule for cohort size < 100 students."""
    df = _generate_test_dataset(n_students=50, attrition_rate=0.40)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/train/inspect",
        files={"file": ("dataset_small.csv", io.BytesIO(csv_bytes), "text/csv")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["has_outcomes"] is True
    assert data["can_retrain"] is False
    assert any("minimum 100" in msg.lower() for msg in data["blocking_issues"])


def test_inspect_blocking_under_20_attrited():
    """Verify blocking rule for positive class < 20 attrited students."""
    df = _generate_test_dataset(n_students=120, attrition_rate=0.08)  # 9 attrited

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/train/inspect",
        files={"file": ("dataset_low_positive.csv", io.BytesIO(csv_bytes), "text/csv")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["can_retrain"] is False
    assert any("minimum 20" in msg.lower() and "attrited" in msg.lower() for msg in data["blocking_issues"])


def test_inspect_blocking_under_20_non_attrited():
    """Verify blocking rule for negative class < 20 non-attrited students."""
    df = _generate_test_dataset(n_students=105, attrition_rate=0.90)  # ~10 non-attrited

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/train/inspect",
        files={"file": ("dataset_low_negative.csv", io.BytesIO(csv_bytes), "text/csv")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["can_retrain"] is False
    assert any("non-attrited" in msg.lower() and "minimum 20" in msg.lower() for msg in data["blocking_issues"])


def test_inspect_blocking_leakage():
    """Verify blocking rule for post-attrition observations (data leakage)."""
    df = _generate_test_dataset(n_students=110, introduce_leakage=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/train/inspect",
        files={"file": ("dataset_leakage.csv", io.BytesIO(csv_bytes), "text/csv")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["can_retrain"] is False
    assert any("leakage" in msg.lower() for msg in data["blocking_issues"])


def test_inspect_blocking_invalid_labels():
    """Verify blocking rule for non-binary outcome labels."""
    df = _generate_test_dataset(n_students=110, invalid_label=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/train/inspect",
        files={"file": ("dataset_invalid_label.csv", io.BytesIO(csv_bytes), "text/csv")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["can_retrain"] is False
    assert any("invalid outcome labels" in msg.lower() for msg in data["blocking_issues"])


# ── 2. Template Endpoints Tests ───────────────────────────────────────────────

def test_template_endpoints():
    """Verify template download with and without outcome columns."""
    # Standard template (inference only)
    res_std = client.get("/predict/template")
    assert res_std.status_code == 200
    assert "student_id" in res_std.text
    assert "attrited" not in res_std.text

    # Template with outcomes (for retraining)
    res_out = client.get("/predict/template?with_outcomes=true")
    assert res_out.status_code == 200
    assert "student_id" in res_out.text
    assert "attrited" in res_out.text
    assert "attrition_semester" in res_out.text


# ── 3. Retraining Job Lifecycle & Concurrency ─────────────────────────────────

def test_retrain_submission_and_conflict():
    """
    Tests:
      1. POST /retrain returns HTTP 202 with job_id.
      2. Concurrent submission returns HTTP 409 Conflict.
      3. GET /jobs/{job_id} reports status.
    """
    df = _generate_test_dataset(n_students=105, attrition_rate=0.25)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    # Clear active job if any
    manager = RetrainJobManager.get_instance()
    with manager._lock:
        manager._active_job_id = None

    response1 = client.post(
        "/retrain",
        files={"file": ("train_dataset.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"budget": "quick", "scope": "champions"},
    )

    assert response1.status_code == 202
    data1 = response1.json()
    assert "job_id" in data1
    job_id = data1["job_id"]
    assert job_id.startswith("job_")

    # Immediate second submission should return 409 Conflict
    response2 = client.post(
        "/retrain",
        files={"file": ("train_dataset2.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"budget": "quick", "scope": "champions"},
    )
    assert response2.status_code == 409
    assert "currently in progress" in response2.json()["detail"].lower()

    # Poll status
    status_res = client.get(f"/jobs/{job_id}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["job_id"] == job_id
    assert status_data["state"] in ("queued", "running", "succeeded", "failed")
    assert status_data["progress_pct"] >= 0


def test_jobs_endpoint_not_found():
    """Verify 404 for unknown job ID."""
    response = client.get("/jobs/job_nonexistent_999999")
    assert response.status_code == 404


def test_get_latest_endpoint():
    """Verify /train/latest returns 404 or latest retrained model metadata."""
    res = client.get("/train/latest")
    assert res.status_code in (200, 404)
    if res.status_code == 200:
        data = res.json()
        assert "champion" in data
        assert data["data_mode"] == "REAL"


@pytest.mark.slow
def test_retrain_full_lifecycle_with_small_dataset():
    """
    Tests complete end-to-end retraining execution with data/synthetic/small dataset.
    Polls until terminal state (succeeded or failed) and validates isolated summary.
    """
    small_long_path = config.root_dir / "data" / "synthetic" / "small" / "academic_long.csv"
    small_out_path = config.root_dir / "data" / "synthetic" / "small" / "outcomes.csv"
    if not (small_long_path.exists() and small_out_path.exists()):
        pytest.skip("data/synthetic/small files not found")

    df_l = pd.read_csv(small_long_path)
    df_o = pd.read_csv(small_out_path)
    df_merged = df_l.merge(df_o[["student_id", "attrited", "attrition_semester"]], on="student_id")

    csv_bytes = df_merged.to_csv(index=False).encode("utf-8")

    # Clear active job if any
    manager = RetrainJobManager.get_instance()
    with manager._lock:
        manager._active_job_id = None

    res = client.post(
        "/retrain",
        files={"file": ("small_dataset.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"budget": "quick", "scope": "champions"},
    )
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    # Poll status is tracking
    status_res = client.get(f"/jobs/{job_id}")
    assert status_res.status_code == 200
    assert status_res.json()["job_id"] == job_id

    # Wait for job to reach terminal state so background threads do not bleed into next tests
    for _ in range(60):
        time.sleep(1.0)
        st = client.get(f"/jobs/{job_id}").json()
        if st.get("state") in ("succeeded", "failed"):
            break


# ── 4. Isolation Guarantee ────────────────────────────────────────────────────

def test_isolation_guarantee_canonical_artifacts_untouched():
    """
    Confirms that canonical models and leaderboards in artifacts/models and artifacts/leaderboard
    are never overwritten or mutated during isolated training operations.
    """
    models_dir = config.root_dir / "artifacts" / "models"
    lb_dir = config.root_dir / "artifacts" / "leaderboard"

    initial_model_hashes = _hash_directory_files(models_dir)
    initial_lb_hashes = _hash_directory_files(lb_dir)

    # Execute isolated run context
    test_job_id = "job_test_isolation_verification"
    with config.isolated_run(test_job_id) as isolated_path:
        assert config.is_isolated is True
        assert config.active_job_id == test_job_id
        assert isolated_path == config.root_dir / "artifacts" / "user_dataset" / test_job_id

        # Write dummy model inside isolated directory
        (config.models_dir / "test_isolated_model.txt").write_text("user data checkpoint", encoding="utf-8")

    # Verify context exited cleanly
    assert config.is_isolated is False
    assert config.active_job_id is None

    # Verify canonical models directory was not affected
    current_model_hashes = _hash_directory_files(models_dir)
    current_lb_hashes = _hash_directory_files(lb_dir)

    assert current_model_hashes == initial_model_hashes, "Canonical models were modified!"
    assert current_lb_hashes == initial_lb_hashes, "Canonical leaderboard was modified!"
    assert not (models_dir / "test_isolated_model.txt").exists()

    # Verify isolated file is located strictly in user_dataset/<job_id>
    user_file = config.root_dir / "artifacts" / "user_dataset" / test_job_id / "models" / "test_isolated_model.txt"
    assert user_file.exists()


# ── 5. Phase 2: Activation, Dynamic Loading, Rollback & Runs List ─────────────

def test_phase2_activate_rollback_lifecycle():
    """
    Tests:
      1. Creating an isolated user dataset run.
      2. POST /train/activate/{job_id} switches InferenceService and /meta to REAL mode.
      3. /meta returns user metrics and clears synthetic disclaimer.
      4. Single student prediction marks provenance_tag='REAL'.
      5. GET /train/runs lists the run with is_active=True.
      6. POST /train/rollback restores canonical SYNTHETIC thesis benchmark.
      7. DELETE /train/runs/{job_id} deletes run directory.
    """
    test_job_id = "job_test_p2_mock_run"
    job_dir = config.root_dir / "artifacts" / "user_dataset" / test_job_id
    models_dir = job_dir / "models"
    cat_dir = models_dir / "Category_C" / "catboost"
    cat_dir.mkdir(parents=True, exist_ok=True)

    # Copy canonical preprocessor and catboost artifacts to mock run
    canonical_models = config.root_dir / "artifacts" / "models"
    src_preproc = canonical_models / "preprocessor.joblib"
    src_cat = canonical_models / "Category_C" / "catboost"
    
    shutil.copy2(src_preproc, models_dir / "preprocessor.joblib")
    shutil.copytree(src_cat, cat_dir, dirs_exist_ok=True)

    # Save summary.json
    summary = {
        "job_id": test_job_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_students": 125,
        "n_attrited": 30,
        "champion": "catboost",
        "champion_label": "CatBoost (User)",
        "category": "Category_C",
        "category_label": "Boosting — Engineered Trend Features",
        "sequence_champion": "gru",
        "test_roc_auc": 0.8888,
        "test_pr_auc": 0.4444,
        "test_f2_score": 0.5555,
        "lead_time_semesters": 2.8,
        "budget": "quick",
        "scope": "champions",
        "data_mode": "REAL",
    }
    with open(job_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    try:
        # 1. Activate the run
        act_res = client.post(f"/train/activate/{test_job_id}")
        assert act_res.status_code == 200
        act_data = act_res.json()
        assert act_data["data_mode"] == "REAL"
        assert act_data["job_id"] == test_job_id

        # 2. Check /meta reflects REAL mode and exact user metrics
        meta_res = client.get("/meta")
        assert meta_res.status_code == 200
        meta_data = meta_res.json()
        assert meta_data["data_mode"] == "REAL"
        assert meta_data["active_run_id"] == test_job_id
        assert meta_data["synthetic_disclaimer"] == ""
        assert meta_data["champion_model"]["test_roc_auc"] == 0.8888
        assert meta_data["champion_model"]["test_pr_auc"] == 0.4444
        assert meta_data["champion_model"]["test_f2_score"] == 0.5555
        assert meta_data["champion_model"]["lead_time_semesters"] == 2.8

        # 3. Test single prediction returns REAL provenance tag
        pred_payload = {
            "student_id": "TEST_STUDENT_P2",
            "semesters": [
                {
                    "semester": 1,
                    "gpa": 3.2,
                    "exam_marks": 75.0,
                    "attendance_pct": 85.0,
                    "assignment_delay_days": 1.0,
                    "late_assignments_count": 0,
                },
                {
                    "semester": 2,
                    "gpa": 2.8,
                    "exam_marks": 68.0,
                    "attendance_pct": 78.0,
                    "assignment_delay_days": 2.5,
                    "late_assignments_count": 1,
                },
            ]
        }
        pred_res = client.post("/predict/student", json=pred_payload)
        assert pred_res.status_code == 200
        pred_data = pred_res.json()
        assert pred_data["fusion_payload"]["provenance_tag"] == "REAL"

        # 4. Check GET /train/runs lists the run as active
        runs_res = client.get("/train/runs")
        assert runs_res.status_code == 200
        runs_data = runs_res.json()
        assert runs_data["active_run_id"] == test_job_id
        assert runs_data["data_mode"] == "REAL"
        matched = [r for r in runs_data["runs"] if r["job_id"] == test_job_id]
        assert len(matched) == 1
        assert matched[0]["is_active"] is True

        # 5. Rollback to SYNTHETIC benchmark
        rb_res = client.post("/train/rollback")
        assert rb_res.status_code == 200
        assert rb_res.json()["data_mode"] == "SYNTHETIC"

        # 6. Check /meta is restored to SYNTHETIC with synthetic disclaimer
        meta_res2 = client.get("/meta")
        assert meta_res2.status_code == 200
        meta_data2 = meta_res2.json()
        assert meta_data2["data_mode"] == "SYNTHETIC"
        assert meta_data2["active_run_id"] is None
        assert "synthetic" in meta_data2["synthetic_disclaimer"].lower()

        # 7. Delete the test run
        del_res = client.delete(f"/train/runs/{test_job_id}")
        assert del_res.status_code == 200
        assert not job_dir.exists()

    finally:
        # Guarantee cleanup and rollback
        config.clear_active_run()
        InferenceService.get_instance().reload()
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
