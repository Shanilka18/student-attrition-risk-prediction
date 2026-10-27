"""
Comprehensive Test Suite for Academic Intelligence API Endpoints (Phase 8).
Validates:
  - System: GET /health, GET /meta
  - Model Zoo & Research: GET /leaderboard, GET /models/{name}, GET /research-answer
  - Data & Explanations: GET /data/summary, GET /explain/global, GET /explain/student/{id}
  - Inference: GET /predict/template, POST /predict/student, POST /predict/upload, GET /predict/download/{token}
  - Validation: Malformed CSVs, out-of-range metrics, missing columns, duplicate semesters, wide format, Excel format, empty files, file size limits.
  - SLAs: NFR1 latency constraint (< 2.0 s per student).

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from backend.src.api.main import app
from backend.src.config import config

client = TestClient(app)


# ── 1. System Endpoints Tests ─────────────────────────────────────────────────

def test_health_endpoint():
    """Verify health endpoint confirms service readiness and directory health."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert data["seed"] == 42
    assert data["directories_initialized"]["models"] is True
    assert data["directories_initialized"]["processed_data"] is True


def test_meta_endpoint():
    """Verify meta endpoint returns project identity, champion info, and synthetic disclaimer."""
    response = client.get("/meta")
    assert response.status_code == 200
    data = response.json()
    assert data["project_id"] == "KND_CDAP_IT_05"
    assert data["data_mode"] == "SYNTHETIC"
    assert "synthetic" in data["synthetic_disclaimer"].lower()
    assert data["champion_model"]["model_key"] == "catboost"
    assert data["sequence_champion"]["model_key"] == "gru"
    assert data["nfr1_latency_sla_seconds"] == 2.0


# ── 2. Leaderboard & Research Question Tests ──────────────────────────────────

def test_leaderboard_endpoint():
    """Verify leaderboard returns multi-model benchmark results and category champions."""
    response = client.get("/leaderboard")
    assert response.status_code == 200
    data = response.json()
    assert data["data_source"] == "SYNTHETIC"
    assert len(data["models"]) >= 1
    assert "ranking_weights" in data
    assert "category_champions" in data


def test_model_details_valid():
    """Verify inspection of valid model (CatBoost) returns metrics, curves, and trials."""
    response = client.get("/models/catboost")
    assert response.status_code == 200
    data = response.json()
    assert data["model_key"] == "catboost"
    assert data["category"] == "C"
    assert "validation_metrics" in data
    assert "best_params" in data
    assert isinstance(data["training_curves"], list)


def test_model_details_not_found():
    """Verify 404 error on requesting non-existent model."""
    response = client.get("/models/non_existent_deep_net")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_research_answer_endpoint():
    """Verify research answer returns formal hypothesis test and deltas."""
    response = client.get("/research-answer")
    assert response.status_code == 200
    data = response.json()
    assert "sequence-aware" in data["research_question"]
    assert data["answer"] in ["POSITIVE", "NULL", "NEGATIVE"]
    assert "sequence_champion" in data
    assert "boosting_baseline" in data
    assert "deltas" in data
    assert "statistical_significance" in data
    assert data["data_source"] == "SYNTHETIC"


# ── 3. Data Summary & Explanations Tests ──────────────────────────────────────

def test_data_summary_endpoint():
    """Verify dataset summary returns cohort metrics and visual plots catalogue."""
    response = client.get("/data/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["data_mode"] == "SYNTHETIC"
    assert data["total_students"] == 5000
    assert len(data["variables"]) == 5
    assert len(data["plots_catalog"]) >= 15
    first_plot = data["plots_catalog"][0]
    assert "url_path" in first_plot
    assert first_plot["url_path"].startswith("/static/plots/")


def test_explain_global_endpoint():
    """Verify global explainability returns 22 features, attributions, and audits."""
    response = client.get("/explain/global")
    assert response.status_code == 200
    data = response.json()
    assert data["model_key"] == "catboost"
    assert data["explainer_type"] == "TreeSHAP"
    assert len(data["features"]) == 22
    assert "faithfulness_audit" in data
    assert "stability_audit" in data


def test_explain_student_endpoint():
    """Verify retrieval of student explanation for existing sample."""
    # STU_00002 is in sample_fusion_contract.json
    response = client.get("/explain/student/STU_00002")
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == "STU_00002"
    assert "academic_risk_index" in data
    assert data["risk_band"] in ["Low", "Moderate", "High", "Critical"]
    assert len(data["top_factors"]) >= 1
    assert "High risk" in data["explanation_text"] or "Low risk" in data["explanation_text"] or "Moderate" in data["explanation_text"]


def test_explain_student_not_found():
    """Verify 404 for unknown student ID."""
    response = client.get("/explain/student/UNKNOWN_NONEXISTENT_999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ── 4. Prediction Template Tests ──────────────────────────────────────────────

def test_predict_template_endpoint():
    """Verify template download delivers valid CSV with attachment headers."""
    response = client.get("/predict/template")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]
    csv_text = response.text
    assert "student_id,semester,gpa,exam_marks,attendance_pct,assignment_delay_days,late_assignments_count" in csv_text


# ── 5. Single Student Prediction Tests ────────────────────────────────────────

def test_predict_student_valid():
    """Verify single-student prediction returns calibrated ARI, SHAP factors, and fusion payload."""
    payload = {
        "student_id": "STU_TEST_001",
        "semesters": [
            {
                "semester": 1,
                "gpa": 3.4,
                "exam_marks": 72.0,
                "attendance_pct": 88.0,
                "assignment_delay_days": 1.0,
                "late_assignments_count": 0,
            },
            {
                "semester": 2,
                "gpa": 3.2,
                "exam_marks": 68.0,
                "attendance_pct": 82.0,
                "assignment_delay_days": 2.0,
                "late_assignments_count": 1,
            },
        ],
    }
    response = client.post("/predict/student", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == "STU_TEST_001"
    assert data["window_semester"] == 2
    assert 0.0 <= data["academic_risk_index"] <= 100.0
    assert data["risk_band"] in ["Low", "Moderate", "High", "Critical"]
    assert len(data["top_factors"]) == 3
    assert "fusion_payload" in data
    assert data["fusion_payload"]["participant_id"] == "STU_TEST_001"
    # Latency NFR1 constraint
    assert data["inference_time_ms"] < 2000.0
    assert data["nfr1_compliant"] is True


def test_predict_student_single_semester_warning():
    """Verify single-semester student triggers minimum-semesters rule warning."""
    payload = {
        "student_id": "STU_FRESHMAN_01",
        "semesters": [
            {
                "semester": 1,
                "gpa": 2.8,
                "exam_marks": 60.0,
                "attendance_pct": 75.0,
                "assignment_delay_days": 3.0,
                "late_assignments_count": 1,
            }
        ],
    }
    response = client.post("/predict/student", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["warnings"]) >= 1
    assert "minimum semesters rule" in data["warnings"][0].lower()


def test_predict_student_invalid_gpa():
    """Verify validation error when GPA exceeds 4.0."""
    payload = {
        "student_id": "STU_INVALID",
        "semesters": [
            {
                "semester": 1,
                "gpa": 4.5,  # Invalid: > 4.0
                "exam_marks": 80.0,
                "attendance_pct": 90.0,
                "assignment_delay_days": 0.0,
                "late_assignments_count": 0,
            }
        ],
    }
    response = client.post("/predict/student", json=payload)
    assert response.status_code == 422


def test_predict_student_out_of_order_semesters():
    """Verify validation error when semesters are not ascending."""
    payload = {
        "student_id": "STU_OUT_OF_ORDER",
        "semesters": [
            {
                "semester": 2,
                "gpa": 3.0,
                "exam_marks": 70.0,
                "attendance_pct": 80.0,
                "assignment_delay_days": 1.0,
                "late_assignments_count": 0,
            },
            {
                "semester": 1,
                "gpa": 3.2,
                "exam_marks": 72.0,
                "attendance_pct": 85.0,
                "assignment_delay_days": 0.5,
                "late_assignments_count": 0,
            },
        ],
    }
    response = client.post("/predict/student", json=payload)
    assert response.status_code == 422


# ── 6. Batch Upload Prediction & Validation Tests ─────────────────────────────

def test_predict_upload_valid_csv():
    """Verify successful upload and batch prediction from official upload_template.csv."""
    tpl_path = config.schema_dir / "upload_template.csv"
    with open(tpl_path, "rb") as f:
        response = client.post("/predict/upload", files={"file": ("template.csv", f, "text/csv")})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["total_students"] == 3
    assert data["total_records"] == 12
    assert "summary" in data
    assert data["summary"]["risk_band_counts"]["Low"] >= 1
    assert len(data["predictions"]) == 3
    assert data["inference_time_per_student_ms"] < 2000.0
    assert data["nfr1_compliant"] is True
    assert len(data["download_token"]) > 10


def test_predict_upload_wide_format_csv():
    """Verify auto-detection and conversion of wide-format table into long predictions."""
    wide_csv = (
        "student_id,gpa_sem1,gpa_sem2,exam_marks_sem1,exam_marks_sem2,attendance_pct_sem1,attendance_pct_sem2,assignment_delay_days_sem1,assignment_delay_days_sem2,late_assignments_count_sem1,late_assignments_count_sem2\n"
        "STU_WIDE_01,3.5,3.2,75.0,70.0,90.0,85.0,0.5,1.5,0,1\n"
        "STU_WIDE_02,2.4,1.8,55.0,42.0,70.0,50.0,4.0,8.0,2,4\n"
    )
    bio = io.BytesIO(wide_csv.encode("utf-8"))
    response = client.post("/predict/upload", files={"file": ("wide_cohort.csv", bio, "text/csv")})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["total_students"] == 2
    assert any("converted wide format" in w.lower() for w in data["warnings"])
    sids = [p["student_id"] for p in data["predictions"]]
    assert "STU_WIDE_01" in sids
    assert "STU_WIDE_02" in sids


def test_predict_upload_excel_file():
    """Verify successful upload of Excel (.xlsx) file."""
    df_excel = pd.DataFrame([
        {"student_id": "STU_XLS_01", "semester": 1, "gpa": 3.6, "exam_marks": 78.0, "attendance_pct": 92.0, "assignment_delay_days": 0.0, "late_assignments_count": 0},
        {"student_id": "STU_XLS_01", "semester": 2, "gpa": 3.5, "exam_marks": 75.0, "attendance_pct": 90.0, "assignment_delay_days": 0.5, "late_assignments_count": 0},
    ])
    bio = io.BytesIO()
    df_excel.to_excel(bio, index=False)
    bio.seek(0)

    response = client.post(
        "/predict/upload",
        files={"file": ("students.xlsx", bio, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_students"] == 1
    assert data["predictions"][0]["student_id"] == "STU_XLS_01"


def test_predict_upload_missing_columns():
    """Verify clear error message when required columns are absent."""
    broken_csv = "student_id,semester,gpa\nSTU_01,1,3.4\n"
    bio = io.BytesIO(broken_csv.encode("utf-8"))
    response = client.post("/predict/upload", files={"file": ("broken.csv", bio, "text/csv")})

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(e["error_type"] == "MISSING_COLUMNS" for e in errors)


def test_predict_upload_out_of_range_values():
    """Verify human-readable errors with row numbers for out-of-range metrics."""
    broken_csv = (
        "student_id,semester,gpa,exam_marks,attendance_pct,assignment_delay_days,late_assignments_count\n"
        "STU_01,1,4.5,75.0,90.0,0.0,0\n"  # Row 2: GPA 4.5 > 4.0
        "STU_01,2,3.2,70.0,120.0,0.0,0\n" # Row 3: Attendance 120% > 100%
    )
    bio = io.BytesIO(broken_csv.encode("utf-8"))
    response = client.post("/predict/upload", files={"file": ("out_of_range.csv", bio, "text/csv")})

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert len(errors) >= 2
    row2_err = [e for e in errors if e.get("row") == 2]
    row3_err = [e for e in errors if e.get("row") == 3]
    assert len(row2_err) >= 1
    assert "gpa" in row2_err[0]["message"].lower()
    assert len(row3_err) >= 1
    assert "attendance_pct" in row3_err[0]["message"].lower()


def test_predict_upload_duplicate_semesters():
    """Verify error with row number when duplicate observation exists for student."""
    dup_csv = (
        "student_id,semester,gpa,exam_marks,attendance_pct,assignment_delay_days,late_assignments_count\n"
        "STU_DUP_01,1,3.5,75.0,90.0,0.0,0\n"
        "STU_DUP_01,1,3.2,70.0,85.0,1.0,0\n"  # Duplicate sem 1
    )
    bio = io.BytesIO(dup_csv.encode("utf-8"))
    response = client.post("/predict/upload", files={"file": ("dup.csv", bio, "text/csv")})

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("duplicate" in e["message"].lower() for e in errors)


def test_predict_upload_empty_file():
    """Verify 422 error on empty 0-byte file."""
    bio = io.BytesIO(b"")
    response = client.post("/predict/upload", files={"file": ("empty.csv", bio, "text/csv")})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert errors[0]["error_type"] == "EMPTY_FILE"


def test_predict_upload_huge_file():
    """Verify HTTP 413 error when upload exceeds maximum size cap (10 MB)."""
    # 10 MB + 1024 bytes
    huge_bytes = b"0" * (10 * 1024 * 1024 + 1024)
    bio = io.BytesIO(huge_bytes)
    response = client.post("/predict/upload", files={"file": ("huge.csv", bio, "text/csv")})
    assert response.status_code == 413
    assert "exceeds maximum allowed limit" in response.json()["detail"]


def test_predict_download_export_roundtrip():
    """Verify batch upload produces downloadable CSV and JSON exports."""
    tpl_path = config.schema_dir / "upload_template.csv"
    with open(tpl_path, "rb") as f:
        up_res = client.post("/predict/upload", files={"file": ("template.csv", f, "text/csv")})
    token = up_res.json()["download_token"]

    # 1. Download CSV
    res_csv = client.get(f"/predict/download/{token}?format=csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]
    assert "STU_DEMO_01" in res_csv.text

    # 2. Download JSON
    res_json = client.get(f"/predict/download/{token}?format=json")
    assert res_json.status_code == 200
    assert "application/json" in res_json.headers["content-type"]
    data = res_json.json()
    assert len(data) == 3


def test_predict_download_invalid_token():
    """Verify 404 on invalid or expired export token."""
    response = client.get("/predict/download/non-existent-token-12345?format=csv")
    assert response.status_code == 404
