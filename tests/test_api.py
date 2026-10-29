"""Comprehensive Integration Tests for the FastAPI Service.

Validates:
- Health and metadata contracts (including the mandatory synthetic data banner).
- Leaderboard and model diagnostic inspection routes.
- SHAP explainability contracts and deterministic non-causal narratives.
- Causal inference results, DAG graph topology, and EconML gate status.
- Multipart file upload batch scoring, PII sanitization, and downloadable bundles.
- Standardized group Fusion Framework score endpoints.
- Guarded asynchronous retraining job orchestration.
- Request payload bounds and security guardrails.
"""

import io
import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from src.api.main import app, job_registry, IP_REQUEST_LOGS


@pytest.fixture(scope="module")
def client():
    """Create FastAPI TestClient instance."""
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# 1. System Health & Metadata Tests
# ---------------------------------------------------------------------------


def test_get_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert data["version"] == "1.0.0"


def test_get_metadata_banner_flag(client):
    response = client.get("/meta")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert data["data_banner"] == "SYNTHETIC - pipeline validation only"
    assert data["is_calibrated"] is True
    assert data["calibration_method"] == "isotonic"


# ---------------------------------------------------------------------------
# 2. Leaderboard & Model Inspection Tests
# ---------------------------------------------------------------------------


def test_get_leaderboard_c(client):
    response = client.get("/leaderboard?track=C")
    assert response.status_code == 200
    data = response.json()
    assert data["track"] == "C"
    assert "selected_model" in data
    assert data["total_models"] >= 15
    assert "category_winners" in data
    assert "leaderboard" in data
    assert len(data["leaderboard"]) > 0


def test_get_model_metrics_and_history(client):
    # Metrics
    res_metrics = client.get("/models/logistic_l2/metrics")
    assert res_metrics.status_code == 200
    metrics_data = res_metrics.json()
    assert metrics_data["model"] == "logistic_l2"
    assert len(metrics_data["splits"]) >= 1
    assert "roc" in metrics_data["curves"]
    assert "pr" in metrics_data["curves"]
    assert "calibration" in metrics_data["curves"]

    # History
    res_history = client.get("/models/logistic_l2/history")
    assert res_history.status_code == 200
    history_data = res_history.json()
    assert history_data["model"] == "logistic_l2"
    assert "epochs_or_iterations" in history_data


def test_get_all_splits_metrics(client):
    response = client.get("/metrics/all-splits")
    assert response.status_code == 200
    data = response.json()
    assert data["records_count"] > 0
    assert "data" in data
    first = data["data"][0]
    assert "model" in first
    assert "split" in first


# ---------------------------------------------------------------------------
# 3. Explainability (SHAP) Tests
# ---------------------------------------------------------------------------


def test_get_shap_global(client):
    response = client.get("/shap/global")
    assert response.status_code == 200
    data = response.json()
    assert "base_value" in data
    assert data["feature_count"] > 0
    assert len(data["global_importance"]) > 0
    assert "does not prove these factors caused the outcome" in data["disclaimer"]


def test_get_shap_student_found_and_not_found(client):
    # Valid student
    response = client.get("/shap/student/P-00002")
    assert response.status_code == 200
    data = response.json()
    assert data["participant_id"] == "P-00002"
    assert "eri_score" in data
    assert "main_drivers" in data
    assert "protective_factors" in data
    assert "does not prove these factors caused the outcome" in data["disclaimer"]

    # Invalid student
    res_404 = client.get("/shap/student/NON_EXISTENT_999")
    assert res_404.status_code == 404


def test_get_shap_stability(client):
    response = client.get("/shap/stability")
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["top3_overlap"] <= 1.0
    assert 0.0 <= data["top5_overlap"] <= 1.0
    assert -1.0 <= data["spearman_rank_correlation"] <= 1.0
    assert isinstance(data["is_stable"], bool)


# ---------------------------------------------------------------------------
# 4. Causal Inference & DAG Tests
# ---------------------------------------------------------------------------


def test_get_causal_summary(client):
    response = client.get("/causal/summary")
    assert response.status_code == 200
    data = response.json()
    assert "exposures" in data
    assert "financial_hardship" in data["exposures"]
    assert "long_commute" in data["exposures"]
    assert "Adjusted association under stated assumptions" in data["disclaimer"]


def test_get_causal_diagnostics(client):
    response = client.get("/causal/diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert "balance_smd" in data
    assert "exposure_diagnostics" in data
    assert "financial_hardship" in data["balance_smd"]


def test_get_causal_heterogeneity_gate(client):
    response = client.get("/causal/heterogeneity")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["gate_passed"] is True
    assert data["n_samples"] >= 300
    assert data["n_events"] >= 60
    assert "subgroups" in data


def test_get_dag_structure_and_image(client):
    # DAG JSON
    res_dag = client.get("/dag")
    assert res_dag.status_code == 200
    data = res_dag.json()
    assert len(data["nodes"]) >= 14
    assert len(data["edges"]) >= 20
    assert "financial_hardship" in data["exposures"]
    assert "baseline_gpa_band" in data["confounders"]
    assert "later_gpa_decline" in data["mediators"]

    # DAG Image
    res_img = client.get("/dag/image")
    assert res_img.status_code == 200
    assert res_img.headers["content-type"] == "image/png"
    assert len(res_img.content) > 10000


# ---------------------------------------------------------------------------
# 5. Fusion Framework Integration Tests
# ---------------------------------------------------------------------------


def test_fusion_score_lookup_by_ids(client):
    payload = {"participant_ids": ["P-00002", "P-00003"]}
    response = client.post("/fusion/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_scored"] == 2
    assert data["explanation_type"] == "model_explanation"
    assert data["scores"][0]["participant_id"] == "P-00002"
    assert 0.0 <= data["scores"][0]["eri"] <= 100.0


def test_fusion_score_inline_students(client):
    student_record = {
        "household_income_bracket": "Below 50,000",
        "financial_stress": 4,
        "commute_time_min": 60,
        "commute_cost_lkr_week": 2000,
        "housing_type": "Rented room",
        "internet_reliability": 2,
        "internet_affordability": 2,
        "device_type": "Smartphone only",
        "device_quality": 2,
        "quiet_study_space": 1,
        "part_time_employment": 1,
        "work_hours_per_week": 20,
        "scholarship_bursary": 0,
        "dependents_count": 2,
        "district_type": "Rural",
        "year_of_study": 2,
        "faculty": "Computing",
        "baseline_gpa_band": "2.0-2.5",
        "baseline_engagement_score": 45.0,
    }
    payload = {"students": [student_record]}
    response = client.post("/fusion/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_scored"] == 1
    assert data["scores"][0]["explanation_type"] == "model_explanation"
    assert 0.0 <= data["scores"][0]["eri"] <= 100.0


# ---------------------------------------------------------------------------
# 6. Batch File Upload & Download Tests
# ---------------------------------------------------------------------------


def test_upload_and_predict_csv_bundle(client):
    csv_sample_path = Path("data/legacy/samples/sample_upload.csv")
    assert csv_sample_path.exists(), "Sample upload CSV must exist"

    with open(csv_sample_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/predict/upload",
        files={"file": ("sample_upload.csv", file_bytes, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    job_id = data["job_id"]
    assert job_id.startswith("job_")

    # Verify schema report
    schema = data["schema_report"]
    assert schema["total_rows_ingested"] == 50
    assert schema["columns_matched_count"] > 10

    # Verify cohort summary
    summary = data["cohort_summary"]
    assert summary["total_students"] == 50
    assert 0.0 <= summary["mean_eri"] <= 100.0

    # Test downloads for all 4 formats
    for fmt in ["csv", "xlsx", "json", "html"]:
        res_dl = client.get(f"/download/{job_id}.{fmt}")
        assert res_dl.status_code == 200
        assert len(res_dl.content) > 0


def test_upload_empty_file_rejected(client):
    response = client.post(
        "/predict/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_unsupported_file_extension(client):
    response = client.post(
        "/predict/upload",
        files={"file": ("document.pdf", b"%PDF-1.4...", "application/pdf")},
    )
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 7. Privacy & PII Scrubbing in Upload
# ---------------------------------------------------------------------------


def test_upload_with_pii_strictly_stripped(client):
    # CSV containing forbidden direct identifiers: name, nic, phone, email
    dirty_csv = (
        "full_name,student_nic,contact_phone,email_address,household_income_bracket,financial_stress\n"
        'Kasun Perera,982341234V,0771234567,kasun@sliit.lk,"Below 50,000",4\n'
        'Amali Silva,995431234V,0719876543,amali@sliit.lk,"150,000 - 250,000",1\n'
    )
    response = client.post(
        "/predict/upload",
        files={"file": ("survey_pii.csv", dirty_csv.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    dropped = data["schema_report"]["dropped_pii_columns"]
    assert any("name" in col for col in dropped)
    assert any("nic" in col for col in dropped)
    assert any("phone" in col for col in dropped)
    assert any("email" in col for col in dropped)

    # Scored sample must contain pseudonymous participant IDs, never the real names
    for student in data["scored_students_sample"]:
        assert "Kasun" not in student["participant_id"]
        assert "Amali" not in student["participant_id"]
        assert student["participant_id"].startswith("P-")


# ---------------------------------------------------------------------------
# 8. Retraining & Background Job Status Tests
# ---------------------------------------------------------------------------


def test_trigger_retraining_and_check_job_status(client):
    payload = {
        "outcome_column": "student_attrition",
        "confirm": False,
        "dataset_path": "data/synthetic/survey_raw.csv",
    }
    response = client.post("/retrain", json=payload)
    assert response.status_code == 200
    data = response.json()
    job_id = data["job_id"]
    assert job_id.startswith("retrain_")
    assert data["status"] in ["queued", "running", "completed"]

    # Query status endpoint
    res_status = client.get(f"/jobs/{job_id}")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["job_id"] == job_id
    assert status_data["job_type"] == "retraining"


def test_job_not_found(client):
    response = client.get("/jobs/retrain_non_existent_12345")
    assert response.status_code == 404


def test_rate_limit_enforced_after_burst(client):
    import time
    from src.api.main import IP_REQUEST_LOGS, RATE_LIMIT_PER_MINUTE

    # Pre-populate bucket with RATE_LIMIT_PER_MINUTE timestamps for testclient
    IP_REQUEST_LOGS["testclient"] = [time.time()] * (RATE_LIMIT_PER_MINUTE + 5)

    # Next request should be rate limited
    res = client.get("/health")
    assert res.status_code == 429
    assert "rate limit" in res.json()["detail"].lower()

    # Reset bucket
    IP_REQUEST_LOGS["testclient"] = []


def test_payload_too_large_rejected(client):
    # Simulate oversized header content-length
    res = client.post(
        "/predict/upload",
        files={"file": ("large.csv", b"a" * 100, "text/csv")},
        headers={"content-length": str(100 * 1024 * 1024)},  # 100 MB declared
    )
    assert res.status_code == 413
    assert "exceeds maximum" in res.json()["detail"].lower()
