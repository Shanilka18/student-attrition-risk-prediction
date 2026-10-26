# tests/test_api.py
"""Pytest suite for the FastAPI backend of the Psychosocial Intelligence Framework.
Covers upload validation, prediction, leaderboard, retraining, and export endpoints.
"""
import io
import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app (assumes src/api/main.py defines "app")
from src.api.main import app

client = TestClient(app)

# Helper to create a minimal valid questionnaire DataFrame
def make_valid_dataframe(num_rows: int = 5) -> pd.DataFrame:
    import itertools
    _vals = {
        "S1": [3, 2, 4, 5, 1], "S2": [2, 3, 4, 5, 1], "S3": [3, 2, 4, 4, 2],
        "M1": [4, 4, 3, 2, 5], "M2": [4, 5, 3, 2, 5], "M3": [4, 4, 3, 2, 5],
        "B1": [5, 4, 3, 2, 4], "B2": [5, 4, 3, 2, 4], "B3": [5, 4, 3, 2, 4],
        "SE1": [4, 3, 4, 2, 5], "SE2": [4, 3, 4, 2, 5], "SE3": [4, 3, 4, 2, 5],
        "PS1": [5, 4, 3, 2, 4], "PS2": [5, 4, 3, 2, 4], "PS3": [5, 4, 3, 2, 4],
    }
    data: dict = {"study_id": [f"STU{i:04d}" for i in range(num_rows)]}
    for col, pattern in _vals.items():
        data[col] = list(itertools.islice(itertools.cycle(pattern), num_rows))
    return pd.DataFrame(data)

# ---------------------------------------------------------------------------
# 1. Upload endpoint tests
# ---------------------------------------------------------------------------
def test_upload_valid_csv(tmp_path):
    df = make_valid_dataframe()
    csv_path = tmp_path / "valid.csv"
    df.to_csv(csv_path, index=False)
    with open(csv_path, "rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("valid.csv", f, "text/csv")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert "upload_id" in payload
    assert payload["row_count"] == len(df)
    assert payload["has_outcomes"] is False

def test_upload_oversize(tmp_path):
    # Generate a file larger than the configured max_upload_size_mb (default 10 MB)
    big_path = tmp_path / "big.csv"
    # 12 MB of commas
    big_path.write_bytes(b"," * (12 * 1024 * 1024))
    with open(big_path, "rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("big.csv", f, "text/csv")},
        )
    assert response.status_code == 413  # Payload Too Large

def test_upload_unsupported_format(tmp_path):
    txt_path = tmp_path / "file.txt"
    txt_path.write_text("just some text")
    with open(txt_path, "rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("file.txt", f, "text/plain")},
        )
    assert response.status_code == 400

# ---------------------------------------------------------------------------
# 2. Prediction endpoint tests (using previously uploaded data)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def uploaded_id(tmp_path):
    df = make_valid_dataframe()
    csv_path = tmp_path / "predict.csv"
    df.to_csv(csv_path, index=False)
    with open(csv_path, "rb") as f:
        resp = client.post(
            "/api/upload",
            files={"file": ("predict.csv", f, "text/csv")},
        )
    assert resp.status_code == 200
    return resp.json()["upload_id"]

def test_predict_success(uploaded_id):
    request_body = {"upload_id": uploaded_id}
    resp = client.post("/api/predict", json=request_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["upload_id"] == uploaded_id
    assert isinstance(data["students"], list)
    assert len(data["students"]) > 0
    first = data["students"][0]
    assert "pri_score" in first and 0 <= first["pri_score"] <= 100
    assert "risk_category" in first
    assert "disclaimer" in data

def test_predict_invalid_upload():
    resp = client.post("/api/predict", json={"upload_id": "nonexistent"})
    assert resp.status_code == 404

# ---------------------------------------------------------------------------
# 3. Leaderboard endpoint test
# ---------------------------------------------------------------------------
def test_leaderboard():
    resp = client.get("/api/leaderboard")
    assert resp.status_code == 200
    payload = resp.json()
    assert "total_models" in payload
    assert isinstance(payload["entries"], list)

# ---------------------------------------------------------------------------
# 4. Retraining job flow (requires allow_retraining = true in config)
# ---------------------------------------------------------------------------
def test_retrain_flow(tmp_path):
    # Upload a dataset WITH an outcome column
    df = make_valid_dataframe(num_rows=30)
    df["attrition_label"] = [0, 1, 0, 1, 0] * 6
    csv_path = tmp_path / "retrain.csv"
    df.to_csv(csv_path, index=False)
    with open(csv_path, "rb") as f:
        upload_resp = client.post(
            "/api/upload",
            files={"file": ("retrain.csv", f, "text/csv")},
        )
    assert upload_resp.status_code == 200
    uid = upload_resp.json()["upload_id"]

    # Initiate retraining
    resp = client.post("/api/retrain", json={"upload_id": uid, "target_column": "attrition_label"})
    assert resp.status_code == 202
    job = resp.json()
    job_id = job["job_id"]
    # Poll until finished (simple back-off, max 30s)
    for _ in range(15):
        status_resp = client.get(f"/api/jobs/{job_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        if status_data["status"] in ("completed", "failed"):
            break
        time.sleep(2)
    else:
        pytest.fail("Retraining job did not complete in time")
    assert status_data["status"] == "completed", f"Retraining failed: {status_data.get('message')}"
    # Verify leaderboard artifact exists for this job
    leaderboard_path = Path("artifacts/user_dataset") / f"user_leaderboard_{job_id}.json"
    assert leaderboard_path.exists()

# ---------------------------------------------------------------------------
# 5. Export endpoints (CSV and Fusion JSON)
# ---------------------------------------------------------------------------
def test_export_csv(uploaded_id):
    # Run a prediction first to ensure results are stored
    client.post("/api/predict", json={"upload_id": uploaded_id})
    resp = client.get(f"/api/export/{uploaded_id}.csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    # Simple sanity: first line contains header "study_id"
    content = resp.text.splitlines()[0]
    assert "study_id" in content.lower()

def test_export_fusion_json(uploaded_id):
    client.post("/api/predict", json={"upload_id": uploaded_id})
    resp = client.get(f"/api/export/fusion/{uploaded_id}.json")
    assert resp.status_code == 200
    payload = resp.json()
    assert "records" in payload and isinstance(payload["records"], list)
    assert all("study_id" in rec for rec in payload["records"])

# ---------------------------------------------------------------------------
# 6. Health endpoint
# ---------------------------------------------------------------------------
def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    # The API returns "healthy" as the status value
    assert resp.json().get("status") in ("ok", "healthy")
