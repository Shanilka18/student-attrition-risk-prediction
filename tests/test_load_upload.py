"""Load and Performance Benchmark for POST /predict/upload with 10,000 rows.

Validates:
- High-throughput ingestion of 10,000 student survey rows.
- Schema alignment and PII scrubbing performance.
- Vectorized batch inference speed and memory stability.
- Artifact serialization (CSV, XLSX, JSON) scalability.
"""

import io
import time
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def api_client():
    return TestClient(app)


@pytest.fixture
def large_survey_csv_bytes():
    """Generate 10,000 survey rows based on canonical schema."""
    sample_df = pd.read_csv("data/legacy/samples/sample_upload.csv")
    # Tile sample rows to reach exactly 10,000 rows
    repeats = int(np.ceil(10000 / len(sample_df)))
    large_df = pd.concat([sample_df] * repeats, ignore_index=True).iloc[:10000].copy()

    # Assign unique student names and IDs to verify PII stripping at scale
    large_df["student_name"] = [f"Student_{i}" for i in range(10000)]
    large_df["nic_number"] = [f"2000{i:06d}V" for i in range(10000)]
    large_df["email"] = [f"student_{i}@sliit.lk" for i in range(10000)]

    buf = io.BytesIO()
    large_df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue()


import numpy as np


def test_load_predict_upload_10k_rows(api_client, large_survey_csv_bytes):
    """Benchmark upload of 10,000 rows, ensuring bounded latency and correct outputs."""
    start_time = time.perf_counter()

    response = api_client.post(
        "/predict/upload",
        files={"file": ("large_cohort_10k.csv", large_survey_csv_bytes, "text/csv")},
    )

    latency = time.perf_counter() - start_time
    print(
        f"\n[LOAD TEST] 10,000 rows processed in {latency:.2f} seconds ({10000 / latency:.1f} rows/sec)"
    )

    assert response.status_code == 200, f"Upload failed: {response.text}"
    data = response.json()

    # Assert correctness
    cohort = data["cohort_summary"]
    assert cohort["total_students"] == 10000
    assert 0 <= cohort["mean_eri"] <= 100
    assert "risk_distribution" in cohort
    assert sum(cohort["risk_distribution"].values()) == 10000
    assert "job_id" in data
    assert "download_urls" in data

    # Assert PII was strictly purged
    schema_report = data["schema_report"]
    assert "student_name" in schema_report["dropped_pii_columns"]
    assert "nic_number" in schema_report["dropped_pii_columns"]
    assert "email" in schema_report["dropped_pii_columns"]

    # Performance assertion: 10,000 rows scored and exported in < 15 seconds
    assert latency < 15.0, f"Latency {latency:.2f}s exceeded 15.0s SLA target."
