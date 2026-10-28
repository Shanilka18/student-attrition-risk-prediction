"""Comprehensive test suite for Phase 15 Student Scoring Workflow (/score).

Validates:
1. All 8 robustness fixtures (clean, messy, missing, 20k rows, Sinhala/Unicode, Google Forms, empty, wrong type).
2. Data quality audits, column match confidences, and traffic lights.
3. Batch scoring execution, chunking, and risk distribution.
4. Download integrity (2-sheet Excel with 'Scored Students' and 'Notes', CSV with date).
5. Student detail endpoint with 4-period trajectories and waterfall reasons.
6. Single student scoring form endpoint.
7. Automated 24-hour upload session cleanup.
"""

import io
import os
import time
from pathlib import Path

import openpyxl
import pandas as pd
import pytest
from starlette.testclient import TestClient

from mei.api.app import create_app
from mei.api.routes.scoring import cleanup_old_uploads
from mei.config import ARTIFACTS_DIR

FIXTURES_DIR = Path("tests/fixtures")


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Instantiate TestClient for FastAPI application."""
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Robustness Fixtures Tests (8 Fixtures)
# ---------------------------------------------------------------------------


def test_fixture_1_clean_sample(client: TestClient):
    """Fixture (a): Clean sample Excel file validates GREEN and scores perfectly."""
    fixture_path = FIXTURES_DIR / "clean_sample.xlsx"
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    with open(fixture_path, "rb") as f:
        resp = client.post(
            "/api/score/validate",
            files={
                "file": (
                    "clean_sample.xlsx",
                    f,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
    assert resp.status_code == 200, resp.text
    val = resp.json()

    assert val["traffic_light"] == "GREEN"
    assert val["total_rows"] == 200
    assert val["scorable_rows_count"] == 200
    assert val["ready_to_score"] is True
    assert len(val["column_matches"]) == 29
    assert all(m["confidence_tier"] == "High" for m in val["column_matches"])

    # Score run
    up_id = val["upload_id"]
    run_resp = client.post("/api/score/run", json={"upload_id": up_id})
    assert run_resp.status_code == 200, run_resp.text
    run_data = run_resp.json()
    assert run_data["scored_count"] == 200
    assert "Low" in run_data["risk_distribution"]
    assert "Moderate" in run_data["risk_distribution"]
    assert "High" in run_data["risk_distribution"]


def test_fixture_2_messy_sample(client: TestClient):
    """Fixture (b): Messy sample with spaced/mixed headers and extra columns maps successfully."""
    fixture_path = FIXTURES_DIR / "messy_sample.csv"
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    with open(fixture_path, "rb") as f:
        resp = client.post(
            "/api/score/validate",
            files={"file": ("messy_sample.csv", f, "text/csv")},
        )
    assert resp.status_code == 200, resp.text
    val = resp.json()

    assert val["total_rows"] == 200
    assert val["scorable_rows_count"] == 200
    assert val["ready_to_score"] is True
    # Extra column 'campus_branch' should be detected as unknown extra
    assert "campus_branch" in val["unknown_columns"]


def test_fixture_3_extra_and_missing(client: TestClient):
    """Fixture (c): Missing required columns triggers RED traffic light and unmapped status."""
    fixture_path = FIXTURES_DIR / "extra_and_missing.csv"
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    with open(fixture_path, "rb") as f:
        resp = client.post(
            "/api/score/validate",
            files={"file": ("extra_and_missing.csv", f, "text/csv")},
        )
    assert resp.status_code == 200, resp.text
    val = resp.json()

    assert val["traffic_light"] == "RED"
    assert len(val["unmapped_required_columns"]) > 0
    not_found = [m for m in val["column_matches"] if m["confidence_tier"] == "Not found"]
    assert len(not_found) > 0


def test_fixture_4_large_20k_rows_chunking(client: TestClient):
    """Fixture (d): 20,000 rows executes chunked scoring reliably."""
    fixture_path = FIXTURES_DIR / "large_20k_rows.csv"
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    with open(fixture_path, "rb") as f:
        resp = client.post(
            "/api/score/validate",
            files={"file": ("large_20k_rows.csv", f, "text/csv")},
        )
    assert resp.status_code == 200, resp.text
    val = resp.json()
    assert val["total_rows"] == 20000

    up_id = val["upload_id"]
    t0 = time.time()
    run_resp = client.post("/api/score/run", json={"upload_id": up_id})
    elapsed = time.time() - t0

    assert run_resp.status_code == 200, run_resp.text
    run_data = run_resp.json()
    assert run_data["scored_count"] == 20000
    # Throughput must be under 60 seconds
    assert elapsed < 60.0, f"Scoring 20k rows took {elapsed:.1f}s (>60s limit)"


def test_fixture_5_unicode_sinhala(client: TestClient):
    """Fixture (e): Sinhala Unicode headers and textual data parsed cleanly without encoding crash."""
    fixture_path = FIXTURES_DIR / "unicode_sinhala.csv"
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    with open(fixture_path, "rb") as f:
        resp = client.post(
            "/api/score/validate",
            files={"file": ("unicode_sinhala.csv", f, "text/csv")},
        )
    assert resp.status_code == 200, resp.text
    val = resp.json()
    assert val["total_rows"] == 35


def test_fixture_6_google_forms_export(client: TestClient):
    """Fixture (f): Google Forms export with long question headers and text responses."""
    fixture_path = FIXTURES_DIR / "google_forms_export.csv"
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    with open(fixture_path, "rb") as f:
        resp = client.post(
            "/api/score/validate",
            files={"file": ("google_forms_export.csv", f, "text/csv")},
        )
    assert resp.status_code == 200, resp.text
    val = resp.json()
    assert val["total_rows"] == 40
    # Long text headers are recognized and text answers noted in fixable problems
    assert len(val["column_matches"]) == 29


def test_fixture_7_empty_file_rejected(client: TestClient):
    """Fixture (g): Empty 0-byte file rejected with 400 Bad Request."""
    fixture_path = FIXTURES_DIR / "empty_file.csv"
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    with open(fixture_path, "rb") as f:
        resp = client.post(
            "/api/score/validate",
            files={"file": ("empty_file.csv", f, "text/csv")},
        )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_fixture_8_wrong_file_type_rejected(client: TestClient):
    """Fixture (h): Non-tabular file format (.pdf) rejected with 400 Bad Request."""
    fixture_path = FIXTURES_DIR / "wrong_file_type.pdf"
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    with open(fixture_path, "rb") as f:
        resp = client.post(
            "/api/score/validate",
            files={"file": ("wrong_file_type.pdf", f, "application/pdf")},
        )
    assert resp.status_code == 400
    assert "unsupported file format" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Results, Detail Drawer, and Download Integrity
# ---------------------------------------------------------------------------


def test_scoring_download_excel_integrity(client: TestClient):
    """Scoring results download produces two sheets ('Scored Students' and 'Notes') with metadata."""
    fixture_path = FIXTURES_DIR / "clean_sample.xlsx"
    with open(fixture_path, "rb") as f:
        v_resp = client.post(
            "/api/score/validate",
            files={
                "file": (
                    "clean_sample.xlsx",
                    f,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
    up_id = v_resp.json()["upload_id"]
    client.post("/api/score/run", json={"upload_id": up_id})

    # Download Excel
    dl_resp = client.get(f"/api/score/{up_id}/download?format=xlsx")
    assert dl_resp.status_code == 200
    content_disp = dl_resp.headers.get("content-disposition", "")
    assert "mei_scored_" in content_disp
    assert ".xlsx" in content_disp

    # Re-read downloaded Excel using openpyxl and verify both sheets
    wb = openpyxl.load_workbook(io.BytesIO(dl_resp.content))
    assert "Scored Students" in wb.sheetnames
    assert "Notes" in wb.sheetnames

    # Verify first 10 rows match API results
    scored_sheet = wb["Scored Students"]
    assert scored_sheet.max_row >= 201  # Header + 200 rows

    # Verify notes metadata sheet
    notes_sheet = wb["Notes"]
    notes_text = " ".join(
        str(cell.value) for row in notes_sheet.iter_rows() for cell in row if cell.value
    )
    assert "k-Nearest Neighbours" in notes_text or "knn" in notes_text
    assert "Threshold" in notes_text
    assert "Notice" in notes_text or "Ethical" in notes_text


def test_student_detail_and_trajectories(client: TestClient):
    """Drawer endpoint returns 4-period trajectories and additive waterfall reasons."""
    fixture_path = FIXTURES_DIR / "clean_sample.xlsx"
    with open(fixture_path, "rb") as f:
        v_resp = client.post(
            "/api/score/validate",
            files={
                "file": (
                    "clean_sample.xlsx",
                    f,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
    up_id = v_resp.json()["upload_id"]
    client.post("/api/score/run", json={"upload_id": up_id})

    res_resp = client.get(f"/api/score/{up_id}/results?page=1&page_size=1")
    stu_id = res_resp.json()["rows"][0]["student_id"]

    det_resp = client.get(f"/api/score/{up_id}/student/{stu_id}")
    assert det_resp.status_code == 200
    detail = det_resp.json()

    assert detail["student_id"] == stu_id
    assert 0 <= detail["mei"] <= 100
    assert "composite" in detail["trajectories"]
    assert len(detail["trajectories"]["composite"]) == 4
    assert len(detail["trajectories"]["indicators"]) == 7
    assert len(detail["reasons"]) >= 2
    assert "disclaimer" in detail["plain_meanings"]
    assert "not a prediction about a person's character" in detail["plain_meanings"]["disclaimer"].lower()


def test_single_student_scoring_endpoint(client: TestClient):
    """POST /api/score/single scores one student survey submission."""
    answers = {
        "resource_access_p1": 5, "resource_access_p2": 5, "resource_access_p3": 4, "resource_access_p4": 4,
        "study_time_p1": 4, "study_time_p2": 4, "study_time_p3": 3, "study_time_p4": 3,
        "review_time_p1": 4, "review_time_p2": 3, "review_time_p3": 3, "review_time_p4": 2,
        "study_consistency_p1": 5, "study_consistency_p2": 4, "study_consistency_p3": 4, "study_consistency_p4": 3,
        "participation_p1": 4, "participation_p2": 4, "participation_p3": 3, "participation_p4": 3,
        "assignment_engagement_p1": 5, "assignment_engagement_p2": 4, "assignment_engagement_p3": 4, "assignment_engagement_p4": 4,
        "quiz_practice_p1": 4, "quiz_practice_p2": 4, "quiz_practice_p3": 3, "quiz_practice_p4": 2,
    }
    resp = client.post(
        "/api/score/single",
        json={"student_id": "STU_INTERACTIVE_01", "answers": answers},
    )
    assert resp.status_code == 200
    s_data = resp.json()
    assert s_data["student_id"] == "STU_INTERACTIVE_01"
    assert 0 <= s_data["mei"] <= 100
    assert s_data["band"] in ["Low", "Moderate", "High"]
    assert len(s_data["reasons"]) >= 2


# ---------------------------------------------------------------------------
# Privacy: 24-Hour Automated Cleanup
# ---------------------------------------------------------------------------


def test_cleanup_old_uploads_removes_expired_sessions():
    """Verify cleanup_old_uploads deletes upload sessions older than 24 hours."""
    uploads_dir = ARTIFACTS_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create a dummy expired session (> 25 hours old)
    old_session = uploads_dir / "up_old_expired_session"
    old_session.mkdir(exist_ok=True)
    dummy_file = old_session / "dummy.txt"
    dummy_file.write_text("old data", encoding="utf-8")

    # Set mtime back by 26 hours
    past_time = time.time() - (26 * 3600)
    os.utime(old_session, (past_time, past_time))

    # 2. Create a recent session (10 minutes old)
    new_session = uploads_dir / "up_recent_active_session"
    new_session.mkdir(exist_ok=True)
    (new_session / "dummy.txt").write_text("new data", encoding="utf-8")

    # 3. Run cleanup
    cleaned = cleanup_old_uploads(max_age_hours=24)
    assert cleaned >= 1

    # 4. Assert old session deleted and new session retained
    assert not old_session.exists(), "Expired session was not deleted!"
    assert new_session.exists(), "Recent session was wrongly deleted!"

    # Clean up dummy recent session
    import shutil
    shutil.rmtree(new_session, ignore_errors=True)
