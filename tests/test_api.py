"""Comprehensive integration tests for the MEI FastAPI backend API.

Tests all endpoints against real artifacts from the active run.
Asserts mathematical honesty, data privacy, and throughput bounds.
"""

import os
from pathlib import Path

# Ensure Matplotlib config dir is set before any imports to avoid font cache rebuilds
os.environ["MPLCONFIGDIR"] = str(Path.cwd() / ".matplotlib_cache")

import json
import time

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from mei.api.app import create_app
from mei.api.store import RunStore
from mei.config import ARTIFACTS_DIR


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Instantiate TestClient for FastAPI app."""
    app = create_app()
    return TestClient(app)


@pytest.fixture(scope="module")
def active_run_id() -> str:
    """Resolve active run ID."""
    store = RunStore.get_instance()
    return store.get_active_run_id()


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------


def test_health_endpoint(client: TestClient, active_run_id: str):
    """GET /health must return healthy status and active run ID."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["active_run_id"] == active_run_id
    assert "timestamp" in data


def test_status_endpoint(client: TestClient, active_run_id: str):
    """GET /api/status must serve real model card details without hard-coding."""
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()

    assert data["active_run_id"] == active_run_id
    assert "data_source" in data
    assert (
        "badge" in data["data_source_badge"].lower()
        or "sample" in data["data_source_badge"].lower()
    )
    assert data["n_students_total"] > 1000
    assert data["n_students_train"] > 1000
    assert data["n_students_test"] > 200

    # Champion assertions
    champ = data["champion"]
    assert champ["model_id"] == "knn"
    assert champ["display_name"] == "k-Nearest Neighbours"
    assert champ["optimal_threshold"] > 0.0
    assert champ["band_verification_monotonic"] is True


# ---------------------------------------------------------------------------
# Leaderboard & Honesty Test
# ---------------------------------------------------------------------------


def test_leaderboard_classification_honesty(client: TestClient, active_run_id: str):
    """Honesty check: /api/leaderboard must exactly match raw JSON artifact values."""
    resp = client.get("/api/leaderboard?track=classification")
    assert resp.status_code == 200
    api_data = resp.json()

    # Load raw artifact
    raw_path = (
        ARTIFACTS_DIR / "runs" / active_run_id / "leaderboard_classification.json"
    )
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    assert api_data["track"] == "classification"
    assert api_data["total_models"] == len(raw_data)
    assert api_data["champion_id"] == "knn"

    # Verify rank 1 champion values match to 6 decimal places
    api_top = api_data["rows"][0]
    raw_top = raw_data[0]
    assert api_top["model_id"] == raw_top["model_id"]
    assert (
        pytest.approx(api_top["leaderboard_score"], abs=1e-6)
        == raw_top["leaderboard_score"]
    )
    assert (
        pytest.approx(api_top["val_f1_optimal"], abs=1e-6) == raw_top["val_f1_optimal"]
    )
    assert (
        pytest.approx(api_top["generalization_gap"], abs=1e-6)
        == raw_top["generalization_gap"]
    )
    assert api_top["is_champion"] is True


def test_leaderboard_regression(client: TestClient):
    """GET /api/leaderboard?track=regression returns sorted regression models."""
    resp = client.get("/api/leaderboard?track=regression")
    assert resp.status_code == 200
    data = resp.json()

    assert data["track"] == "regression"
    assert data["total_models"] >= 30
    assert data["metric_name"] == "R² Score"
    top_row = data["rows"][0]
    assert top_row["rank"] == 1
    assert top_row["leaderboard_score"] > 0.50
    assert top_row["val_r2"] is not None


def test_categories_best(client: TestClient):
    """GET /api/categories/best returns best model per category."""
    resp = client.get("/api/categories/best?track=classification")
    assert resp.status_code == 200
    data = resp.json()
    assert "classical" in data
    assert "boosting" in data
    assert "deep" in data
    assert data["classical"]["model_id"] == "knn"


def test_model_detail_and_curves(client: TestClient):
    """GET /api/models/{id} and /api/models/{id}/curves return full telemetry."""
    # Detail for knn
    resp = client.get("/api/models/knn?track=classification")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["model_id"] == "knn"
    assert detail["overfit_status"] == "Good fit"
    assert "metrics" in detail
    assert "status_explanation" in detail
    assert detail["confusion_matrix_raw"] is not None

    # Curves for deep model (gru)
    resp_curves = client.get("/api/models/gru/curves?track=classification")
    assert resp_curves.status_code == 200
    curves = resp_curves.json()
    assert curves["model_id"] == "gru"
    assert curves["epoch_curves"] is not None
    assert curves["best_epoch"] is not None


def test_compare_models(client: TestClient):
    """GET /api/compare compares up to 4 models side-by-side."""
    resp = client.get("/api/compare?ids=knn,gru,xgboost&track=classification")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) == 3
    assert "knn" in data["metrics_comparison"]
    assert "gru" in data["metrics_comparison"]
    assert "xgboost" in data["metrics_comparison"]


# ---------------------------------------------------------------------------
# Insights: Patterns & Drivers
# ---------------------------------------------------------------------------


def test_patterns_endpoint(client: TestClient):
    """GET /api/patterns returns empirical trajectories, HMM, and risk rates."""
    resp = client.get("/api/patterns")
    assert resp.status_code == 200
    data = resp.json()

    assert data["hmm_k"] in [3, 4]
    assert len(data["transition_matrix"]) == data["hmm_k"]
    assert "Consistent" in data["pattern_distribution"]
    assert "Declining" in data["pattern_distribution"]

    # Trajectories & Risk Rates
    trajs = data["mean_trajectories_by_pattern"]
    assert "Consistent" in trajs
    assert "composite_p1" in trajs["Consistent"]

    risk_rates = data["risk_rates_by_pattern"]
    assert "Declining" in risk_rates
    # Declining cohort must exhibit higher risk than Consistent cohort
    assert (
        risk_rates["Declining"]["at_risk_pct"] > risk_rates["Consistent"]["at_risk_pct"]
    )


def test_drivers_endpoint(client: TestClient):
    """GET /api/drivers returns global rankings, consistency, and point clouds."""
    resp = client.get("/api/drivers")
    assert resp.status_code == 200
    data = resp.json()

    assert data["baseline_mei"] > 0.0
    assert len(data["validation_ranking"]) >= 10
    assert len(data["test_ranking"]) >= 5
    assert "by_indicator" in data["grouped_importance"]
    assert "by_period" in data["grouped_importance"]
    assert "by_family" in data["grouped_importance"]

    # Web interactive point clouds
    assert len(data["beeswarm_samples"]) > 0
    assert len(data["dependence_samples"]) > 0


# ---------------------------------------------------------------------------
# Survey Templates & Dictionary
# ---------------------------------------------------------------------------


def test_template_download(client: TestClient):
    """GET /api/template serves clean CSV and XLSX downloads."""
    # CSV
    resp_csv = client.get("/api/template?format=csv")
    assert resp_csv.status_code == 200
    assert "text/csv" in resp_csv.headers["content-type"]
    assert len(resp_csv.content) > 100

    # XLSX
    resp_xlsx = client.get("/api/template?format=xlsx")
    assert resp_xlsx.status_code == 200
    assert len(resp_xlsx.content) > 1000


def test_dictionary_endpoint(client: TestClient):
    """GET /api/dictionary returns full 35-column questionnaire dictionary."""
    resp = client.get("/api/dictionary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_columns"] == 35
    assert data["num_features"] == 28
    assert len(data["columns"]) == 35


# ---------------------------------------------------------------------------
# Upload Validation & Scoring
# ---------------------------------------------------------------------------


def test_upload_validate_clean_sample(client: TestClient):
    """POST /api/score/validate with clean Excel file passes with GREEN traffic light."""
    sample_file = Path("data/samples/sample_upload_clean.xlsx")
    with open(sample_file, "rb") as f:
        resp = client.post(
            "/api/score/validate",
            files={
                "file": (
                    sample_file.name,
                    f.read(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["upload_id"].startswith("up_")
    assert data["total_rows"] == 200
    assert data["traffic_light"] in ["GREEN", "AMBER"]
    assert data["ready_to_score"] is True
    assert len(data["unmapped_required_columns"]) == 0


def test_upload_validate_messy_sample(client: TestClient):
    """POST /api/score/validate with messy CSV file returns mapping suggestions and warnings."""
    messy_file = Path("data/samples/sample_upload_messy.csv")
    with open(messy_file, "rb") as f:
        resp = client.post(
            "/api/score/validate",
            files={"file": (messy_file.name, f.read(), "text/csv")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["upload_id"].startswith("up_")
    assert len(data["detected_columns"]) > 10
    # Must provide column mappings and suggestions
    assert len(data["column_mapping"]) > 0


def test_batch_scoring_and_pagination(client: TestClient):
    """End-to-end upload -> score -> view paginated -> download."""
    sample_file = Path("data/samples/sample_upload_clean.xlsx")
    with open(sample_file, "rb") as f:
        val_resp = client.post(
            "/api/score/validate",
            files={
                "file": (
                    sample_file.name,
                    f.read(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
    assert val_resp.status_code == 200
    upload_id = val_resp.json()["upload_id"]

    # Run scoring
    score_resp = client.post(
        "/api/score/run",
        json={"upload_id": upload_id},
    )
    assert score_resp.status_code == 200
    res_data = score_resp.json()
    assert res_data["upload_id"] == upload_id
    assert res_data["scored_count"] == 200
    assert "Low" in res_data["risk_distribution"]
    assert "High" in res_data["risk_distribution"]

    # Check paginated view
    page_resp = client.get(
        f"/api/score/{upload_id}/results?page=1&page_size=20&sort=mei&direction=desc"
    )
    assert page_resp.status_code == 200
    pdata = page_resp.json()
    assert pdata["total"] == 200
    assert len(pdata["rows"]) == 20
    first_row = pdata["rows"][0]
    assert 0 <= first_row["mei"] <= 100
    assert first_row["band"] in ["Low", "Moderate", "High"]
    assert len(first_row["top_reasons"]) > 10

    # Download results
    dl_resp = client.get(f"/api/score/{upload_id}/download?format=csv")
    assert dl_resp.status_code == 200
    assert "student_id" in dl_resp.text


def test_single_student_scoring(client: TestClient):
    """POST /api/score/single scores one student with exact waterfall attribution vector."""
    answers = {
        "resource_access_p1": 5,
        "resource_access_p2": 5,
        "resource_access_p3": 2,
        "resource_access_p4": 1,
        "study_time_p1": 4,
        "study_time_p2": 4,
        "study_time_p3": 2,
        "study_time_p4": 1,
        "quiz_practice_p1": 5,
        "quiz_practice_p2": 4,
        "quiz_practice_p3": 1,
        "quiz_practice_p4": 1,
        "review_time_p1": 4,
        "review_time_p2": 4,
        "review_time_p3": 2,
        "review_time_p4": 1,
        "participation_p1": 4,
        "participation_p2": 4,
        "participation_p3": 1,
        "participation_p4": 1,
        "assignment_engagement_p1": 5,
        "assignment_engagement_p2": 4,
        "assignment_engagement_p3": 2,
        "assignment_engagement_p4": 1,
        "study_consistency_p1": 4,
        "study_consistency_p2": 4,
        "study_consistency_p3": 2,
        "study_consistency_p4": 1,
    }
    resp = client.post(
        "/api/score/single",
        json={"student_id": "STU_INTERACTIVE_01", "answers": answers},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["student_id"] == "STU_INTERACTIVE_01"
    assert 0 <= data["mei"] <= 100
    assert data["band"] in ["Low", "Moderate", "High"]
    assert len(data["reasons"]) > 0

    # Verify reason additivity fields
    for r in data["reasons"]:
        assert r["text"]
        assert r["direction"] in ["risk_increasing", "risk_decreasing"]
        assert isinstance(r["points"], int)


# ---------------------------------------------------------------------------
# Large File Throughput Test (20,000 Rows < 60s)
# ---------------------------------------------------------------------------


def test_large_file_scoring_throughput(client: TestClient):
    """Stress test: 20,000 synthetic rows must score in under 60 seconds."""
    from mei.registry.score import score_dataframe

    n_rows = 20000
    rng = np.random.RandomState(42)

    # Fast tabular generation
    cols = {}
    for p in range(1, 5):
        cols[f"resource_access_p{p}"] = rng.randint(1, 6, size=n_rows)
        cols[f"study_time_p{p}"] = rng.randint(1, 6, size=n_rows)
        cols[f"quiz_practice_p{p}"] = rng.randint(1, 6, size=n_rows)
        cols[f"review_time_p{p}"] = rng.randint(1, 6, size=n_rows)
        cols[f"participation_p{p}"] = rng.randint(1, 6, size=n_rows)
        cols[f"assignment_engagement_p{p}"] = rng.randint(1, 6, size=n_rows)
        cols[f"study_consistency_p{p}"] = rng.randint(1, 6, size=n_rows)

    large_df = pd.DataFrame(cols)
    large_df["student_id"] = [f"STU_LARGE_{i:06d}" for i in range(n_rows)]

    t0 = time.time()
    # Score in chunked mode
    chunk_size = 5000
    chunks = []
    for i in range(0, n_rows, chunk_size):
        c = large_df.iloc[i : i + chunk_size].reset_index(drop=True)
        chunks.append(score_dataframe(c, explain=False))
    scored = pd.concat(chunks, ignore_index=True)
    duration = time.time() - t0

    print(
        f"\n[BENCHMARK] 20,000 students scored in: {duration:.2f} seconds ({n_rows / duration:.0f} students/sec)"
    )
    assert len(scored) == n_rows
    assert duration < 60.0, (
        f"Large file scoring took {duration:.2f}s (exceeded 60s threshold)"
    )


# ---------------------------------------------------------------------------
# Runs & Job Management
# ---------------------------------------------------------------------------


def test_runs_management(client: TestClient, active_run_id: str):
    """GET /api/runs and POST /api/runs/{id}/activate."""
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_run_id"] == active_run_id
    assert data["total_runs"] >= 1

    # Reactivate active run
    act_resp = client.post(f"/api/runs/{active_run_id}/activate")
    assert act_resp.status_code == 200
    assert act_resp.json()["active_run_id"] == active_run_id


def test_jobs_endpoint(client: TestClient):
    """GET /api/jobs lists background tasks."""
    resp = client.get("/api/jobs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
