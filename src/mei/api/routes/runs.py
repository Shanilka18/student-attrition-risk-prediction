"""Experiment runs management, active run switching, and comparison endpoints."""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query

from mei.api.models import RunsListResponse, RunSummary, StatusResponse
from mei.api.store import RunStore
from mei.registry.retrain import activate_pipeline_run, compare_pipeline_runs

router = APIRouter(tags=["Runs"])


def get_store() -> RunStore:
    return RunStore.get_instance()


@router.get("/api/runs", response_model=RunsListResponse)
def list_runs(store: RunStore = Depends(get_store)) -> RunsListResponse:
    """List all available pipeline runs with training origins and champion summaries."""
    try:
        active_id = store.get_active_run_id()
        runs_data = store.get_all_runs()
        return RunsListResponse(
            active_run_id=active_id,
            total_runs=len(runs_data),
            runs=[RunSummary(**r) for r in runs_data],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed listing runs: {e}")


@router.post("/api/runs/{id}/activate", response_model=StatusResponse)
def activate_run(
    id: str,
    actor: str = Query(
        "Web UI Administrator", description="User or process activating the model"
    ),
    store: RunStore = Depends(get_store),
) -> StatusResponse:
    """Switch the globally active run and log audit record. Automatically clears all in-memory caches."""
    try:
        activate_pipeline_run(id, actor=actor)
        return store.get_status()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Run '{id}' does not exist on disk"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed activating run '{id}': {e}"
        )


@router.get("/api/runs/compare")
def compare_runs(
    a: str = Query(..., description="First run ID (baseline or active run)"),
    b: str = Query(..., description="Second run ID (candidate run)"),
) -> Dict[str, Any]:
    """Side-by-side run comparison for model governance, rank changes, and performance deltas."""
    try:
        return compare_pipeline_runs(run_id_a=a, run_id_b=b)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed comparing runs: {e}")
