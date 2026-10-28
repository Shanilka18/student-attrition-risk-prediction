"""Experiment runs management and active run switching endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from mei.api.models import RunsListResponse, RunSummary, StatusResponse
from mei.api.store import RunStore

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
    store: RunStore = Depends(get_store),
) -> StatusResponse:
    """Switch the globally active run. Automatically clears all in-memory caches."""
    try:
        store.set_active_run(id)
        return store.get_status()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Run '{id}' does not exist on disk"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed activating run '{id}': {e}"
        )
