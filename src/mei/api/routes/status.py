"""Status and health endpoints for MEI API."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from mei.api.models import HealthResponse, StatusResponse
from mei.api.store import RunStore

router = APIRouter(tags=["Status"])


def get_store() -> RunStore:
    return RunStore.get_instance()


@router.get("/health", response_model=HealthResponse)
def get_health(store: RunStore = Depends(get_store)) -> HealthResponse:
    """Service health check returning server time and active run identifier."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
        active_run_id=store.get_active_run_id(),
    )


@router.get("/api/status", response_model=StatusResponse)
def get_status(store: RunStore = Depends(get_store)) -> StatusResponse:
    """Get active run status, training data origin, cohort size, and champion essentials."""
    return store.get_status()
