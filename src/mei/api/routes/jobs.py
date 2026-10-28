"""Background asynchronous jobs monitoring and cancellation endpoints."""

from typing import List

from fastapi import APIRouter, HTTPException

from mei.api.jobs import JobManager
from mei.api.models import JobStatusResponse

router = APIRouter(tags=["Jobs"])


def get_job_manager() -> JobManager:
    return JobManager.get_instance()


@router.get("/api/jobs", response_model=List[JobStatusResponse])
def list_jobs() -> List[JobStatusResponse]:
    """List all registered background jobs sorted by creation time."""
    mgr = get_job_manager()
    return mgr.list_jobs()


@router.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Fetch live progress, message, and execution state for a background job."""
    mgr = get_job_manager()
    job = mgr.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Request graceful cancellation of a running background job."""
    mgr = get_job_manager()
    success = mgr.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=404, detail=f"Job '{job_id}' not found or already finished"
        )
    return {"status": "cancelled", "job_id": job_id}
