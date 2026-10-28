"""Retraining pipeline API endpoints for model governance and dataset retraining."""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from mei.api.jobs import JobManager
from mei.config import ARTIFACTS_DIR
from mei.registry.retrain import (
    activate_pipeline_run,
    run_full_pipeline,
    validate_retrain_dataset,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/retrain", tags=["Retraining"])

UPLOADS_DIR = ARTIFACTS_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class RetrainStartRequest(BaseModel):
    """Configuration payload to launch background retraining."""

    upload_id: str = Field(..., description="Uploaded dataset ID from validation step")
    mode: str = Field(
        "quick",
        description="Retraining mode: 'quick' (10-15 min) or 'full' (all models)",
    )
    options: Optional[Dict[str, Any]] = Field(
        None, description="Optional label overrides or configuration"
    )
    seed: int = Field(42, description="Random seed for cross-validation splitting")


class ActivateRunRequest(BaseModel):
    """Payload to activate a specific model run."""

    run_id: str = Field(..., description="Pipeline run ID to activate")
    actor: str = Field(
        "Web UI User", description="User or entity authorizing model activation"
    )


@router.post("/validate")
async def validate_retrain_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Validate uploaded survey response file for complete pipeline retraining.

    Checks:
    - Minimum cohort size (N >= 150)
    - Small cohort rules (150 <= N < 500)
    - Class balance and presence of required 3 outcome questions (A1-A3)
    - Missingness and indicator ranges
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file missing filename")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes)")

    # Verify format
    filename_lower = file.filename.lower()
    if not (
        filename_lower.endswith(".csv") or filename_lower.endswith((".xlsx", ".xls"))
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Only .csv, .xlsx, and .xls are supported.",
        )

    # Persist file for subsequent training run
    upload_id = f"retrain_{uuid.uuid4().hex[:10]}"
    safe_name = Path(file.filename).name
    target_path = UPLOADS_DIR / f"{upload_id}_{safe_name}"
    target_path.write_bytes(content)

    # Validate dataset using small-data guards and cleaning pipeline
    val_res = validate_retrain_dataset(target_path)

    return {
        "upload_id": upload_id,
        "filename": file.filename,
        "total_rows": val_res["total_rows"],
        "usable_rows": val_res["scorable_rows"],
        "class_balance": val_res["class_balance"],
        "missingness": val_res["missingness"],
        "warnings": val_res["warnings"],
        "estimated_time_min": val_res["estimated_time"]["quick_minutes"],
        "traffic_light": val_res["traffic_light"],
        "is_refusal": not val_res["can_retrain"],
        "refusal_reason": val_res.get("refusal_reason"),
        "can_proceed": val_res["can_retrain"],
        "guards": val_res.get("small_data_guard", {}),
    }


@router.post("/start")
def start_retraining(req: RetrainStartRequest) -> Dict[str, Any]:
    """Launch asynchronous background retraining pipeline job."""
    # Find uploaded dataset
    matching_files = list(UPLOADS_DIR.glob(f"{req.upload_id}_*"))
    if not matching_files:
        raise HTTPException(
            status_code=404,
            detail=f"Upload ID '{req.upload_id}' not found or session expired. Please re-upload.",
        )

    dataset_path = matching_files[0]
    mgr = JobManager.get_instance()

    # Check if a retrain job is already running
    for job in mgr.list_jobs():
        if job.job_type == "retrain" and job.status == "running":
            raise HTTPException(
                status_code=409,
                detail=f"A retraining job ({job.job_id}) is currently in progress. Cancel it first or wait for completion.",
            )

    job_id = mgr.submit_job(
        job_type="retrain",
        task_fn=run_full_pipeline,
        dataset_path=dataset_path,
        mode=req.mode,
        label_config=req.options,
        seed=req.seed,
    )

    logger.info(
        "Launched retraining job %s (mode=%s, seed=%d)", job_id, req.mode, req.seed
    )
    return {
        "job_id": job_id,
        "status": "running",
        "mode": req.mode,
        "dataset_name": dataset_path.name,
    }


@router.get("/{job_id}/status")
def get_retrain_status(job_id: str) -> Dict[str, Any]:
    """Poll live progress, active stage, and intermediate artifacts for a retraining job."""
    mgr = JobManager.get_instance()
    job = mgr.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job_dict = job.model_dump()
    result = job_dict.get("result") or {}

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "result": result,
        "error": job.error,
        "active_stage": result.get("active_stage", job.message),
        "run_id": result.get("run_id"),
        "champion_name": result.get("champion_name"),
    }


@router.post("/{job_id}/cancel")
def cancel_retraining(job_id: str) -> Dict[str, Any]:
    """Signal cooperative cancellation to a running retraining pipeline job."""
    mgr = JobManager.get_instance()
    success = mgr.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found or cannot be cancelled in its current state.",
        )
    return {"job_id": job_id, "status": "cancelling"}


@router.post("/{job_id}/activate")
def activate_retrain_job_result(job_id: str) -> Dict[str, Any]:
    """Activate the model produced by a completed retraining job as current champion."""
    mgr = JobManager.get_instance()
    job = mgr.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job.status != "finished" or not job.result or "run_id" not in job.result:
        raise HTTPException(
            status_code=400,
            detail=f"Job '{job_id}' is not in a finished state with valid run output (status: {job.status}).",
        )

    target_run_id = job.result["run_id"]
    try:
        activation_res = activate_pipeline_run(
            target_run_id, actor="Web UI Retraining Wizard"
        )
        return activation_res
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed activating run '{target_run_id}': {exc}"
        )


@router.post("/activate")
def activate_run_endpoint(req: ActivateRunRequest) -> Dict[str, Any]:
    """Explicitly activate any retrained run ID and record audit entry."""
    try:
        activation_res = activate_pipeline_run(req.run_id, actor=req.actor)
        return activation_res
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed activating run: {exc}")
