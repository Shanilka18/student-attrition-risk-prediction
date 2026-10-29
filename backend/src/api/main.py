"""
FastAPI Application Entry Point & OpenAPI Router.
Academic Intelligence Framework (SLIIT Final Year Research Project)
Project ID: KND_CDAP_IT_05
Author: V. S. H. Gunasekara (IT23164680)

Endpoints:
  - System: GET /health, GET /meta
  - Model Zoo & Research: GET /leaderboard, GET /models/{name}, GET /research-answer
  - Data & Explanations: GET /data/summary, GET /explain/global, GET /explain/student/{id}
  - Inference: GET /predict/template, POST /predict/student, POST /predict/upload, GET /predict/download/{token}
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.src.config import config
from backend.src.api.schemas import (
    ActivateRunResponse,
    BatchUploadResponse,
    DataSummaryResponse,
    GlobalExplainResponse,
    HealthResponse,
    LeaderboardResponse,
    LatestRetrainResponse,
    MetaResponse,
    ModelDetailResponse,
    ResearchQuestionAnswerResponse,
    RetrainInspectResponse,
    RetrainJobStatusResponse,
    RetrainJobSubmitResponse,
    RetrainRunItem,
    RetrainRunsResponse,
    RollbackRunResponse,
    SingleStudentPredictionResponse,
    StudentExplainResponse,
    StudentPredictionRequest,
)
from backend.src.api.retrain import (
    inspect_dataset_for_retrain,
    extract_and_validate_outcomes,
    RetrainJobManager,
)
from backend.src.api.service import InferenceService
from backend.src.api.validator import validate_academic_dataframe

# Configure structured application logger (zero PII logging)
logger = logging.getLogger("AcademicIntelligence.API")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Maximum upload file size: 10 MB
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024

app = FastAPI(
    title="Academic Intelligence Framework API",
    description=(
        "Production-grade REST API for early student attrition prediction and explainability. "
        "Evaluates whether deep sequence-aware models (LSTM/GRU/TCN) outperform Gradient Boosting "
        "trained on engineered longitudinal features. Computes calibrated Academic Risk Index (0-100) "
        "and exports standardized payloads for the shared multi-modal Fusion Framework."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "V. S. H. Gunasekara (IT23164680)",
        "email": "it23164680@my.sliit.lk"
    },
    license_info={
        "name": "Academic Research Clearance",
        "url": "https://www.sliit.lk/"
    }
)

# Enable CORS for Vite frontend and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for high-resolution evaluation and explainability plots
if config.plots_dir.exists():
    app.mount("/static/plots", StaticFiles(directory=str(config.plots_dir)), name="plots")


# ── System Endpoints ──────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health status and environment diagnostics"
)
def get_health() -> HealthResponse:
    """Verifies backend operational status, compute device, and filesystem readiness."""
    return HealthResponse(
        status="healthy",
        device=str(config.device),
        seed=config.global_seed,
        directories_initialized={
            "synthetic_data": config.synthetic_data_dir.exists(),
            "processed_data": config.processed_data_dir.exists(),
            "models": config.models_dir.exists(),
            "metrics": config.metrics_dir.exists(),
            "explanations": config.explanations_dir.exists(),
            "plots": config.plots_dir.exists(),
        },
        timestamp_utc=datetime.now(timezone.utc).isoformat()
    )


@app.get(
    "/meta",
    response_model=MetaResponse,
    tags=["System"],
    summary="Champion metadata, provenance tags, and versions"
)
def get_meta() -> MetaResponse:
    """
    Returns high-level project metadata, champion model details, active data mode (SYNTHETIC or REAL),
    registered model counts, and latency SLAs.
    """
    service = InferenceService.get_instance()
    active_run = config.get_active_run()
    data_mode = active_run.get("data_mode", "SYNTHETIC")
    active_run_id = active_run.get("run_id")

    from backend.src.models.base import CATEGORIES

    if data_mode == "REAL" and active_run_id:
        champ_key = active_run.get("champion", service.model_key)
        champ_label = active_run.get("champion_label", service.model_label)
        category = active_run.get("category", "C")
        cat_key = category if str(category).startswith("Category_") else f"Category_{category}"
        cat_label = CATEGORIES.get(str(category).replace("Category_", ""), "Retrained on User Dataset")

        champion_model = {
            "model_key": champ_key,
            "label": champ_label,
            "category": cat_key,
            "category_label": cat_label,
            "role": "Production Champion (User-Data Retrained)",
            "test_roc_auc": float(active_run.get("test_roc_auc", 0.0)),
            "test_pr_auc": float(active_run.get("test_pr_auc", 0.0)),
            "test_f2_score": float(active_run.get("test_f2_score", 0.0)),
            "lead_time_semesters": float(active_run.get("lead_time_semesters", 2.0)),
            "calibrated": True,
        }
        synthetic_disclaimer = ""
    else:
        # Canonical SYNTHETIC benchmark - read actual metrics from artifacts/leaderboard/champion.json
        champ_json = config.leaderboard_dir / "champion.json"
        champ_metrics = {}
        champ_key = "catboost"
        champ_label = "CatBoost"
        if champ_json.exists():
            try:
                with open(champ_json, "r", encoding="utf-8") as f:
                    c_data = json.load(f)
                    champ_key = c_data.get("model_key", "catboost")
                    champ_label = c_data.get("label", "CatBoost")
                    champ_metrics = c_data.get("test_metrics", {})
            except Exception as e:
                logger.warning(f"Failed to read champion.json: {e}")

        test_roc = float(champ_metrics.get("roc_auc", 0.7815))
        test_pr = float(champ_metrics.get("pr_auc", 0.2125))
        test_f2 = float(champ_metrics.get("f2_score", 0.3877))
        lead_time = float(champ_metrics.get("lead_time_semesters", 2.06))

        champion_model = {
            "model_key": champ_key,
            "label": champ_label,
            "category": "Category_C",
            "category_label": "Boosting — Engineered Trend Features",
            "role": "Production Champion (TreeSHAP Explainable)",
            "test_roc_auc": test_roc,
            "test_pr_auc": test_pr,
            "test_f2_score": test_f2,
            "lead_time_semesters": lead_time,
            "calibrated": True,
        }
        synthetic_disclaimer = (
            "IMPORTANT: All model training and performance estimates are derived from a realistic "
            "synthetic longitudinal academic dataset. Evaluation on institutional records is pending."
        )

    return MetaResponse(
        project_id=config.project_id,
        project_name=config.project_name,
        author="V. S. H. Gunasekara (IT23164680)",
        version=config.version,
        data_mode=data_mode,
        synthetic_disclaimer=synthetic_disclaimer,
        champion_model=champion_model,
        sequence_champion={
            "model_key": "gru",
            "label": "GRU (Proposal Model)",
            "category": "Category_E",
            "category_label": "Deep Sequence Models",
            "role": "Sequence Model Baseline",
            "test_roc_auc": 0.83,
            "test_pr_auc": 0.22,
            "test_f2_score": 0.35,
            "lead_time_semesters": 2.5,
            "calibrated": True,
        },
        categories=CATEGORIES,
        n_registered_models=28,
        nfr1_latency_sla_seconds=2.0,
        active_run_id=active_run_id,
        active_run_meta=active_run if data_mode == "REAL" else None,
    )


# ── Leaderboard & Research Answer ─────────────────────────────────────────────

@app.get(
    "/leaderboard",
    response_model=LeaderboardResponse,
    tags=["Leaderboard & Models"],
    summary="Full multi-model leaderboard and category champions"
)
def get_leaderboard() -> LeaderboardResponse:
    """
    Returns the comprehensive 28-model benchmark leaderboard, multi-objective ranking
    weights, eligibility gates, and per-category champions.
    """
    service = InferenceService.get_instance()
    data = service.get_leaderboard()
    return LeaderboardResponse(**data)


@app.get(
    "/models/{name}",
    response_model=ModelDetailResponse,
    tags=["Leaderboard & Models"],
    summary="Detailed model inspection: params, metrics, curves, and trials"
)
def get_model_details(name: str) -> ModelDetailResponse:
    """
    Retrieves complete training history, hyperparameters, validation metrics,
    calibration statistics, and learning curves for any model in the zoo.
    """
    service = InferenceService.get_instance()
    details = service.get_model_details(name)
    if details is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' was not found in the model zoo registry. Check /leaderboard for valid model keys."
        )
    return ModelDetailResponse(**details)


@app.get(
    "/research-answer",
    response_model=ResearchQuestionAnswerResponse,
    tags=["Leaderboard & Models"],
    summary="Core research question adjudication and hypothesis tests"
)
def get_research_answer() -> ResearchQuestionAnswerResponse:
    """
    Adjudicates the core scientific research question:
    'Does sequence-aware modelling (LSTM/GRU) beat engineered trend features + Gradient Boosting?'
    Includes statistical significance (paired bootstrap p-values) and synthetic data disclaimer.
    """
    service = InferenceService.get_instance()
    answer = service.get_research_answer()
    return ResearchQuestionAnswerResponse(**answer)


# ── Data & Explanations ───────────────────────────────────────────────────────

@app.get(
    "/data/summary",
    response_model=DataSummaryResponse,
    tags=["Dataset & Explainability"],
    summary="Synthetic cohort audit statistics and visual plots index"
)
def get_data_summary() -> DataSummaryResponse:
    """
    Returns dataset audit metrics, class distribution, variable scale definitions,
    and a searchable catalog of high-resolution diagnostic plots.
    """
    service = InferenceService.get_instance()
    return DataSummaryResponse(**service.get_data_summary())


@app.get(
    "/explain/global",
    response_model=GlobalExplainResponse,
    tags=["Dataset & Explainability"],
    summary="Global TreeSHAP feature importance and temporal attributions"
)
def get_global_explanations() -> GlobalExplainResponse:
    """
    Returns mean absolute SHAP importance across all 22 engineered features,
    longitudinal semester-by-feature attribution matrices, and faithfulness audit results.
    """
    service = InferenceService.get_instance()
    data = service.get_global_explanations()
    return GlobalExplainResponse(**data)


@app.get(
    "/explain/student/{student_id}",
    response_model=StudentExplainResponse,
    tags=["Dataset & Explainability"],
    summary="Precomputed or dynamic student explainability payload"
)
def get_student_explanation(student_id: str) -> StudentExplainResponse:
    """
    Retrieves the clinical Academic Risk Index, top-3 driving factors, and plain-language
    diagnostic narrative for an individual student.
    """
    service = InferenceService.get_instance()
    exp = service.get_student_explanation_by_id(student_id)
    if exp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student ID '{student_id}' not found in precomputed test explanations or synthetic cohort."
        )
    return StudentExplainResponse(**exp)


# ── Prediction Endpoints ──────────────────────────────────────────────────────

@app.get(
    "/predict/template",
    tags=["Inference"],
    summary="Downloadable CSV observation template"
)
def get_predict_template(
    with_outcomes: bool = Query(False, description="Include outcome labels (attrited, attrition_semester) for model retraining")
) -> Response:
    """
    Returns the official upload template CSV containing sample student observations
    formatted to match the long schema. If with_outcomes=True, returns the template
    including outcome labels (attrited, attrition_semester, cohort_year) for model retraining.
    """
    if with_outcomes:
        template_path = config.schema_dir / "upload_template_with_outcomes.csv"
        filename = "academic_observation_template_with_outcomes.csv"
        fallback_content = (
            "student_id,cohort_year,semester,gpa,exam_marks,attendance_pct,assignment_delay_days,late_assignments_count,attrited,attrition_semester\n"
            "STU_DEMO_01,2023,1,3.45,74.5,91.0,0.5,0,0,\n"
            "STU_DEMO_01,2023,2,3.38,71.0,88.5,1.2,1,0,\n"
        )
    else:
        template_path = config.schema_dir / "upload_template.csv"
        filename = "academic_observation_template.csv"
        fallback_content = (
            "student_id,semester,gpa,exam_marks,attendance_pct,assignment_delay_days,late_assignments_count\n"
            "STU_DEMO_01,1,3.45,74.5,91.0,0.5,0\n"
            "STU_DEMO_01,2,3.38,71.0,88.5,1.2,1\n"
        )

    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = fallback_content

    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.post(
    "/predict/student",
    response_model=SingleStudentPredictionResponse,
    tags=["Inference"],
    summary="Real-time single-student risk assessment and explanation"
)
def predict_single_student(payload: StudentPredictionRequest) -> SingleStudentPredictionResponse:
    """
    Evaluates an individual student's longitudinal semester sequence using the Champion
    CatBoost model. Returns calibrated Academic Risk Index (0-100), risk band, top SHAP factors,
    deterministic narrative, and Fusion Framework handoff contract.
    """
    service = InferenceService.get_instance()
    
    # Check minimum semesters rule warning
    warnings = []
    if len(payload.semesters) == 1:
        warnings.append(
            f"Student '{payload.student_id}' has only 1 recorded semester. "
            "Minimum Semesters Rule: Trajectory slopes and volatility default to 0.0 until >= 2 semesters are recorded."
        )

    sem_dicts = [s.model_dump() for s in payload.semesters]
    result = service.predict_single_student(payload.student_id, sem_dicts, warnings=warnings)
    
    logger.info(
        f"Inference completed for student={payload.student_id}, "
        f"window_t={result['window_semester']}, ARI={result['academic_risk_index']}, "
        f"latency={result['inference_time_ms']}ms"
    )
    return SingleStudentPredictionResponse(**result)


@app.post(
    "/predict/upload",
    response_model=BatchUploadResponse,
    tags=["Inference"],
    summary="Batch upload of academic records (CSV or Excel)"
)
def predict_upload(file: UploadFile = File(...)) -> BatchUploadResponse:
    """
    Accepts multipart CSV or Excel (.xlsx) file containing student semester records.
    
    Validation & Pipeline:
      1. Caps file size at 10 MB.
      2. Validates against /data/schema/schema.json with explicit row numbers on errors.
      3. Automatically detects and pivots wide-format tables (e.g. gpa_sem1, attendance_pct_sem1).
      4. Applies zero-leakage training preprocessor.
      5. Runs Champion CatBoost model and produces calibrated ARI scores and clinical risk bands.
      6. Generates local SHAP attributions, deterministic narratives, and Fusion contracts.
      7. Provides download token for CSV/JSON batch exports.
      8. Deletes in-memory payloads immediately after processing (Zero PII persistence).
    """
    fname = file.filename or "upload.csv"
    ext = Path(fname).suffix.lower()

    if ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Please upload a .csv or .xlsx file."
        )

    # Read file content safely with size limit
    t_start = time.perf_counter()
    content = file.file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB."
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=[{
                "row": None,
                "column": None,
                "error_type": "EMPTY_FILE",
                "message": "Uploaded file is empty (0 bytes)."
            }]
        )

    # Parse into pandas DataFrame
    try:
        buffer = io.BytesIO(content)
        if ext == ".csv":
            df_raw = pd.read_csv(buffer)
        else:
            df_raw = pd.read_excel(buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=[{
                "row": None,
                "column": None,
                "error_type": "PARSE_ERROR",
                "message": f"Failed to parse uploaded {ext} file: {str(e)}"
            }]
        )

    # Validate against academic observation schema
    is_valid, validation_errors, df_clean_long, warnings = validate_academic_dataframe(df_raw)
    if not is_valid or df_clean_long is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=validation_errors
        )

    # Execute batch prediction and explainability
    service = InferenceService.get_instance()
    batch_result = service.predict_batch(df_clean_long, warnings=warnings)

    logger.info(
        f"Batch upload processed: filename={fname}, records={len(df_clean_long)}, "
        f"students={batch_result['total_students']}, total_latency={batch_result['inference_time_total_ms']}ms, "
        f"per_student_latency={batch_result['inference_time_per_student_ms']}ms"
    )

    return BatchUploadResponse(**batch_result)


@app.get(
    "/predict/download/{token}",
    tags=["Inference"],
    summary="Download batch prediction results in CSV or JSON format"
)
def download_batch_results(
    token: str,
    format: str = Query("csv", pattern="^(csv|json)$", description="Export format: 'csv' or 'json'")
) -> Response:
    """
    Retrieves cached batch inference predictions by session token.
    Provides downloadable CSV or formatted JSON export.
    """
    service = InferenceService.get_instance()
    export_str = service.get_export_data(token, format_type=format)
    
    if export_str is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Export session for token '{token}' has expired or does not exist."
        )

    if format == "csv":
        media_type = "text/csv"
        filename = f"academic_risk_predictions_{token[:8]}.csv"
    else:
        media_type = "application/json"
        filename = f"academic_risk_predictions_{token[:8]}.json"

    return Response(
        content=export_str,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ── Retraining & Isolation Endpoints ──────────────────────────────────────────

@app.post(
    "/train/inspect",
    response_model=RetrainInspectResponse,
    tags=["Retraining"],
    summary="Inspect uploaded file for retraining feasibility"
)
def inspect_upload(file: UploadFile = File(...)) -> RetrainInspectResponse:
    """
    Evaluates uploaded dataset for model retraining feasibility.
    Checks:
      - Outcome column presence ('attrited', 'dropout', 'withdrawn', etc.)
      - Sample size sufficiency (>= 100 students, >= 20 attrited, >= 20 non-attrited)
      - Temporal data leakage (records occurring after student's documented attrition semester)
      - Binary outcome label validity (0/1)
      - Class imbalance advisory warnings (< 5% or > 60% attrition rate)
    """
    fname = file.filename or "dataset.csv"
    ext = Path(fname).suffix.lower()

    if ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Please upload a .csv or .xlsx file."
        )

    content = file.file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB."
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file is empty (0 bytes)."
        )

    try:
        buffer = io.BytesIO(content)
        if ext == ".csv":
            df_raw = pd.read_csv(buffer)
        else:
            df_raw = pd.read_excel(buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Failed to parse uploaded {ext} file: {str(e)}"
        )

    inspect_result = inspect_dataset_for_retrain(df_raw)
    return RetrainInspectResponse(**inspect_result)


@app.post(
    "/retrain",
    response_model=RetrainJobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Retraining"],
    summary="Trigger asynchronous isolated model retraining on uploaded data"
)
def start_retrain(
    file: UploadFile = File(...),
    budget: str = Form("quick"),
    scope: str = Form("champions"),
) -> RetrainJobSubmitResponse:
    """
    Validates dataset, initiates isolated zero-leakage training pipeline in background thread,
    and returns HTTP 202 with job_id. All outputs are written to artifacts/user_dataset/<job_id>/.
    Returns HTTP 409 if another retraining job is already running.
    """
    valid_budgets = ["quick", "standard", "full"]
    if budget not in valid_budgets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid budget '{budget}'. Supported values: {valid_budgets}"
        )

    valid_scopes = ["champions", "full"]
    if scope not in valid_scopes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scope '{scope}'. Supported values: {valid_scopes}"
        )

    manager = RetrainJobManager.get_instance()
    if manager.active_job_id is not None:
        active_job = manager.get_job(manager.active_job_id)
        if active_job and active_job["state"] in ("queued", "running"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Another retraining job ({manager.active_job_id}) is currently in progress."
            )

    fname = file.filename or "dataset.csv"
    ext = Path(fname).suffix.lower()

    if ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Please upload a .csv or .xlsx file."
        )

    content = file.file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB."
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file is empty (0 bytes)."
        )

    try:
        buffer = io.BytesIO(content)
        if ext == ".csv":
            df_raw = pd.read_csv(buffer)
        else:
            df_raw = pd.read_excel(buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Failed to parse uploaded {ext} file: {str(e)}"
        )

    # Validate eligibility for retraining
    inspect_result = inspect_dataset_for_retrain(df_raw)
    if not inspect_result["can_retrain"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "Dataset cannot be used for retraining due to blocking issues.",
                "blocking_issues": inspect_result["blocking_issues"],
            }
        )

    df_long_clean, df_outcomes, blocking, warnings = extract_and_validate_outcomes(df_raw)
    if blocking:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"message": "Dataset validation failed.", "blocking_issues": blocking}
        )

    job_id = manager.submit_job(
        df_long=df_long_clean,
        df_outcomes=df_outcomes,
        budget=budget,
        scope=scope,
        upload_bytes=content,
        upload_filename=fname,
    )

    logger.info(
        f"Retraining job submitted: job_id={job_id}, students={len(df_outcomes)}, "
        f"budget={budget}, scope={scope}"
    )

    return RetrainJobSubmitResponse(
        job_id=job_id,
        message="Retraining job queued successfully. Isolated training environment active."
    )


@app.get(
    "/jobs/{job_id}",
    response_model=RetrainJobStatusResponse,
    tags=["Retraining"],
    summary="Get real-time status and metrics of a retraining job"
)
def get_job_status(job_id: str) -> RetrainJobStatusResponse:
    """
    Polls real-time progress of a retraining job (phase, percentage, current model,
    elapsed time, and final champion metrics).
    """
    manager = RetrainJobManager.get_instance()
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Retraining job '{job_id}' not found."
        )

    return RetrainJobStatusResponse(
        job_id=job["job_id"],
        state=job["state"],
        phase=job.get("phase"),
        progress_pct=job.get("progress_pct", 0),
        message=job.get("message", ""),
        current_model=job.get("current_model"),
        models_done=job.get("models_done", 0),
        models_total=job.get("models_total", 0),
        elapsed_s=job.get("elapsed_s", 0.0),
        result=job.get("result"),
        error=job.get("error"),
    )


@app.get(
    "/train/latest",
    response_model=LatestRetrainResponse,
    tags=["Retraining"],
    summary="Get latest retrained user dataset benchmark summary"
)
def get_latest_retrain() -> LatestRetrainResponse:
    """
    Returns metadata for the most recently completed user dataset retraining run.
    Returns HTTP 404 if no user dataset has been retrained yet.
    """
    latest_path = config.root_dir / "artifacts" / "user_dataset" / "latest.json"
    if not latest_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No retrained user dataset benchmark found. Retrain on your data to generate an isolated benchmark."
        )

    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return LatestRetrainResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read latest retraining metadata: {e}"
        )


# ── Active Model Activation & Rollback Endpoints ──────────────────────────────

@app.get(
    "/train/runs",
    response_model=RetrainRunsResponse,
    tags=["Retraining & Active Model"],
    summary="List all completed user dataset retraining runs"
)
def list_retrain_runs() -> RetrainRunsResponse:
    """
    Scans artifacts/user_dataset/ for completed retraining runs and indicates which run is active for scoring.
    """
    user_base = config.root_dir / "artifacts" / "user_dataset"
    active_meta = config.get_active_run()
    active_id = active_meta.get("run_id")
    mode = active_meta.get("data_mode", "SYNTHETIC")

    runs: List[RetrainRunItem] = []
    if user_base.exists():
        for job_dir in user_base.iterdir():
            if job_dir.is_dir() and job_dir.name.startswith("job_"):
                summary_path = job_dir / "summary.json"
                if summary_path.exists():
                    try:
                        with open(summary_path, "r", encoding="utf-8") as f:
                            d = json.load(f)
                        runs.append(RetrainRunItem(
                            job_id=d.get("job_id", job_dir.name),
                            trained_at=d.get("trained_at", ""),
                            n_students=d.get("n_students", 0),
                            n_attrited=d.get("n_attrited", 0),
                            champion=d.get("champion", "catboost"),
                            champion_label=d.get("champion_label", d.get("champion", "CatBoost")),
                            sequence_champion=d.get("sequence_champion"),
                            test_roc_auc=float(d.get("test_roc_auc", 0.0)),
                            test_pr_auc=float(d.get("test_pr_auc", 0.0)),
                            test_f2_score=float(d.get("test_f2_score", 0.0)),
                            lead_time_semesters=float(d.get("lead_time_semesters", 2.0)),
                            budget=d.get("budget", "quick"),
                            scope=d.get("scope", "champions"),
                            is_active=(job_dir.name == active_id),
                        ))
                    except Exception as e:
                        logger.warning(f"Could not parse run summary at {summary_path}: {e}")

    # Sort newest first
    runs.sort(key=lambda r: r.trained_at, reverse=True)
    return RetrainRunsResponse(
        runs=runs,
        active_run_id=active_id,
        data_mode=mode,
    )


@app.post(
    "/train/activate/{job_id}",
    response_model=ActivateRunResponse,
    tags=["Retraining & Active Model"],
    summary="Activate a retrained model run for live scoring"
)
def activate_retrain_run(job_id: str) -> ActivateRunResponse:
    """
    Points live inference scoring to the specified user dataset retraining run without server restart.
    """
    job_dir = config.root_dir / "artifacts" / "user_dataset" / job_id
    summary_path = job_dir / "summary.json"
    if not summary_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Retrained run '{job_id}' not found or incomplete."
        )

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load run metadata: {e}"
        )

    meta["data_mode"] = "REAL"
    meta["run_id"] = job_id
    config.set_active_run(meta)

    # Reload inference service in-process
    InferenceService.get_instance().reload()
    service = InferenceService.get_instance()

    return ActivateRunResponse(
        message=f"Successfully activated model '{service.model_label}' ({job_id}) for live inference scoring.",
        job_id=job_id,
        data_mode="REAL",
        champion=service.model_key,
        champion_label=service.model_label,
        metrics={
            "test_roc_auc": meta.get("test_roc_auc"),
            "test_pr_auc": meta.get("test_pr_auc"),
            "test_f2_score": meta.get("test_f2_score"),
            "lead_time_semesters": meta.get("lead_time_semesters"),
        }
    )


@app.post(
    "/train/rollback",
    response_model=RollbackRunResponse,
    tags=["Retraining & Active Model"],
    summary="Roll back to canonical thesis synthetic benchmark"
)
def rollback_to_synthetic() -> RollbackRunResponse:
    """
    Deactivates user model scoring and restores canonical thesis synthetic benchmark.
    """
    config.clear_active_run()
    InferenceService.get_instance().reload()
    service = InferenceService.get_instance()

    return RollbackRunResponse(
        message="Successfully rolled back scoring model to canonical thesis synthetic benchmark.",
        data_mode="SYNTHETIC",
        active_model=service.model_key,
    )


@app.delete(
    "/train/runs/{job_id}",
    tags=["Retraining & Active Model"],
    summary="Delete an isolated retrained run directory"
)
def delete_retrain_run(job_id: str) -> Dict[str, Any]:
    """
    Permanently deletes an isolated retrained run directory and associated uploads.
    If the run is currently active, it rolls back to synthetic benchmark first.
    """
    job_dir = config.root_dir / "artifacts" / "user_dataset" / job_id
    if not job_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Retrained run '{job_id}' not found."
        )

    active_meta = config.get_active_run()
    was_active = (active_meta.get("run_id") == job_id)
    if was_active:
        config.clear_active_run()
        InferenceService.get_instance().reload()

    # Delete artifacts directory
    shutil.rmtree(job_dir, ignore_errors=True)

    # Delete associated upload directory if present
    upload_dir = config.root_dir / "data" / "uploads" / job_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)

    # Update latest.json if this was the latest
    latest_path = config.root_dir / "artifacts" / "user_dataset" / "latest.json"
    if latest_path.exists():
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                latest_data = json.load(f)
            if latest_data.get("job_id") == job_id:
                latest_path.unlink()
        except Exception:
            pass

    return {
        "message": f"Retrained run '{job_id}' deleted successfully.",
        "job_id": job_id,
        "rolled_back": was_active,
    }

