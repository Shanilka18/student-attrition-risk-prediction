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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.src.config import config
from backend.src.api.schemas import (
    BatchUploadResponse,
    DataSummaryResponse,
    GlobalExplainResponse,
    HealthResponse,
    LeaderboardResponse,
    MetaResponse,
    ModelDetailResponse,
    ResearchQuestionAnswerResponse,
    SingleStudentPredictionResponse,
    StudentExplainResponse,
    StudentPredictionRequest,
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
    Returns high-level project metadata, champion model details, synthetic data mode,
    registered model counts, and latency SLAs.
    """
    service = InferenceService.get_instance()
    lb = service.get_leaderboard()
    
    # Champion details from Category C
    champ_details = service.get_model_details("catboost") or {}
    seq_champ_details = service.get_model_details("gru") or {}

    from backend.src.models.base import CATEGORIES

    return MetaResponse(
        project_id=config.project_id,
        project_name=config.project_name,
        author="V. S. H. Gunasekara (IT23164680)",
        version=config.version,
        data_mode="SYNTHETIC",
        synthetic_disclaimer=(
            "IMPORTANT: All model training and performance estimates are derived from a realistic "
            "synthetic longitudinal academic dataset. Evaluation on institutional records is pending."
        ),
        champion_model={
            "model_key": "catboost",
            "label": "CatBoost",
            "category": "Category_C",
            "category_label": "Boosting — Engineered Trend Features",
            "role": "Production Champion (TreeSHAP Explainable)",
            "test_roc_auc": 0.81,
            "test_pr_auc": 0.20,
            "test_f2_score": 0.32,
            "lead_time_semesters": 2.0,
            "calibrated": True,
        },
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
        nfr1_latency_sla_seconds=2.0
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
def get_predict_template() -> Response:
    """
    Returns the official upload template CSV containing sample student observations
    formatted to match the long schema.
    """
    template_path = config.schema_dir / "upload_template.csv"
    if not template_path.exists():
        # Fallback to creating sample template
        content = (
            "student_id,semester,gpa,exam_marks,attendance_pct,assignment_delay_days,late_assignments_count\n"
            "STU_DEMO_01,1,3.45,74.5,91.0,0.5,0\n"
            "STU_DEMO_01,2,3.38,71.0,88.5,1.2,1\n"
        )
    else:
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()

    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=academic_observation_template.csv"}
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
async def predict_upload(file: UploadFile = File(...)) -> BatchUploadResponse:
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
    content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
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
