"""Production FastAPI Service for the Environmental Risk Intelligence (ERI) Framework.

Exposes RESTful endpoints for:
- System health and pipeline validation metadata
- Model zoo benchmark leaderboards, metrics, and training histories
- SHAP explainability (global, per-student narratives, cross-fold stability)
- Causal inference (DAG, sIPW/AIPW effects, refutations, conditional Causal Forest)
- Batch upload prediction & report bundle generation
- Standardized group Fusion Framework score integration
- Guarded model retraining jobs
"""

from __future__ import annotations

import os
import tempfile
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import asyncio
from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import uuid

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import pandas as pd

from src.api.schemas import (
    AllSplitsResponse,
    CausalDiagnosticsResponse,
    CausalExposureSummary,
    CausalHeterogeneityResponse,
    CausalSummaryResponse,
    CohortSummary,
    DagEdge,
    DagNode,
    DagResponse,
    FusionScoreItem,
    FusionScoreRequest,
    FusionScoreResponse,
    HealthResponse,
    JobStatusResponse,
    LeaderboardResponse,
    MetaResponse,
    ModelHistoryResponse,
    ModelMetricsResponse,
    PredictionItem,
    RetrainRequest,
    SchemaReport,
    ShapGlobalFeature,
    ShapGlobalResponse,
    ShapStabilityResponse,
    ShapStudentResponse,
    UploadPredictResponse,
)
from src.causal.dag import (
    CONFOUNDERS,
    EXPOSURES,
    MEDIATORS,
    OUTCOME,
    build_causal_dag,
)
from src.cleaning.cleaner import EnvironmentalFinancialCleaner
from src.inference.export import export_inference_bundle
from src.inference.ingestion import align_dataframe_schema
from src.inference.predict import ProductionInferenceEngine
from src.utils.logging_config import setup_logger
from src.utils.paths import ensure_dir, get_artifacts_dir

logger = setup_logger("api_main")

# Service Start Time
START_TIME = time.time()
DATA_BANNER = "SYNTHETIC - pipeline validation only"
ARTIFACTS_DIR = get_artifacts_dir()

# ---------------------------------------------------------------------------
# In-Memory Job Registry for Async Retraining and Batch Prediction
# ---------------------------------------------------------------------------
job_registry: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Rate Limiting State (In-Memory IP Bucket)
# ---------------------------------------------------------------------------
IP_REQUEST_LOGS: Dict[str, List[float]] = {}
RATE_LIMIT_PER_MINUTE = 200
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB limit


# ---------------------------------------------------------------------------
# Custom Middlewares: Structured Logging, Upload Cap & Rate Limiter
# ---------------------------------------------------------------------------

class SecurityAndLoggingMiddleware(BaseHTTPMiddleware):
    """Logs requests without PII, checks payload bounds, and enforces rate limits."""

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        method = request.method
        request_id = str(uuid.uuid4())[:8]

        # 1. Enforce simple sliding window rate limiting
        now = time.time()
        timestamps = IP_REQUEST_LOGS.get(client_ip, [])
        # retain timestamps within the last 60 seconds
        timestamps = [t for t in timestamps if now - t < 60.0]
        if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
            logger.warning(
                "Rate limit exceeded for IP: %s (limit: %d/min)",
                client_ip,
                RATE_LIMIT_PER_MINUTE,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Maximum 200 requests per minute."},
            )
        timestamps.append(now)
        IP_REQUEST_LOGS[client_ip] = timestamps

        # 2. Check content-length header for file uploads
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES:
            logger.warning(
                "Upload payload too large (%s bytes) from IP: %s",
                content_length,
                client_ip,
            )
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": f"File size exceeds maximum allowable limit of {MAX_UPLOAD_BYTES // (1024*1024)}MB."},
            )

        # 3. Log request arrival (Strict Rule: NEVER log request bodies or PII)
        start_ts = time.time()
        logger.info(
            "[%s] START %s %s from IP %s",
            request_id,
            method,
            path,
            client_ip,
        )

        try:
            response = await call_next(request)
            elapsed_ms = (time.time() - start_ts) * 1000.0
            logger.info(
                "[%s] COMPLETED %s %s -> Status %d (%.2f ms)",
                request_id,
                method,
                path,
                response.status_code,
                elapsed_ms,
            )
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as exc:
            elapsed_ms = (time.time() - start_ts) * 1000.0
            logger.error(
                "[%s] FAILED %s %s with exception: %s (%.2f ms)",
                request_id,
                method,
                path,
                str(exc),
                elapsed_ms,
            )
            raise exc


# ---------------------------------------------------------------------------
# FastAPI Application Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SLIIT CDAP ERI - Environmental Risk Intelligence API",
    description=(
        "Production RESTful Microservice for the Environmental & Financial Causal Risk Intelligence "
        "Framework (Project KND-CDAP-IT-05). Provides endpoints for model benchmarking, explainability, "
        "applied causal inference, batch prediction, and Fusion Framework scoring integration."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",  # permissive during development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SecurityAndLoggingMiddleware)


# ---------------------------------------------------------------------------
# Helper Loaders
# ---------------------------------------------------------------------------

def _load_json_artifact(relative_path: str) -> Any:
    full_path = ARTIFACTS_DIR / relative_path
    if not full_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact not found: {relative_path}",
        )
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to read JSON artifact %s: %s", relative_path, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading artifact {relative_path}",
        )


# ---------------------------------------------------------------------------
# 1. Health & Metadata Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """Return system liveness, uptime, and service status."""
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - START_TIME, 2),
        version="1.0.0",
        environment="development",
    )


@app.get("/meta", response_model=MetaResponse, tags=["System"])
def get_metadata() -> MetaResponse:
    """Return pipeline metadata, active model version, and mandatory synthetic data banner."""
    selection = _load_json_artifact("leaderboard/selection_decision.json")
    manifest = _load_json_artifact("manifest.json")
    run_id = manifest.get("run_id", selection.get("run_id", "run_prod_01"))
    timestamp = manifest.get("timestamp", selection.get("timestamp", "2026-10-25T10:00:00Z"))
    return MetaResponse(
        model_name=selection.get("selected_model", "logistic_l2"),
        model_version="1.0.0",
        data_banner=DATA_BANNER,
        is_calibrated=True,
        calibration_method="isotonic",
        training_date="2026-10-25",
        framework_component="Environmental & Financial Causal Risk Intelligence Framework",
        student_id_format="Pseudonymous (P-XXXXX / STU_XXXX)",
        run_id=run_id,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# 2. Leaderboard & Model Inspection Endpoints
# ---------------------------------------------------------------------------

@app.get("/leaderboard", response_model=LeaderboardResponse, tags=["Models"])
def get_leaderboard(track: str = Query("C", pattern="^(C|R)$")) -> LeaderboardResponse:
    """Get the full model zoo benchmark leaderboard, category winners, and parsimonious selection."""
    if track == "C":
        lb_data = _load_json_artifact("leaderboard/leaderboard_C.json")
    else:
        # Fallback to general or R leaderboard if present
        r_path = ARTIFACTS_DIR / "leaderboard" / "leaderboard_R.json"
        if r_path.exists():
            lb_data = _load_json_artifact("leaderboard/leaderboard_R.json")
        else:
            lb_data = _load_json_artifact("leaderboard/leaderboard_C.json")

    selection = _load_json_artifact("leaderboard/selection_decision.json")
    manifest = _load_json_artifact("manifest.json")

    # Load category winners strictly from single source of truth (frontend_data/category_winners.json or qualified models)
    cat_winners_file = ARTIFACTS_DIR / "frontend_data" / "category_winners.json"
    category_winners: Dict[str, Any] = {}
    if cat_winners_file.exists():
        category_winners = _load_json_artifact("frontend_data/category_winners.json")
    else:
        if isinstance(lb_data, list):
            for entry in lb_data:
                if entry.get("qualified", True):
                    cat = entry.get("category", "General")
                    if cat not in category_winners:
                        category_winners[cat] = entry

    run_id = manifest.get("run_id", selection.get("run_id"))
    timestamp = manifest.get("timestamp", selection.get("timestamp"))

    return LeaderboardResponse(
        track=track,
        total_models=len(lb_data) if isinstance(lb_data, list) else 0,
        selected_model=selection.get("selected_model", "logistic_l2"),
        selection_decision=selection,
        category_winners=category_winners,
        leaderboard=lb_data if isinstance(lb_data, list) else [],
        run_id=run_id,
        timestamp=timestamp,
        split_used="validation",
    )


@app.get("/models/{name}/history", response_model=ModelHistoryResponse, tags=["Models"])
def get_model_history(name: str) -> ModelHistoryResponse:
    """Retrieve iterative or per-epoch training/validation loss and metric curves."""
    candidates = [
        f"history/{name}_C_history.json",
        f"history/{name}_R_history.json",
        f"history/{name}_history.json",
    ]
    history_file = None
    for c in candidates:
        if (ARTIFACTS_DIR / c).exists():
            history_file = c
            break

    if not history_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"History curve data not found for model: {name}",
        )

    raw_history = _load_json_artifact(history_file)
    epochs = raw_history.get("epoch", raw_history.get("iteration", []))
    metrics = {k: v for k, v in raw_history.items() if k not in ["epoch", "iteration"]}

    return ModelHistoryResponse(
        model=name,
        epochs_or_iterations=epochs,
        metrics=metrics,
    )


@app.get("/models/{name}/metrics", response_model=ModelMetricsResponse, tags=["Models"])
def get_model_metrics(name: str) -> ModelMetricsResponse:
    """Retrieve multi-split metrics (train/val/test) and associated diagnostic curves (ROC, PR, Calibration)."""
    candidates = [
        (f"metrics/{name}_C_metrics.json", "C"),
        (f"metrics/{name}_R_metrics.json", "R"),
    ]
    metrics_file = None
    track = "C"
    for mf, tr in candidates:
        if (ARTIFACTS_DIR / mf).exists():
            metrics_file = mf
            track = tr
            break

    if not metrics_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metrics not found for model: {name}",
        )

    splits_data = _load_json_artifact(metrics_file)
    if isinstance(splits_data, dict):
        splits_list = [splits_data]
    else:
        splits_list = splits_data

    # Gather curve data if present
    curves: Dict[str, Any] = {}
    roc_path = ARTIFACTS_DIR / f"curves/{name}_{track}_roc_curve.json"
    if roc_path.exists():
        with open(roc_path, "r", encoding="utf-8") as f:
            curves["roc"] = json.load(f)

    pr_path = ARTIFACTS_DIR / f"curves/{name}_{track}_pr_curve.json"
    if pr_path.exists():
        with open(pr_path, "r", encoding="utf-8") as f:
            curves["pr"] = json.load(f)

    calib_path = ARTIFACTS_DIR / f"curves/{name}_{track}_calibration_curve.json"
    if calib_path.exists():
        with open(calib_path, "r", encoding="utf-8") as f:
            curves["calibration"] = json.load(f)

    return ModelMetricsResponse(
        model=name,
        track=track,
        splits=splits_list,
        curves=curves,
    )


@app.get("/metrics/all-splits", response_model=AllSplitsResponse, tags=["Models"])
def get_all_splits_metrics() -> AllSplitsResponse:
    """Retrieve comprehensive multi-split evaluation metrics table across all zoo models."""
    csv_path = ARTIFACTS_DIR / "metrics" / "all_splits_metrics.csv"
    if not csv_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="All-splits metrics table not found.",
        )
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    return AllSplitsResponse(records_count=len(records), data=records)


# ---------------------------------------------------------------------------
# 3. Explainability (SHAP) Endpoints
# ---------------------------------------------------------------------------

@app.get("/shap/global", response_model=ShapGlobalResponse, tags=["Explainability"])
def get_shap_global() -> ShapGlobalResponse:
    """Retrieve global SHAP feature importance rankings and base value."""
    base_val_data = _load_json_artifact("shap/base_value.json")
    importance_data = _load_json_artifact("shap/global_importance.json")

    base_val = base_val_data.get("base_value", 0.0) if isinstance(base_val_data, dict) else float(base_val_data)

    items = []
    if isinstance(importance_data, list):
        for idx, item in enumerate(importance_data):
            items.append(
                ShapGlobalFeature(
                    feature=item.get("feature", "unknown"),
                    mean_abs_shap=item.get("mean_abs_shap", 0.0),
                    rank=item.get("rank", idx + 1),
                )
            )

    return ShapGlobalResponse(
        base_value=base_val,
        feature_count=len(items),
        global_importance=items,
    )


@app.get("/shap/student/{student_id}", response_model=ShapStudentResponse, tags=["Explainability"])
def get_shap_student(student_id: str) -> ShapStudentResponse:
    """Retrieve individual student waterfall explanation, top drivers, and narrative."""
    narratives = _load_json_artifact("shap/student_narratives.json")
    matched = None
    if isinstance(narratives, list):
        for entry in narratives:
            if entry.get("participant_id") == student_id:
                matched = entry
                break
    elif isinstance(narratives, dict):
        matched = narratives.get(student_id)

    if not matched:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student ID '{student_id}' not found in explainability records.",
        )

    return ShapStudentResponse(
        participant_id=matched["participant_id"],
        eri_score=matched.get("eri_score", 0.0),
        risk_band=matched.get("risk_band", "Unknown"),
        narrative=matched.get("narrative", ""),
        main_drivers=matched.get("main_drivers", []),
        protective_factors=matched.get("protective_factors", []),
        disclaimer=matched.get("disclaimer", "This explains the model's score. It does not prove these factors caused the outcome."),
    )


@app.get("/shap/stability", response_model=ShapStabilityResponse, tags=["Explainability"])
def get_shap_stability() -> ShapStabilityResponse:
    """Retrieve cross-validation fold stability metrics for SHAP feature rankings."""
    stability = _load_json_artifact("shap/stability.json")
    return ShapStabilityResponse(
        top3_overlap=stability.get("top3_overlap", 0.0),
        top5_overlap=stability.get("top5_overlap", 0.0),
        spearman_rank_correlation=stability.get("spearman_rank_correlation", 0.0),
        is_stable=stability.get("is_stable", True),
        unstable_features=stability.get("unstable_features", []),
        interpretation=stability.get("interpretation", "Feature attribution ranking is robust across folds."),
    )


# ---------------------------------------------------------------------------
# 4. Causal Analysis & DAG Endpoints
# ---------------------------------------------------------------------------

@app.get("/causal/summary", response_model=CausalSummaryResponse, tags=["Causal Inference"])
def get_causal_summary() -> CausalSummaryResponse:
    """Retrieve multi-estimator causal effects table (Risk Differences) across all exposures."""
    summary = _load_json_artifact("causal/causal_effects_summary.json")
    exposures_dict = {}
    for exp_name, exp_data in summary.get("exposures", {}).items():
        exposures_dict[exp_name] = CausalExposureSummary(
            exposure=exp_name,
            **exp_data,
        )
    return CausalSummaryResponse(
        metadata=summary.get("metadata", {}),
        exposures=exposures_dict,
    )


@app.get("/causal/diagnostics", response_model=CausalDiagnosticsResponse, tags=["Causal Inference"])
def get_causal_diagnostics() -> CausalDiagnosticsResponse:
    """Retrieve propensity score balance diagnostics (SMD Love Plot) and sensitivity bounds."""
    love_plot = _load_json_artifact("causal/love_plot_data.json")
    summary = _load_json_artifact("causal/causal_effects_summary.json")
    exposures_data = summary.get("exposures", {})
    diagnostics = {
        exp: data.get("diagnostics", {})
        for exp, data in exposures_data.items()
    }

    return CausalDiagnosticsResponse(
        balance_smd=love_plot,
        exposure_diagnostics=diagnostics,
    )


@app.get("/causal/heterogeneity", response_model=CausalHeterogeneityResponse, tags=["Causal Inference"])
def get_causal_heterogeneity() -> CausalHeterogeneityResponse:
    """Retrieve Conditional Average Treatment Effects (CATE) and Causal Forest gate validation status."""
    summary = _load_json_artifact("causal/causal_effects_summary.json")
    forest_data = summary.get("causal_forest", {})

    return CausalHeterogeneityResponse(
        status=forest_data.get("status", "unknown"),
        gate_passed=forest_data.get("gate_passed", False),
        n_samples=forest_data.get("n_samples", 0),
        n_events=forest_data.get("n_events", 0),
        reason=forest_data.get("reason"),
        overall_cate_mean=forest_data.get("overall_cate_mean"),
        subgroups=forest_data.get("subgroups", {}),
    )


@app.get("/dag", response_model=DagResponse, tags=["Causal Inference"])
def get_dag_specification() -> DagResponse:
    """Retrieve the formal Causal DAG structure (nodes, roles, edges) and visualization link."""
    G = build_causal_dag()

    nodes: List[DagNode] = []
    for node, attrs in G.nodes(data=True):
        role = attrs.get("role", "confounder")
        label = node.replace("_", " ").title()
        nodes.append(DagNode(id=node, label=label, role=role))

    edges: List[DagEdge] = []
    for u, v in G.edges():
        edges.append(DagEdge(source=u, target=v))

    return DagResponse(
        nodes=nodes,
        edges=edges,
        exposures=EXPOSURES,
        mediators=MEDIATORS,
        confounders=CONFOUNDERS,
        primary_outcome=OUTCOME,
        image_url="/dag/image",
    )


@app.get("/dag/image", tags=["Causal Inference"])
def get_dag_image() -> FileResponse:
    """Return the high-resolution publication-quality DAG diagram PNG."""
    image_path = Path("reports/figures/dag.png")
    if not image_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="DAG diagram image (reports/figures/dag.png) not found.",
        )
    return FileResponse(image_path, media_type="image/png", filename="causal_dag.png")


# ---------------------------------------------------------------------------
# 5. Production Inference & Batch Upload Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict/upload", response_model=UploadPredictResponse, tags=["Inference"])
async def upload_and_predict(
    file: UploadFile = File(...),
) -> UploadPredictResponse:
    """Upload a survey batch file (.csv, .xlsx, .xls), sanitize PII, score with model, and return bundle."""
    # 1. Validate file extension
    filename = file.filename or "uploaded_data.csv"
    ext = Path(filename).suffix.lower()
    if ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Accepted formats: .csv, .xlsx, .xls",
        )

    # 2. Read bytes into memory
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    try:
        if ext == ".csv":
            df_raw = pd.read_csv(io.BytesIO(content))
        else:
            df_raw = pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        logger.error("Failed to parse uploaded table: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to parse table: {str(exc)}",
        )

    # 3. Run the shared EF cleaner on the original upload for quality findings.
    try:
        clean_result = EnvironmentalFinancialCleaner().clean(
            content, mode="scoring", filename=filename
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Survey cleaning failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Survey cleaning error: {str(exc)}",
        )

    # 4. Pipeline Ingestion & Schema Alignment (Strict PII Drop)
    try:
        df_aligned, report = align_dataframe_schema(df_raw)
    except Exception as exc:
        logger.error("Ingestion pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Data schema ingestion error: {str(exc)}",
        )

    # 5. Score Batch with Production Engine
    engine = ProductionInferenceEngine()
    df_scored, cohort_summary = engine.score_dataframe(df_aligned)

    # 6. Export deliverables bundle
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    out_dir = ARTIFACTS_DIR / "predictions" / job_id
    ensure_dir(out_dir)

    bundle_paths = export_inference_bundle(df_scored, cohort_summary, output_dir=out_dir)

    # Save job record
    job_registry[job_id] = {
        "job_id": job_id,
        "job_type": "batch_inference",
        "status": "completed",
        "progress_percent": 100,
        "message": f"Successfully scored {len(df_scored)} students.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "artifacts_dir": str(out_dir),
        "bundle": {k: str(v) for k, v in bundle_paths.items()},
    }

    # Format schema report
    schema_report = SchemaReport(
        total_rows_ingested=report.get("total_rows_ingested", len(df_aligned)),
        columns_matched_count=report.get("columns_matched_count", 0),
        matched_features=report.get("matched_features", []),
        missing_features_imputed=report.get("missing_features_imputed", []),
        ignored_unknown_columns=report.get("ignored_unknown_columns", []),
        dropped_pii_columns=report.get("dropped_pii_columns", []),
        has_outcome_column=report.get("has_outcome_column", False),
        generated_pseudonyms=True,
        data_quality_report=clean_result.report,
    )

    # Format cohort summary
    std_val = float(df_scored["eri_score"].std()) if len(df_scored) > 1 else 0.0
    risk_dist = {str(k): int(v) for k, v in df_scored["risk_band"].value_counts().items()}
    risk_pct = {k: round(v / len(df_scored) * 100, 2) for k, v in risk_dist.items()}

    summary_obj = CohortSummary(
        total_students=cohort_summary.get("total_students_scored", len(df_scored)),
        mean_eri=cohort_summary.get("mean_eri", float(df_scored["eri_score"].mean())),
        median_eri=cohort_summary.get("median_eri", float(df_scored["eri_score"].median())),
        std_eri=round(std_val, 2),
        risk_distribution=risk_dist,
        risk_percentages=risk_pct,
        top_cohort_drivers=cohort_summary.get("top_cohort_drivers", []),
        causal_summary_status=str(cohort_summary.get("causal_summary", "Not evaluated")),
    )

    # Format sample preview (first 10 records)
    sample_preview: List[PredictionItem] = []
    for _, row in df_scored.head(10).iterrows():
        sample_preview.append(
            PredictionItem(
                participant_id=str(row["participant_id"]),
                eri=float(row["eri_score"]),
                risk_band=str(row["risk_band"]),
                intervention_sla=str(row.get("intervention_sla", "Within 30 days")),
                recommended_action=str(row.get("recommended_action", "Routine monitoring")),
                narrative=str(row.get("narrative", "")),
                data_quality_flags=int(row.get("missing_fields_count", 0)),
                confidence_note=str(row.get("confidence_note", "Standard")),
            )
        )

    download_urls = {
        "csv": f"/download/{job_id}.csv",
        "xlsx": f"/download/{job_id}.xlsx",
        "json": f"/download/{job_id}.json",
        "html": f"/download/{job_id}.html",
    }

    return UploadPredictResponse(
        job_id=job_id,
        status="completed",
        schema_report=schema_report,
        cohort_summary=summary_obj,
        scored_students_sample=sample_preview,
        download_urls=download_urls,
    )


@app.get("/download/{job_id}.{file_format}", tags=["Inference"])
def download_inference_file(job_id: str, file_format: str) -> FileResponse:
    """Download exported inference deliverable by job ID (.csv, .xlsx, .json, .html)."""
    clean_format = file_format.lower()
    if clean_format not in ["csv", "xlsx", "json", "html"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format '{clean_format}'. Options: csv, xlsx, json, html",
        )

    job = job_registry.get(job_id)
    if job and "bundle" in job:
        format_key_map = {
            "csv": "csv",
            "xlsx": "xlsx",
            "json": "fusion_json",
            "html": "html_report",
        }
        target_path_str = job["bundle"].get(format_key_map[clean_format])
        if target_path_str and Path(target_path_str).exists():
            file_path = Path(target_path_str)
        else:
            file_path = ARTIFACTS_DIR / "predictions" / job_id / f"scored_students.{clean_format}"
    else:
        # Check standard predictions folder
        file_path = ARTIFACTS_DIR / "predictions" / job_id / f"scored_students.{clean_format}"
        if not file_path.exists():
            if clean_format == "html":
                file_path = ARTIFACTS_DIR / "predictions" / job_id / "cohort_report.html"
            elif clean_format == "json":
                file_path = ARTIFACTS_DIR / "predictions" / job_id / "fusion_payload.json"

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Requested artifact not found for job {job_id} in format {clean_format}",
        )

    media_type_map = {
        "csv": "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "json": "application/json",
        "html": "text/html",
    }

    return FileResponse(
        file_path,
        media_type=media_type_map[clean_format],
        filename=f"{job_id}_eri_export.{clean_format}",
    )


# ---------------------------------------------------------------------------
# 6. Fusion Framework Integration Endpoint
# ---------------------------------------------------------------------------

@app.post("/fusion/score", response_model=FusionScoreResponse, tags=["Fusion Framework"])
def score_fusion_payload(request: FusionScoreRequest) -> FusionScoreResponse:
    """Score or retrieve calibrated ERI results for the group Fusion Framework."""
    scored_items: List[FusionScoreItem] = []

    # Case A: Explicit student questionnaires provided inline
    if request.students:
        df_in = pd.DataFrame(request.students)
        df_aligned, _ = align_dataframe_schema(df_in)
        engine = ProductionInferenceEngine()
        df_scored, _ = engine.score_dataframe(df_aligned)

        for _, row in df_scored.iterrows():
            top_drivers = []
            if "top_drivers" in row and isinstance(row["top_drivers"], list):
                top_drivers = [str(d.get("feature", d)) if isinstance(d, dict) else str(d) for d in row["top_drivers"]]

            scored_items.append(
                FusionScoreItem(
                    participant_id=str(row["participant_id"]),
                    eri=float(row["eri_score"]),
                    band=str(row["risk_band"]),
                    explanation_type="model_explanation",
                    model_version="1.0.0",
                    top_drivers=top_drivers,
                    narrative=str(row.get("narrative", "")),
                )
            )

    # Case B: Participant IDs provided to query latest scored validation or batch records
    elif request.participant_ids:
        # Load from narratives lookup or latest scored csv
        narratives = _load_json_artifact("shap/student_narratives.json")
        id_lookup = {item["participant_id"]: item for item in narratives}

        for pid in request.participant_ids:
            if pid in id_lookup:
                entry = id_lookup[pid]
                drivers = [d["feature"] for d in entry.get("main_drivers", [])]
                scored_items.append(
                    FusionScoreItem(
                        participant_id=pid,
                        eri=float(entry.get("eri_score", 0.0)),
                        band=str(entry.get("risk_band", "Low")),
                        explanation_type="model_explanation",
                        model_version="1.0.0",
                        top_drivers=drivers,
                        narrative=entry.get("narrative"),
                    )
                )
            else:
                # Default baseline record if ID is not recognized
                scored_items.append(
                    FusionScoreItem(
                        participant_id=pid,
                        eri=0.0,
                        band="Unknown",
                        explanation_type="model_explanation",
                        model_version="1.0.0",
                        top_drivers=[],
                        narrative="Participant ID not found in current scored pool.",
                    )
                )
    else:
        # If neither is provided, return first 50 validation set records
        narratives = _load_json_artifact("shap/student_narratives.json")[:50]
        for entry in narratives:
            drivers = [d["feature"] for d in entry.get("main_drivers", [])]
            scored_items.append(
                FusionScoreItem(
                    participant_id=entry["participant_id"],
                    eri=float(entry.get("eri_score", 0.0)),
                    band=str(entry.get("risk_band", "Low")),
                    explanation_type="model_explanation",
                    model_version="1.0.0",
                    top_drivers=drivers,
                    narrative=entry.get("narrative"),
                )
            )

    return FusionScoreResponse(
        total_scored=len(scored_items),
        explanation_type="model_explanation",
        scores=scored_items,
    )


# ---------------------------------------------------------------------------
# 7. Guarded Retraining & Background Job Status
# ---------------------------------------------------------------------------

async def _run_retraining_worker(job_id: str, req: RetrainRequest):
    """Background async worker for guarded model retraining."""
    job = job_registry[job_id]
    job["status"] = "running"
    job["progress_percent"] = 10
    job["message"] = "Initializing retraining environment and validating outcome column..."

    await asyncio.sleep(0.5)

    try:
        from src.inference.retrain import retrain_pipeline_on_file

        data_file = req.dataset_path or "data/raw/survey_raw_sample.csv"

        job["progress_percent"] = 35
        job["message"] = "Data cleaning and stratified cross-validation split..."
        await asyncio.sleep(0.5)

        job["progress_percent"] = 65
        job["message"] = "Training benchmark model zoo and computing leaderboards..."

        result = retrain_pipeline_on_file(
            file_path=data_file,
            outcome_col=req.outcome_column,
            confirm_overwrite=req.confirm,
        )

        job["progress_percent"] = 100
        job["status"] = "failed" if result.get("status") == "failed" else "completed"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["message"] = result.get(
            "message",
            "Retraining completed successfully."
            if result.get("status") == "success"
            else "Cleaning and validation completed; no model artifacts were updated.",
        )
        job["result"] = result
    except Exception as exc:
        logger.error("Retraining background task %s failed: %s", job_id, exc)
        job["status"] = "failed"
        job["progress_percent"] = 100
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["error"] = str(exc)
        job["message"] = f"Retraining failed: {str(exc)}"


@app.post("/retrain", response_model=JobStatusResponse, tags=["Retraining"])
async def trigger_retraining(
    request: RetrainRequest,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    """Trigger a guarded async retraining job. Requires outcome_column."""
    job_id = f"retrain_{uuid.uuid4().hex[:8]}"

    job_record = {
        "job_id": job_id,
        "job_type": "retraining",
        "status": "queued",
        "progress_percent": 0,
        "message": "Job queued for retraining.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
    }
    job_registry[job_id] = job_record

    # Dispatch to background tasks
    background_tasks.add_task(_run_retraining_worker, job_id, request)

    return JobStatusResponse(**job_record)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
def get_job_status(job_id: str) -> JobStatusResponse:
    """Get the current progress, status, or results of an asynchronous background job."""
    if job_id not in job_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job ID '{job_id}' not found.",
        )
    return JobStatusResponse(**job_registry[job_id])
