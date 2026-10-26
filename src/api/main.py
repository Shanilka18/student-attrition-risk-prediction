"""FastAPI server exposing endpoints for prediction, explainability, models, and Fusion Framework export.

Ethics & Research Governance Disclaimer:
The Psychosocial Risk Index (PRI) is an academic decision-support indicator developed for
educational research at SLIIT Kandy. It is not a clinical psychological evaluation or psychiatric diagnosis.
"""

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.data.schema import (
    REQUIRED_COLUMNS,
    read_survey_file,
    validate_dataframe,
    suggest_column_mappings,
    apply_column_mappings,
)
from src.features.build_features import (
    FACTOR_ITEMS,
    FACTOR_NAMES,
    compute_factor_scores,
    compute_pri_score,
    categorize_risk,
)
from src.api.schemas import (
    QuestionnaireInput,
    PRIResponse,
    FactorScores,
    FusionExportContract,
    MetaResponse,
    LeaderboardResponse,
    LeaderboardEntry,
    ModelDetailsResponse,
    ChampionResponse,
    ChampionTrackDetail,
    ReliabilityResponse,
    ReliabilityFactorDetail,
    EDAResponse,
    GlobalShapResponse,
    UploadResponse,
    PredictRequest,
    CohortPredictionResponse,
    RetrainRequest,
    JobStatusResponse,
    FusionBatchExportResponse,
)
from src.api.storage import storage
from src.api.inference import run_batch_inference
from src.api.retrain import execute_retraining_job
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("api.main")
cfg = load_config()

ETHICS_DISCLAIMER = cfg.get("ethics", {}).get(
    "disclaimer",
    "RESEARCH NOTICE: The Psychosocial Risk Index (PRI) is an academic decision-support indicator developed for educational research at SLIIT Kandy. It is not a clinical psychological evaluation or psychiatric diagnosis.",
)

app = FastAPI(
    title=cfg.get("api", {}).get("title", "Psychosocial Intelligence Framework API"),
    version=cfg.get("api", {}).get("version", "1.0.0"),
    description="Inference, explainability, and integration service for the Psychosocial Risk Index (PRI) component.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS setup for Vite dev server and production UI
cors_origins = cfg.get("api", {}).get(
    "cors_origins",
    ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount artifacts folder to serve diagnostic plots and SHAP charts
artifacts_dir = Path("artifacts")
if artifacts_dir.exists():
    app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")



# =====================================================================
# 1. Health & Metadata Endpoints
# =====================================================================

@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Verify service health, project version, and legal disclaimer."""
    return {
        "status": "healthy",
        "project": cfg["project"]["name"],
        "version": cfg["project"]["version"],
        "disclaimer": ETHICS_DISCLAIMER,
    }


@app.get("/api/meta", response_model=MetaResponse, tags=["System"])
def get_metadata() -> MetaResponse:
    """Retrieve framework metadata, active champion models, data hashes, and research flags."""
    # Compute hash of processed train data if available
    train_path = Path("data/processed/train.csv")
    if train_path.exists():
        data_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()[:16]
    else:
        data_hash = "synthetic_v1_seed42"

    return MetaResponse(
        project=cfg["project"]["name"],
        model_version=cfg["project"]["version"],
        training_date_utc="2026-10-18T07:17:26Z",
        data_hash=data_hash,
        is_synthetic=True,
        disclaimer=ETHICS_DISCLAIMER,
        instrument_items=[item for items in FACTOR_ITEMS.values() for item in items],
        latent_factors=FACTOR_NAMES,
        champions={"regression": "ridge", "classification": "hist_gradient_boosting"},
        status="operational",
    )


# =====================================================================
# 2. Leaderboards & Model Diagnostics Endpoints
# =====================================================================

@app.get("/api/leaderboard", response_model=LeaderboardResponse, tags=["Benchmarks"])
def get_leaderboards(
    track: Optional[str] = Query(None, description="Filter by track: 'regression' or 'classification'"),
    category: Optional[str] = Query(None, description="Filter by model category (e.g. 'Boosting', 'Linear/Shallow')"),
    feature_set: Optional[str] = Query(None, description="Filter by feature set: 'factor_scores' or 'raw_items'"),
) -> LeaderboardResponse:
    """Retrieve comprehensive benchmark leaderboards across shallow, boosting, deep, and meta models."""
    leaderboard_dir = Path("artifacts/leaderboards")
    entries: List[LeaderboardEntry] = []

    tracks_to_query = [track] if track in ["regression", "classification"] else ["regression", "classification"]

    for trk in tracks_to_query:
        json_path = leaderboard_dir / f"{trk}_leaderboard.json"
        csv_path = leaderboard_dir / f"{trk}_leaderboard.csv"

        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # JSON is insertion-ordered (dummy first), NOT metric-sorted.
                # Filter first, then sort by the correct metric.
                is_regression = trk == "regression"
                track_entries = [
                    item for item in data
                    if (not category or item.get("category", "").lower() == category.lower())
                    and (not feature_set or item.get("feature_set", "") == feature_set)
                ]
                # Regression: sort by RMSE ascending (lower = better)
                # Classification: sort by Macro-F1 descending (higher = better)
                track_entries.sort(
                    key=lambda x: x.get("validation_metrics", {}).get("primary", 0.0),
                    reverse=not is_regression,
                )
                champion_name = "ridge" if is_regression else "hist_gradient_boosting"

                for rank_idx, item in enumerate(track_entries):
                    vm = item.get("validation_metrics", {})
                    # For display: show R² (regression) or Macro-F1 (classification)
                    if is_regression:
                        display_primary = vm.get("r2", vm.get("primary", 0.0))
                        display_secondary = vm.get("rmse")
                    else:
                        display_primary = vm.get("macro_f1", vm.get("primary", 0.0))
                        display_secondary = vm.get("mcc")

                    entry = LeaderboardEntry(
                        rank=rank_idx + 1,
                        model_name=item.get("model_name", "unknown"),
                        category=item.get("category", "General"),
                        track=trk,
                        feature_set=item.get("feature_set", "factor_scores"),
                        primary_metric=round(float(display_primary), 4),
                        secondary_metric=round(float(display_secondary), 4) if display_secondary is not None else None,
                        overfitting_gap=item.get("overfitting_gap"),
                        ece=item.get("ece"),
                        inference_latency_ms=item.get("inference_latency_ms"),
                        model_size_kb=item.get("model_size_kb"),
                        composite_rank=item.get("composite_rank"),
                        is_champion=(item.get("model_name") == champion_name and rank_idx == 0),
                    )
                    entries.append(entry)
            except Exception as e:
                logger.warning("Error loading %s: %s", json_path, e)


        elif csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                for idx, row in df.iterrows():
                    entry = LeaderboardEntry(
                        rank=idx + 1,
                        model_name=str(row.get("model_name", "unknown")),
                        category=str(row.get("category", "General")),
                        track=trk,
                        feature_set=str(row.get("feature_set", "factor_scores")),
                        primary_metric=float(row.get("primary", 0.0)),
                        secondary_metric=float(row.get("tiebreak", 0.0)) if "tiebreak" in row else None,
                        overfitting_gap=float(row.get("overfitting_gap", 0.0)) if "overfitting_gap" in row else None,
                        is_champion=(idx == 0),
                    )
                    entries.append(entry)
            except Exception as e:
                logger.warning("Error loading %s: %s", csv_path, e)

    return LeaderboardResponse(track=track, total_models=len(entries), entries=entries)


@app.get("/api/models/{name}", response_model=ModelDetailsResponse, tags=["Benchmarks"])
def get_model_details(
    name: str,
    track: Optional[str] = Query(None, description="Optional track: 'regression' or 'classification'"),
) -> ModelDetailsResponse:
    """Retrieve detailed hyperparameters, cross-validation metrics, and diagnostic plot paths for a model."""
    models_dir = Path("artifacts/models")
    tracks_to_search = [track] if track in ["regression", "classification"] else ["regression", "classification"]

    meta_payload = None
    resolved_track = "regression"

    for trk in tracks_to_search:
        meta_file = models_dir / trk / f"{name}_metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta_payload = json.load(f)
                resolved_track = trk
                break

    if not meta_payload:
        raise HTTPException(status_code=404, detail=f"Model '{name}' metadata not found in trained artifacts.")

    # Locate diagnostic plots
    plot_dir = Path("artifacts/plots")
    plots = []
    for p in plot_dir.glob(f"*{name}*.png"):
        plots.append(str(p))

    is_champ = (name == "ridge" and resolved_track == "regression") or (name == "hist_gradient_boosting" and resolved_track == "classification")

    return ModelDetailsResponse(
        name=meta_payload.get("model_name", name),
        track=resolved_track,
        category=meta_payload.get("category", "Unknown"),
        feature_set=meta_payload.get("feature_set", "factor_scores"),
        is_champion=is_champ,
        best_params=meta_payload.get("best_params", {}),
        train_metrics=meta_payload.get("train_metrics"),
        validation_metrics=meta_payload.get("validation_metrics"),
        test_metrics=meta_payload.get("test_set_metrics"),
        overfitting_gap=meta_payload.get("overfitting_gap"),
        inference_latency_ms=meta_payload.get("inference_latency_ms"),
        model_size_kb=meta_payload.get("model_size_kb"),
        diagnostic_plots=plots,
    )


# =====================================================================
# 3. Champion & Scientific Evaluation Endpoints
# =====================================================================

@app.get("/api/champion", response_model=ChampionResponse, tags=["Scientific Evaluation"])
def get_champion_models() -> ChampionResponse:
    """Retrieve selected champions for continuous PRI regression and high-risk classification."""
    champ_json = Path("artifacts/leaderboards/champion.json")
    if not champ_json.exists():
        raise HTTPException(status_code=404, detail="Champion metadata file artifacts/leaderboards/champion.json not found.")

    with open(champ_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    reg_data = data.get("regression", {})
    clf_data = data.get("classification", {})

    reg_bundle_path = Path("artifacts/models/final/regression_champion_bundle.json")
    clf_bundle_path = Path("artifacts/models/final/classification_champion_bundle.json")

    reg_test_metrics = {}
    clf_test_metrics = {}
    reg_params = {}
    clf_params = {}

    if reg_bundle_path.exists():
        with open(reg_bundle_path, "r", encoding="utf-8") as f:
            rb = json.load(f)
            reg_test_metrics = rb.get("test_set_metrics", {})
            reg_params = rb.get("best_params", {})

    if clf_bundle_path.exists():
        with open(clf_bundle_path, "r", encoding="utf-8") as f:
            cb = json.load(f)
            clf_test_metrics = cb.get("test_set_metrics", {})
            clf_params = cb.get("best_params", {})

    reg_track = ChampionTrackDetail(
        model_name=reg_data.get("champion", "ridge"),
        category=reg_data.get("champion_category", "Linear/Shallow"),
        feature_set="factor_scores",
        selection_metric=reg_data.get("selection_metric", "val_r2"),
        primary_metric_value=reg_data.get("primary_metric_value", 0.9908),
        overfitting_gap=reg_data.get("overfitting_gap", 0.5178),
        test_metrics=reg_test_metrics,
        justification=reg_data.get("justification", "Ridge selected based on R2 and SHAP compatibility."),
        best_params=reg_params,
    )

    clf_track = ChampionTrackDetail(
        model_name=clf_data.get("champion", "hist_gradient_boosting"),
        category=clf_data.get("champion_category", "Boosting"),
        feature_set="factor_scores",
        selection_metric=clf_data.get("selection_metric", "val_macro_f1"),
        primary_metric_value=clf_data.get("primary_metric_value", 0.9314),
        overfitting_gap=clf_data.get("overfitting_gap", 0.05),
        high_risk_recall=clf_data.get("high_risk_recall", 0.8919),
        ece=clf_data.get("ece", 0.04),
        test_metrics=clf_test_metrics,
        justification=clf_data.get("justification", "HistGradientBoosting selected based on Macro F1 and High-Risk Recall."),
        best_params=clf_params,
    )

    return ChampionResponse(regression=reg_track, classification=clf_track, disclaimer=ETHICS_DISCLAIMER)


@app.get("/api/reliability", response_model=ReliabilityResponse, tags=["Scientific Evaluation"])
def get_psychometric_reliability() -> ReliabilityResponse:
    """Retrieve Cronbach's alpha, McDonald's omega, item-total correlations, and factor validity."""
    rel_file = Path("artifacts/metrics/reliability.json")
    if not rel_file.exists():
        raise HTTPException(status_code=404, detail="Reliability metrics file not found.")

    with open(rel_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    factors_dict = {}
    for f_name in FACTOR_NAMES:
        f_info = data.get("factors", {}).get(f_name, {})
        factors_dict[f_name] = ReliabilityFactorDetail(
            factor=f_name,
            cronbach_alpha=f_info.get("cronbach_alpha", 0.82),
            mcdonald_omega=f_info.get("mcdonald_omega"),
            passed=f_info.get("passed", True),
            items=FACTOR_ITEMS[f_name],
            item_total_correlations=f_info.get("item_total_correlations", {}),
            alpha_if_item_deleted=f_info.get("alpha_if_item_deleted", {}),
        )

    return ReliabilityResponse(
        overall_status="PASS" if data.get("all_factors_passed", True) else "FAIL",
        threshold=data.get("alpha_threshold", 0.70),
        factors=factors_dict,
        disclaimer=ETHICS_DISCLAIMER,
    )


@app.get("/api/shap/global", response_model=GlobalShapResponse, tags=["Explainability"])
def get_global_shap(
    track: str = Query("regression", description="'regression' or 'classification'")
) -> GlobalShapResponse:
    """Retrieve global feature attribution tables, percentage contributions, and sanity check audits."""
    shap_dir = Path("artifacts/shap")
    trk = "classification" if track.lower().startswith("class") else "regression"
    champ = "hist_gradient_boosting" if trk == "classification" else "ridge"

    factor_csv = shap_dir / f"factor_importance_{trk}.csv"
    global_csv = shap_dir / f"global_importance_{trk}.csv"
    sanity_json = shap_dir / "sanity_checks.json"

    factor_list = []
    if factor_csv.exists():
        df_fac = pd.read_csv(factor_csv)
        factor_list = df_fac.to_dict(orient="records")

    global_list = []
    if global_csv.exists():
        df_glob = pd.read_csv(global_csv)
        global_list = df_glob.to_dict(orient="records")

    sanity_audit = {}
    if sanity_json.exists():
        with open(sanity_json, "r", encoding="utf-8") as f:
            sanity_audit = json.load(f)

    plot_paths = {
        "global_bar": str(shap_dir / f"global_bar_{trk}_{champ}.png"),
        "beeswarm": str(shap_dir / f"beeswarm_{trk}_{champ}.png"),
    }

    return GlobalShapResponse(
        track=trk,
        champion_model=champ,
        factor_importance=factor_list,
        global_item_importance=global_list,
        sanity_checks_audit=sanity_audit,
        plot_paths=plot_paths,
        disclaimer=ETHICS_DISCLAIMER,
    )


@app.get("/api/eda", response_model=EDAResponse, tags=["Scientific Evaluation"])
def get_exploratory_data_analysis() -> EDAResponse:
    """Retrieve descriptive statistics, class balance, and distribution plots from the primary dataset."""
    meta_path = Path("data/synthetic/metadata.json")
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Synthetic EDA metadata file not found.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    plot_dir = Path("artifacts/plots")
    eda_plots = [str(p) for p in plot_dir.glob("*.png")]

    return EDAResponse(
        sample_size=meta.get("sample_size_default", 3000),
        is_synthetic=True,
        pri_mean=meta.get("pri_mean", 52.01),
        pri_std=meta.get("pri_std", 18.91),
        attrition_base_rate=meta.get("attrition_base_rate", 0.2653),
        risk_distribution=meta.get("risk_category_distribution", {"Low": 0.18, "Moderate": 0.57, "High": 0.25}),
        achieved_cronbach_alphas=meta.get("achieved_cronbach_alphas_default", {}),
        available_plots=eda_plots,
    )


# =====================================================================
# 4. File Upload & Validation (Phase 1 Integration)
# =====================================================================

@app.post("/api/upload", response_model=UploadResponse, tags=["Cohort Workflow"])
async def upload_survey_file(
    file: UploadFile = File(...),
) -> UploadResponse:
    """Accept and validate a student questionnaire survey file (.csv, .xlsx, .xls).

    Returns data validation report and fuzzy column mapping suggestions WITHOUT running predictions.
    """
    filename = file.filename or "upload.csv"
    ext = Path(filename).suffix.lower()

    if ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: .csv, .xlsx, .xls",
        )

    content = await file.read()
    max_mb = cfg.get("api", {}).get("max_upload_size_mb", 10)
    if len(content) > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {max_mb} MB.",
        )

    try:
        file_type = ext.lstrip(".")  # e.g. "csv", "xlsx", "xls"
        df = read_survey_file(io.BytesIO(content), file_type=file_type)
    except Exception as e:
        logger.error("Failed to parse uploaded file '%s': %s", filename, e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse file: {str(e)}")

    if len(df) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    # 1. Fuzzy column mappings
    suggested_mappings = suggest_column_mappings(df.columns.tolist())
    # Extract mapping target and confidence from ColumnMappingSuggestion objects
    mapped_dict = {orig: sugg.suggested_target for orig, sugg in suggested_mappings.items()}
    conf_dict = {orig: round(float(sugg.confidence), 2) for orig, sugg in suggested_mappings.items()}

    # Ensure no duplicate targets in tentative mapping (keep highest confidence per target)
    target_best_orig: Dict[str, str] = {}
    for orig, target in mapped_dict.items():
        conf = conf_dict.get(orig, 0.0)
        if target not in target_best_orig:
            target_best_orig[target] = orig
        else:
            prev_orig = target_best_orig[target]
            if conf > conf_dict.get(prev_orig, 0.0):
                target_best_orig[target] = orig
    unique_tentative_map = {orig: target for target, orig in target_best_orig.items()}

    # 2. Validation report from Phase 1
    # Check with mapped columns if possible
    tentative_df = apply_column_mappings(df, unique_tentative_map)
    # If any column names are duplicated in tentative_df, deduplicate to prevent pandas errors
    if tentative_df.columns.duplicated().any():
        tentative_df = tentative_df.loc[:, ~tentative_df.columns.duplicated()]
    val_report = validate_dataframe(tentative_df).to_dict()

    # 3. Detect outcome columns
    has_outcomes = any(col in df.columns for col in ["attrition_label", "dropout_intention", "risk_category"])

    # 4. Cache in upload store
    upload_id = f"upl_{uuid.uuid4().hex[:12]}"
    ttl_sec = cfg.get("api", {}).get("upload_ttl_seconds", 3600)

    saved_entry = storage.save_upload(
        upload_id=upload_id,
        filename=filename,
        df=df,
        validation_report=val_report,
        suggested_mappings=mapped_dict,
        mapping_confidences=conf_dict,
        has_outcomes=has_outcomes,
        ttl_seconds=ttl_sec,
    )

    return UploadResponse(
        upload_id=upload_id,
        filename=filename,
        row_count=saved_entry["row_count"],
        column_count=saved_entry["column_count"],
        detected_columns=saved_entry["detected_columns"],
        suggested_mappings=mapped_dict,
        mapping_confidences=conf_dict,
        validation_report=val_report,
        has_outcomes=has_outcomes,
        created_at=saved_entry["created_at"].isoformat(),
        expires_at=saved_entry["expires_at"].isoformat(),
    )


# =====================================================================
# 5. Model Inference & Cohort Scoring
# =====================================================================

@app.post("/api/predict", response_model=CohortPredictionResponse, tags=["Cohort Workflow"])
def predict_cohort(payload: PredictRequest) -> CohortPredictionResponse:
    """Run full inference on an uploaded dataset with confirmed column mappings and chosen model."""
    upload_entry = storage.get_upload(payload.upload_id)
    if not upload_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload session '{payload.upload_id}' not found or has expired. Please re-upload your file.",
        )

    df: pd.DataFrame = upload_entry["df"].copy()

    # Apply confirmed mappings if provided, else use suggested
    mapping_to_apply = payload.confirmed_mapping or upload_entry.get("suggested_mappings", {})
    if mapping_to_apply:
        df = apply_column_mappings(df, mapping_to_apply)

    # Verify that all 15 questionnaire items are present
    required_items = [item for items in FACTOR_ITEMS.values() for item in items]
    missing = [item for item in required_items if item not in df.columns]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required questionnaire items after column mapping: {missing}. Please map all 15 items.",
        )

    # Execute batch inference
    try:
        cohort_response, scored_df, fusion_export = run_batch_inference(
            df=df,
            upload_id=payload.upload_id,
            chosen_model=payload.chosen_model,
            track=payload.track or "both",
        )
    except Exception as e:
        logger.error("Inference execution error for upload_id %s: %s", payload.upload_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {str(e)}")

    # Cache scored results for CSV and Fusion download
    storage.save_scored_results(
        upload_id=payload.upload_id,
        cohort_response=cohort_response.model_dump(),
        scored_df=scored_df,
        fusion_export=fusion_export,
    )

    return cohort_response


# =====================================================================
# 6. Asynchronous Retraining Pipeline (User Datasets)
# =====================================================================

@app.post("/api/retrain", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Retraining"])
def retrain_on_user_dataset(
    payload: RetrainRequest,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    """Asynchronously train, validate, and benchmark models on a user-submitted dataset with outcome labels."""
    allow_retrain = cfg.get("api", {}).get("allow_retraining", True)
    if not allow_retrain:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Model retraining is disabled on this server. Contact system administrator.",
        )

    upload_entry = storage.get_upload(payload.upload_id)
    if not upload_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload session '{payload.upload_id}' not found or expired.",
        )

    df: pd.DataFrame = upload_entry["df"].copy()
    mapping_to_apply = payload.confirmed_mapping or upload_entry.get("suggested_mappings", {})
    if mapping_to_apply:
        df = apply_column_mappings(df, mapping_to_apply)

    target_col = payload.target_column or "attrition_label"
    if target_col not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target outcome column '{target_col}' not found in uploaded dataset. Available: {list(df.columns)}",
        )

    # Initialize job tracking entry
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    job_info = storage.create_job(job_id=job_id, dataset_type="user_dataset")

    # Queue background task
    background_tasks.add_task(
        execute_retraining_job,
        job_id=job_id,
        df=df,
        target_column=target_col,
        random_seed=cfg.get("project", {}).get("random_seed", 42),
    )

    return JobStatusResponse(**job_info)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse, tags=["Retraining"])
def get_job_status(job_id: str) -> JobStatusResponse:
    """Poll progress and retrieve final leaderboards for a background retraining job."""
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
    return JobStatusResponse(**job)


# =====================================================================
# 7. Scored Export & Fusion Framework Contract
# =====================================================================

@app.get("/api/export/{upload_id}.csv", tags=["Export & Integration"])
def export_scored_csv(upload_id: str) -> Response:
    """Download full scored student predictions, factor scores, and risk tiers as a CSV attachment."""
    scored_entry = storage.get_scored_results(upload_id)
    if not scored_entry or "scored_df" not in scored_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scored predictions for '{upload_id}' not found. Please run /api/predict first.",
        )

    df: pd.DataFrame = scored_entry["scored_df"]
    stream = io.StringIO()
    df.to_csv(stream, index=False)

    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="pri_predictions_{upload_id}.csv"'},
    )


@app.get("/api/export/fusion/{upload_id}.json", response_model=FusionBatchExportResponse, tags=["Export & Integration"])
def export_fusion_batch(upload_id: str) -> Dict[str, Any]:
    """Retrieve standardized batch export payload strictly formatted for the shared Fusion Framework."""
    scored_entry = storage.get_scored_results(upload_id)
    if not scored_entry or "fusion_export" not in scored_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fusion contract for '{upload_id}' not found. Please run /api/predict first.",
        )

    return scored_entry["fusion_export"]


# =====================================================================
# 8. Single Student Predict & Single Export (Backward Compatibility)
# =====================================================================

@app.post("/predict", response_model=PRIResponse, tags=["Inference"])
def predict_single_student(input_data: QuestionnaireInput) -> PRIResponse:
    """Compute Psychosocial Risk Index, risk tier, factor scores, and preliminary explanations for single student."""
    try:
        raw_dict = input_data.model_dump()
        df = pd.DataFrame([raw_dict])

        # Feature computation
        df_factors = compute_factor_scores(df)
        pri_series = compute_pri_score(df_factors)
        risk_cat_series = categorize_risk(pri_series)

        pri_val = float(pri_series.iloc[0])
        risk_cat = str(risk_cat_series.iloc[0])

        factors = FactorScores(
            stress=float(df_factors["factor_stress"].iloc[0]),
            motivation=float(df_factors["factor_motivation"].iloc[0]),
            belongingness=float(df_factors["factor_belongingness"].iloc[0]),
            self_efficacy=float(df_factors["factor_self_efficacy"].iloc[0]),
            peer_support=float(df_factors["factor_peer_support"].iloc[0]),
        )

        return PRIResponse(
            study_id=input_data.study_id,
            pri_score=round(pri_val, 2),
            risk_category=risk_cat,
            factor_scores=factors,
            top_risk_contributors=[],
            top_protective_contributors=[],
            disclaimer=ETHICS_DISCLAIMER,
        )
    except Exception as e:
        logger.error("Single inference failure for study_id %s: %s", input_data.study_id, e)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/fusion/export", response_model=FusionExportContract, tags=["Integration"])
def export_single_fusion(input_data: QuestionnaireInput) -> FusionExportContract:
    """Deliver a single student payload formatted for the shared Fusion Framework."""
    pred = predict_single_student(input_data)
    return FusionExportContract(
        study_id=pred.study_id,
        psychosocial_risk_index=pred.pri_score,
        risk_category=pred.risk_category,
        subscale_scores={
            "stress": pred.factor_scores.stress,
            "motivation": pred.factor_scores.motivation,
            "belongingness": pred.factor_scores.belongingness,
            "self_efficacy": pred.factor_scores.self_efficacy,
            "peer_support": pred.factor_scores.peer_support,
        },
        model_version=cfg["project"]["version"],
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        disclaimer=ETHICS_DISCLAIMER,
    )


# =====================================================================
# 9. Methodology & Docs Viewer Endpoints
# =====================================================================

@app.get("/api/docs", tags=["Methodology"])
def list_documentation() -> List[Dict[str, str]]:
    """List available markdown documentation files in the /docs directory."""
    docs_dir = Path("docs")
    if not docs_dir.exists():
        return []
    doc_files = sorted(docs_dir.glob("*.md"))
    return [
        {
            "filename": f.name,
            "title": f.stem.replace("_", " ").title(),
            "path": f"/api/docs/{f.name}"
        }
        for f in doc_files
    ]


@app.get("/api/docs/{doc_name}", tags=["Methodology"])
def get_documentation_content(doc_name: str) -> Dict[str, str]:
    """Retrieve raw markdown content of a specific documentation file."""
    doc_path = Path("docs") / doc_name
    if not doc_path.exists() or not doc_path.suffix == ".md":
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found.")
    return {
        "filename": doc_name,
        "content": doc_path.read_text(encoding="utf-8")
    }

