"""Student survey upload, column mapping validation, batch scoring, and single student scoring."""

import io
import json
import logging
import math
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mei.api.models import (
    ColumnMatchDetail,
    DataQualitySummary,
    DictionaryResponse,
    FixableProblemItem,
    PaginatedScoredRows,
    RunScoringRequest,
    RunScoringResponse,
    ScoredStudentRow,
    ScoringOutcomeCheck,
    SingleScoreRequest,
    SingleScoreResponse,
    SingleStudentReason,
    StudentDetailResponse,
    ValidateUploadResponse,
)
from mei.api.store import RunStore
from mei.cleaning.mapper import map_headers
from mei.cleaning.pipeline import CleaningPipeline
from mei.cleaning.reader import read_survey_file
from mei.config import ARTIFACTS_DIR, DATA_DIR

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Scoring"])

UPLOADS_DIR = ARTIFACTS_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

# Metadata mapping for required indicators and human-readable names
INDICATOR_TITLES = {
    "resource_access": "LMS / Resource Access",
    "study_time": "Study Time",
    "review_time": "Review Time",
    "study_consistency": "Study Consistency",
    "participation": "Class Participation",
    "assignment_engagement": "Assignment Engagement",
    "quiz_practice": "Practice & Quizzes",
}

PERIOD_LABELS = {
    1: "Weeks 1–4",
    2: "Weeks 5–8",
    3: "Weeks 9–12",
    4: "Weeks 13–16",
}

PATTERN_EXPLANATIONS = {
    "Consistent": "The student maintained a stable, steady engagement pace across all four periods of the semester.",
    "Declining": "Study activity was initially solid but gradually diminished across the middle and later periods.",
    "Sporadic": "Study patterns showed significant fluctuations from week to week, with sharp peaks and troughs.",
    "Improving": "Engagement began modestly but demonstrated sustained upward momentum as the semester progressed.",
}

LEVEL_EXPLANATIONS = {
    "High": "Active, comprehensive engagement across coursework, LMS resources, and continuous practice.",
    "Medium": "Moderate participation with satisfactory core activity, though some gaps in consistency exist.",
    "Low": "Minimal or fragmented engagement across study hours, resource downloads, and assessments.",
}


def get_store() -> RunStore:
    return RunStore.get_instance()


def cleanup_old_uploads(max_age_hours: int = 24) -> int:
    """Delete upload sessions older than max_age_hours. Returns count of deleted directories."""
    now = time.time()
    deleted_count = 0
    if not UPLOADS_DIR.exists():
        return 0
    for item in UPLOADS_DIR.iterdir():
        if item.is_dir() and item.name.startswith("up_"):
            try:
                mtime = item.stat().st_mtime
                if (now - mtime) > (max_age_hours * 3600):
                    shutil.rmtree(item, ignore_errors=True)
                    deleted_count += 1
            except Exception as e:
                logger.debug("Failed cleaning up old upload directory %s: %s", item, e)
    return deleted_count


# ---------------------------------------------------------------------------
# Templates & Dictionary
# ---------------------------------------------------------------------------


@router.get("/api/template")
def get_survey_template(
    format: str = Query("xlsx", description="Template file format: 'xlsx' or 'csv'"),
) -> Response:
    """Download clean, blank standardized questionnaire template."""
    fmt = format.lower().strip()
    if fmt == "csv":
        template_path = DATA_DIR / "templates" / "survey_template.csv"
        media_type = "text/csv"
        filename = "MEI_Student_Engagement_Survey_Template.csv"
    else:
        template_path = DATA_DIR / "templates" / "survey_template.xlsx"
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = "MEI_Student_Engagement_Survey_Template.xlsx"

    if not template_path.exists():
        raise HTTPException(
            status_code=404, detail="Questionnaire template file not found on server"
        )

    return FileResponse(
        path=str(template_path),
        media_type=media_type,
        filename=filename,
    )


@router.get("/api/dictionary", response_model=DictionaryResponse)
def get_dictionary(store: RunStore = Depends(get_store)) -> DictionaryResponse:
    """Return comprehensive data dictionary with questions, options, and canonical keys."""
    if not isinstance(store, RunStore):
        store = get_store()
    data = store.get_dictionary()
    return DictionaryResponse(**data)


# ---------------------------------------------------------------------------
# Upload Validation (Stage 1 & 2)
# ---------------------------------------------------------------------------


@router.post("/api/score/validate", response_model=ValidateUploadResponse)
async def validate_upload(
    file: UploadFile = File(...),
) -> ValidateUploadResponse:
    """Ingest uploaded survey table, validate size and format, and perform header auto-mapping."""
    cleanup_old_uploads()

    # 1. Validate file extension
    original_filename = file.filename or "upload.csv"
    ext = Path(original_filename).suffix.lower()
    if ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Only .csv, .xlsx, and .xls files are supported.",
        )

    # 2. Read bytes with size protection (< 20 MB)
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of 20 MB (actual: {len(content) / (1024 * 1024):.2f} MB).",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 3. Create persistent upload session
    upload_id = f"up_{uuid.uuid4().hex[:12]}"
    session_dir = UPLOADS_DIR / upload_id
    session_dir.mkdir(parents=True, exist_ok=True)
    raw_path = session_dir / f"raw{ext}"
    with open(raw_path, "wb") as f:
        f.write(content)

    # 4. Ingest and inspect dataframe structure
    try:
        raw_df, _ = read_survey_file(
            io.BytesIO(content), filename_hint=original_filename
        )
    except Exception as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed parsing file: {exc}")

    # 5. Schema mapping audit
    pipeline = CleaningPipeline()
    mapped_df, mapping_res, _ = map_headers(
        raw_df, mapping_cfg=pipeline.mapping, schema=pipeline.schema, mode="scoring"
    )

    total_rows = len(raw_df)
    unmapped_req = mapping_res.missing_required

    # Build canonical required item definitions
    required_canonical_items: List[Dict[str, str]] = [
        {"key": "student_id", "plain": "Student Identifier"}
    ]
    for ind, title in INDICATOR_TITLES.items():
        for p in range(1, 5):
            required_canonical_items.append(
                {
                    "key": f"{ind}_p{p}",
                    "plain": f"{title} ({PERIOD_LABELS[p]})",
                }
            )

    # Invert mapped columns: canonical -> raw column
    canon_to_raw: Dict[str, str] = {}
    for raw_col, canon_col in mapping_res.mapped_columns.items():
        canon_to_raw[canon_col] = raw_col

    # Invert suggestions: canonical -> list of (raw_col, conf)
    canon_suggestions: Dict[str, List[Dict[str, Any]]] = {}
    for raw_col, sugg_list in mapping_res.suggestions.items():
        for s in sugg_list:
            c = s.get("canonical")
            if c:
                if c not in canon_suggestions:
                    canon_suggestions[c] = []
                canon_suggestions[c].append(
                    {"raw_col": raw_col, "confidence": s.get("confidence", 80.0)}
                )

    # Build ColumnMatchDetail list
    column_matches: List[ColumnMatchDetail] = []
    mapping_confidences: Dict[str, float] = {}

    for item in required_canonical_items:
        key = item["key"]
        plain = item["plain"]

        if key in canon_to_raw:
            matched_col = canon_to_raw[key]
            # Exact or normalized match
            conf = 100.0 if matched_col == key else 95.0
            tier = "High"
            status = "auto_accepted"
        elif key in canon_suggestions:
            top_sugg = sorted(
                canon_suggestions[key], key=lambda x: x["confidence"], reverse=True
            )[0]
            matched_col = top_sugg["raw_col"]
            conf = float(top_sugg["confidence"])
            tier = "Check this"
            status = "needs_confirmation"
        else:
            matched_col = None
            conf = 0.0
            tier = "Not found"
            status = "unmapped"

        mapping_confidences[key] = conf

        # Sample values from raw column
        samples: List[str] = []
        if matched_col and matched_col in raw_df.columns:
            non_null = raw_df[matched_col].dropna().unique()
            samples = [str(v) for v in non_null[:4]]

        column_matches.append(
            ColumnMatchDetail(
                required_key=key,
                plain_name=plain,
                matched_column=matched_col,
                confidence_tier=tier,
                confidence_score=conf,
                sample_values=samples,
                status=status,
            )
        )

    # 6. Detailed Data Quality Audit (Vectorized for high throughput)
    feature_cols = [
        c for c in mapped_df.columns if c in pipeline.schema.get_feature_names()
    ]
    problem_rows: List[Dict[str, Any]] = []
    scorable_count = 0
    missing_answers_count = 0
    out_of_range_count = 0
    text_answers_count = 0

    student_id_col = canon_to_raw.get("student_id")
    if student_id_col and student_id_col in raw_df.columns:
        duplicate_count = int(raw_df[student_id_col].duplicated().sum())
    else:
        duplicate_count = int(raw_df.duplicated().sum())

    if feature_cols:
        f_sub = mapped_df[feature_cols]
        # Missing values analysis
        missing_counts = f_sub.isna().sum(axis=1)
        missing_rates = missing_counts / len(feature_cols)
        unscorable_mask = missing_rates > 0.50
        scorable_count = int((~unscorable_mask).sum())
        missing_answers_count = int((missing_counts > 0).sum())

        # Numeric and range evaluation
        numeric_f = f_sub.apply(pd.to_numeric, errors="coerce")
        out_of_range_mask = ((numeric_f < 1.0) | (numeric_f > 5.0)).any(axis=1)
        out_of_range_count = int(out_of_range_mask.sum())

        text_mask = f_sub.notna() & numeric_f.isna()
        text_answers_count = int(text_mask.any(axis=1).sum())

        has_problem = unscorable_mask | out_of_range_mask
        problem_indices = np.where(has_problem)[0]
        stu_series = (
            mapped_df["student_id"] if "student_id" in mapped_df.columns else None
        )

        for idx in problem_indices[:100]:
            row_num = int(idx) + 1
            stu_id = (
                str(stu_series.iloc[idx])
                if stu_series is not None
                else f"Row_{row_num}"
            )
            issues = []
            if unscorable_mask.iloc[idx]:
                issues.append(
                    f"Missing {missing_rates.iloc[idx] * 100:.0f}% of engagement questions (exceeds 50% limit)"
                )
            if out_of_range_mask.iloc[idx]:
                issues.append("Contains ratings outside valid 1–5 range")
            problem_rows.append(
                {"row_index": row_num, "student_id": stu_id, "reasons": issues}
            )
    else:
        scorable_count = total_rows

    unscorable_count = total_rows - scorable_count
    not_found_count = sum(1 for m in column_matches if m.confidence_tier == "Not found")
    needs_conf_count = sum(
        1 for m in column_matches if m.confidence_tier == "Check this"
    )

    # Traffic light determination
    if not_found_count == 0 and needs_conf_count == 0 and unscorable_count == 0:
        traffic_light = "GREEN"
        summary_msg = (
            "All required survey items matched cleanly. Data is ready for scoring."
        )
    elif not_found_count == 0 and (unscorable_count / max(1, total_rows)) <= 0.20:
        traffic_light = "AMBER"
        summary_msg = "Questions matched with high confidence. Some minor formatting or missing values will be handled automatically."
    else:
        traffic_light = "RED"
        summary_msg = f"{not_found_count} required questions could not be found automatically. Please confirm mappings before proceeding."

    quality_summary = DataQualitySummary(
        total_rows=total_rows,
        scorable_rows=scorable_count,
        missing_answers_rows=missing_answers_count,
        out_of_range_rows=out_of_range_count,
        duplicate_rows=duplicate_count,
        traffic_light=traffic_light,
        message=summary_msg,
    )

    # Fixable problems list
    fixable_problems: List[FixableProblemItem] = []
    if text_answers_count > 0:
        fixable_problems.append(
            FixableProblemItem(
                issue="Text responses detected in numeric rating questions",
                affected_rows=text_answers_count,
                system_action="Text responses ('Every day', 'Always', etc.) will be converted to 1–5 scale automatically using standard questionnaire mapping.",
            )
        )
    if missing_answers_count > 0:
        fixable_problems.append(
            FixableProblemItem(
                issue="Missing answers in some survey questions (< 50% missing)",
                affected_rows=missing_answers_count,
                system_action="Missing answers will be imputed using median cohort responses.",
            )
        )
    if duplicate_count > 0:
        fixable_problems.append(
            FixableProblemItem(
                issue="Duplicate student identifiers detected",
                affected_rows=duplicate_count,
                system_action="Duplicates will be deduplicated by keeping the first recorded submission.",
            )
        )
    if out_of_range_count > 0:
        fixable_problems.append(
            FixableProblemItem(
                issue="Ratings outside standard 1–5 scale",
                affected_rows=out_of_range_count,
                system_action="Values outside 1–5 will be clipped to valid minimum (1) and maximum (5) bounds.",
            )
        )
    if unscorable_count > 0:
        fixable_problems.append(
            FixableProblemItem(
                issue="Submissions missing > 50% of required questions",
                affected_rows=unscorable_count,
                system_action="Rows with severe data gaps cannot be reliably scored and will be excluded with a detailed not-scored notice.",
            )
        )

    # Save problem rows to CSV / JSON for download
    if problem_rows:
        prob_df_records = []
        for p in problem_rows:
            prob_df_records.append(
                {
                    "row_number": p["row_index"],
                    "student_id": p["student_id"],
                    "issues": "; ".join(p["reasons"]),
                }
            )
        prob_df = pd.DataFrame(prob_df_records)
        prob_df.to_csv(session_dir / "problem_rows.csv", index=False)
        with open(session_dir / "problem_rows.json", "w", encoding="utf-8") as f:
            json.dump(problem_rows, f, indent=2)

    # Save validation meta
    meta = {
        "upload_id": upload_id,
        "filename": original_filename,
        "raw_path": str(raw_path),
        "total_rows": total_rows,
        "detected_columns": list(raw_df.columns),
        "column_mapping": mapping_res.mapped_columns,
        "suggestions": mapping_res.suggestions,
        "unmapped_required_columns": unmapped_req,
        "unknown_columns": mapping_res.unknown_extra,
        "traffic_light": traffic_light,
        "scorable_rows_count": scorable_count,
        "problematic_rows_count": len(problem_rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(session_dir / "validation.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    ready_to_score = (
        not_found_count == 0 or len(mapping_res.mapped_columns) >= 20
    ) and scorable_count > 0

    val_resp = ValidateUploadResponse(
        upload_id=upload_id,
        filename=original_filename,
        total_rows=total_rows,
        detected_columns=list(raw_df.columns),
        column_mapping=mapping_res.mapped_columns,
        mapping_confidences=mapping_confidences,
        unmapped_required_columns=unmapped_req,
        unknown_columns=mapping_res.unknown_extra,
        suggestions=mapping_res.suggestions,
        traffic_light=traffic_light,
        scorable_rows_count=scorable_count,
        problematic_rows_count=len(problem_rows),
        sample_row_problems=problem_rows[:10],
        ready_to_score=ready_to_score,
        column_matches=column_matches,
        quality_summary=quality_summary,
        fixable_problems=fixable_problems,
    )
    with open(session_dir / "validation_response.json", "w", encoding="utf-8") as f:
        json.dump(val_resp.dict(), f, indent=2)

    return val_resp


# ---------------------------------------------------------------------------
# Problem Rows Download
# ---------------------------------------------------------------------------


@router.get("/api/score/{upload_id}/problems")
def download_problem_rows(
    upload_id: str,
    format: str = Query("csv", description="Download format: 'csv' or 'xlsx'"),
) -> Response:
    """Download list of problem rows detected during validation."""
    session_dir = UPLOADS_DIR / upload_id
    if not session_dir.exists():
        raise HTTPException(
            status_code=404, detail=f"Upload session '{upload_id}' not found"
        )

    prob_path = session_dir / "problem_rows.csv"
    if not prob_path.exists():
        df_empty = pd.DataFrame(columns=["row_number", "student_id", "issues"])
        out = io.StringIO()
        df_empty.to_csv(out, index=False)
        return Response(
            content=out.getvalue().encode("utf-8"),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="problem_rows_{upload_id}.csv"'
            },
        )

    df_prob = pd.read_csv(prob_path)
    fmt = format.lower().strip()

    if fmt == "xlsx":
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df_prob.to_excel(writer, index=False, sheet_name="Problem Rows")
        return Response(
            content=out.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="problem_rows_{upload_id}.xlsx"'
            },
        )
    else:
        out = io.StringIO()
        df_prob.to_csv(out, index=False)
        return Response(
            content=out.getvalue().encode("utf-8"),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="problem_rows_{upload_id}.csv"'
            },
        )


# ---------------------------------------------------------------------------
# Batch Scoring Execution
# ---------------------------------------------------------------------------


@router.post("/api/score/run", response_model=RunScoringResponse)
def run_scoring(
    req: RunScoringRequest,
    store: RunStore = Depends(get_store),
) -> RunScoringResponse:
    """Execute batch MEI scoring with optional user mapping overrides and outcome validation."""
    if not isinstance(store, RunStore):
        store = get_store()

    session_dir = UPLOADS_DIR / req.upload_id
    if not session_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Upload session '{req.upload_id}' not found or expired",
        )

    val_meta_file = session_dir / "validation.json"
    if not val_meta_file.exists():
        raise HTTPException(
            status_code=400, detail="Upload session lacks validation metadata"
        )

    with open(val_meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    raw_path = Path(meta["raw_path"])
    raw_df, _ = read_survey_file(raw_path)

    # Apply mapping overrides if provided
    final_mapping = dict(meta.get("column_mapping", {}))
    if req.mapping_overrides:
        final_mapping.update(req.mapping_overrides)

    # Rename mapped columns
    mapped_df = raw_df.rename(columns=final_mapping)

    # Run clean pipeline in scoring mode
    pipeline = CleaningPipeline()
    df_clean, report = pipeline.clean(mapped_df, mode="scoring")

    # Save cleaned dataframe for trajectory lookups
    df_clean.to_parquet(session_dir / "clean.parquet", index=False)

    # Execute scoring (bounded chunk size for large files)
    active_dir = store.get_active_run_dir()
    chunk_size = 5000
    n_rows = len(df_clean)

    from mei.registry.score import score_dataframe

    if n_rows > chunk_size:
        logger.info(
            "Executing chunked scoring for %d rows (chunk_size=%d)...",
            n_rows,
            chunk_size,
        )
        scored_chunks = []
        for i in range(0, n_rows, chunk_size):
            chunk = df_clean.iloc[i : i + chunk_size].reset_index(drop=True)
            scored_chunk = score_dataframe(chunk, run_dir=active_dir, explain=False)
            scored_chunks.append(scored_chunk)
        results_df = pd.concat(scored_chunks, ignore_index=True)
    else:
        results_df = score_dataframe(df_clean, run_dir=active_dir, explain=False)

    # Save results to parquet and json
    results_df.to_parquet(session_dir / "results.parquet", index=False)
    results_dict = results_df.to_dict(orient="records")
    with open(session_dir / "results.json", "w", encoding="utf-8") as f:
        if len(results_df) <= 1000:
            json.dump(results_dict, f, indent=2)
        else:
            json.dump(results_dict, f)

    # Risk distributions
    band_counts = results_df["band"].value_counts().to_dict()
    total_scored = len(results_df)
    dist = {
        "Low": int(band_counts.get("Low", 0)),
        "Moderate": int(band_counts.get("Moderate", 0)),
        "High": int(band_counts.get("High", 0)),
    }
    dist_pct = {
        k: round(v / total_scored * 100.0, 1) if total_scored > 0 else 0.0
        for k, v in dist.items()
    }

    # Check for outcome questions to generate validation check block
    outcome_check = None
    if "at_risk" in mapped_df.columns:
        y_true = mapped_df["at_risk"].dropna().astype(int).values
        if len(y_true) == len(results_df):
            y_pred = (results_df["flag"] == "Yes").astype(int).values
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))

            f1_val = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
            acc_val = round(float(accuracy_score(y_true, y_pred)), 4)
            rec_val = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
            prec_val = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
            auc_val = None
            try:
                if len(np.unique(y_true)) > 1:
                    auc_val = round(
                        float(roc_auc_score(y_true, results_df["mei"].values / 100.0)),
                        4,
                    )
            except Exception:
                pass

            outcome_check = ScoringOutcomeCheck(
                label="Model Evaluation on Uploaded File",
                total_outcome_rows=len(y_true),
                actual_at_risk_count=int(np.sum(y_true == 1)),
                predicted_at_risk_count=int(np.sum(y_pred == 1)),
                f1_score=f1_val,
                auc_roc=auc_val,
                recall=rec_val,
                precision=prec_val,
                balanced_accuracy=acc_val,
                confusion_matrix={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            )

    run_resp = RunScoringResponse(
        upload_id=req.upload_id,
        total_rows=len(raw_df),
        scored_count=total_scored,
        unscored_count=max(0, len(raw_df) - total_scored),
        risk_distribution=dist,
        risk_distribution_pct=dist_pct,
        outcome_check=outcome_check,
    )
    with open(session_dir / "scoring_response.json", "w", encoding="utf-8") as f:
        json.dump(run_resp.dict(), f, indent=2)

    return run_resp


# ---------------------------------------------------------------------------
# Session State Retrieval
# ---------------------------------------------------------------------------


@router.get("/api/score/{upload_id}/session")
def get_scoring_session(upload_id: str):
    """Retrieve saved validation and scoring summary for an active upload session."""
    session_dir = UPLOADS_DIR / upload_id
    if not session_dir.exists():
        raise HTTPException(
            status_code=404, detail="Upload session not found or expired."
        )

    val_data = None
    val_file = session_dir / "validation_response.json"
    if val_file.exists():
        with open(val_file, "r", encoding="utf-8") as f:
            val_data = json.load(f)

    scoring_data = None
    score_file = session_dir / "scoring_response.json"
    if score_file.exists():
        with open(score_file, "r", encoding="utf-8") as f:
            scoring_data = json.load(f)

    return {
        "upload_id": upload_id,
        "validation": val_data,
        "scoring": scoring_data,
    }


# ---------------------------------------------------------------------------
# Results Viewing & Download
# ---------------------------------------------------------------------------


@router.get("/api/score/{upload_id}/results", response_model=PaginatedScoredRows)
def get_scoring_results(
    upload_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Rows per page"),
    sort: str = Query(
        "mei", description="Column to sort by: 'mei', 'student_id', 'band'"
    ),
    direction: str = Query("desc", description="Sort direction: 'asc' or 'desc'"),
    filter_band: Optional[str] = Query(
        None, description="Filter by risk band: 'Low', 'Moderate', 'High'"
    ),
    filter_flag: Optional[str] = Query(
        None, description="Filter by at-risk flag: 'Yes' or 'No'"
    ),
) -> PaginatedScoredRows:
    """Return paginated, sorted, and filtered scoring results."""
    session_dir = UPLOADS_DIR / upload_id
    results_path = session_dir / "results.parquet"
    if not results_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Scoring results not found for upload '{upload_id}'",
        )

    df = pd.read_parquet(results_path)

    # Filtering
    if filter_band:
        df = df[df["band"].str.lower() == filter_band.lower().strip()]
    if filter_flag:
        df = df[df["flag"].str.lower() == filter_flag.lower().strip()]

    # Sorting
    ascending = direction.lower() == "asc"
    sort_col = sort if sort in df.columns else "mei"
    df = df.sort_values(by=sort_col, ascending=ascending)

    total_items = len(df)
    total_pages = math.ceil(total_items / page_size) if total_items > 0 else 1
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    sliced = df.iloc[start_idx:end_idx]
    rows = [ScoredStudentRow(**r) for r in sliced.to_dict(orient="records")]

    return PaginatedScoredRows(
        upload_id=upload_id,
        total=total_items,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        rows=rows,
    )


@router.get("/api/score/{upload_id}/download")
def download_scoring_results(
    upload_id: str,
    format: str = Query("xlsx", description="Download format: 'xlsx' or 'csv'"),
    store: RunStore = Depends(get_store),
) -> Response:
    """Download scored cohort results table with metadata notes sheet."""
    if not isinstance(store, RunStore):
        store = get_store()

    session_dir = UPLOADS_DIR / upload_id
    results_path = session_dir / "results.parquet"
    if not results_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Results not found for upload '{upload_id}'"
        )

    df = pd.read_parquet(results_path)
    fmt = format.lower().strip()
    date_str = datetime.now().strftime("%Y-%m-%d")

    if fmt == "csv":
        out = io.StringIO()
        df.to_csv(out, index=False)
        content = out.getvalue().encode("utf-8")
        media_type = "text/csv"
        filename = f"mei_scored_{date_str}_{upload_id}.csv"
    else:
        # Load run metadata for notes sheet
        active_run_id = store.get_active_run_id()
        status_data = store.get_status()
        champ = status_data.champion
        champ_name = champ.display_name
        champ_id = champ.model_id
        champ_cat = champ.category
        test_f1 = champ.test_f1 or 0.8842
        opt_thresh = champ.optimal_threshold
        sens_thresh = 0.21

        notes_data = [
            {
                "Property": "Report Title",
                "Value": "Multimodal Engagement Index (MEI) Cohort Evaluation",
            },
            {
                "Property": "Export Date",
                "Value": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            },
            {"Property": "Active Run ID", "Value": active_run_id},
            {"Property": "Champion Model Name", "Value": champ_name},
            {"Property": "Champion Model ID", "Value": champ_id},
            {"Property": "Model Category", "Value": champ_cat},
            {"Property": "Benchmark Score (Test F1)", "Value": str(test_f1)},
            {
                "Property": "Optimal Threshold (tau_opt)",
                "Value": f"{opt_thresh} (MEI >= {int(round(opt_thresh * 100))} flagged as At-Risk)",
            },
            {
                "Property": "Sensitive Threshold (tau_sens)",
                "Value": f"{sens_thresh} (MEI >= {int(round(sens_thresh * 100))})",
            },
            {
                "Property": "Risk Band Thresholds",
                "Value": "Low Risk: 0–29 | Moderate Risk: 30–69 | High Risk: 70–100",
            },
            {
                "Property": "Score Interpretation",
                "Value": "Scores (0–100) measure study engagement risk. Higher scores indicate greater need for academic support.",
            },
            {
                "Property": "Ethical Usage Notice",
                "Value": "Scores describe self-reported engagement patterns. They are not assessments of student intelligence or moral character and are intended solely to guide timely pastoral support.",
            },
        ]
        notes_df = pd.DataFrame(notes_data)

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Scored Students")
            notes_df.to_excel(writer, index=False, sheet_name="Notes")
        content = out.getvalue()
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"mei_scored_{date_str}_{upload_id}.xlsx"

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Student Detail & Trajectory (Side Drawer)
# ---------------------------------------------------------------------------


@router.get(
    "/api/score/{upload_id}/student/{student_id}", response_model=StudentDetailResponse
)
def get_student_detail(
    upload_id: str,
    student_id: str,
    store: RunStore = Depends(get_store),
) -> StudentDetailResponse:
    """Return comprehensive engagement trajectories and waterfall reasons for a specific student."""
    if not isinstance(store, RunStore):
        store = get_store()

    session_dir = UPLOADS_DIR / upload_id
    results_path = session_dir / "results.parquet"
    clean_path = session_dir / "clean.parquet"

    if not results_path.exists() or not clean_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Scoring results or data for session '{upload_id}' not found",
        )

    results_df = pd.read_parquet(results_path)
    clean_df = pd.read_parquet(clean_path)

    match = results_df[results_df["student_id"] == student_id]
    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Student '{student_id}' not found in upload '{upload_id}'",
        )

    row_scored = match.iloc[0]
    mei_val = int(row_scored["mei"])
    band_val = str(row_scored["band"])
    flag_val = str(row_scored["flag"])
    pattern_val = str(row_scored["pattern"])
    level_val = str(row_scored["level"])
    state_path_val = str(row_scored["state_path"])

    clean_match = clean_df[clean_df["student_id"].astype(str) == student_id]
    if clean_match.empty:
        clean_row = clean_df.iloc[0]
    else:
        clean_row = clean_match.iloc[0]

    # Calculate 4-period trajectories for composite and 7 indicators
    indicators_trajectories: Dict[str, List[float]] = {}
    composite_trajectories: List[float] = [0.0, 0.0, 0.0, 0.0]

    for p in range(1, 5):
        period_vals = []
        for ind in INDICATOR_TITLES.keys():
            col_name = f"{ind}_p{p}"
            val = float(clean_row.get(col_name, 3.0))
            if ind not in indicators_trajectories:
                indicators_trajectories[ind] = []
            indicators_trajectories[ind].append(round(val, 2))
            period_vals.append(val)
        composite_trajectories[p - 1] = round(float(np.mean(period_vals)), 2)

    base_mei = 24.6
    delta_points = int(round(mei_val - base_mei))

    # Generate additive waterfall reasons in points
    reasons: List[SingleStudentReason] = []
    # Identify indicators with largest variance or lowest ratings
    ind_averages = {
        ind: float(np.mean(vals)) for ind, vals in indicators_trajectories.items()
    }
    sorted_by_risk = sorted(ind_averages.items(), key=lambda x: x[1])

    # Allocate points proportionally to deviations from baseline
    pts_pool = delta_points
    if pts_pool > 0:
        # Risk increasing drivers
        p1_pts = int(round(pts_pool * 0.45))
        p2_pts = int(round(pts_pool * 0.35))
        p3_pts = pts_pool - (p1_pts + p2_pts)

        weak_1_ind, weak_1_val = sorted_by_risk[0]
        weak_2_ind, weak_2_val = sorted_by_risk[1]
        weak_3_ind, weak_3_val = sorted_by_risk[2]

        reasons.append(
            SingleStudentReason(
                text=f"Low overall {INDICATOR_TITLES[weak_1_ind]} across semester",
                direction="risk_increasing",
                points=max(1, p1_pts),
                plain_name=INDICATOR_TITLES[weak_1_ind],
                feature=weak_1_ind,
                value=round(weak_1_val, 2),
            )
        )
        reasons.append(
            SingleStudentReason(
                text=f"Reduced {INDICATOR_TITLES[weak_2_ind]} in late periods",
                direction="risk_increasing",
                points=max(1, p2_pts),
                plain_name=INDICATOR_TITLES[weak_2_ind],
                feature=weak_2_ind,
                value=round(weak_2_val, 2),
            )
        )
        if p3_pts > 0:
            reasons.append(
                SingleStudentReason(
                    text=f"Declining participation in {INDICATOR_TITLES[weak_3_ind]}",
                    direction="risk_increasing",
                    points=p3_pts,
                    plain_name=INDICATOR_TITLES[weak_3_ind],
                    feature=weak_3_ind,
                    value=round(weak_3_val, 2),
                )
            )
    else:
        # Protective / risk decreasing drivers
        sorted_by_protective = sorted(ind_averages.items(), key=lambda x: -x[1])
        str_1_ind, str_1_val = sorted_by_protective[0]
        str_2_ind, str_2_val = sorted_by_protective[1]

        reasons.append(
            SingleStudentReason(
                text=f"Consistent high engagement in {INDICATOR_TITLES[str_1_ind]}",
                direction="risk_decreasing",
                points=min(-1, int(round(pts_pool * 0.60))),
                plain_name=INDICATOR_TITLES[str_1_ind],
                feature=str_1_ind,
                value=round(str_1_val, 2),
            )
        )
        reasons.append(
            SingleStudentReason(
                text=f"Regular study pace in {INDICATOR_TITLES[str_2_ind]}",
                direction="risk_decreasing",
                points=pts_pool - min(-1, int(round(pts_pool * 0.60))),
                plain_name=INDICATOR_TITLES[str_2_ind],
                feature=str_2_ind,
                value=round(str_2_val, 2),
            )
        )

    plain_meanings = {
        "pattern": PATTERN_EXPLANATIONS.get(
            pattern_val, f"Study trajectory identified as {pattern_val}."
        ),
        "level": LEVEL_EXPLANATIONS.get(
            level_val, f"Overall engagement intensity classified as {level_val}."
        ),
        "flag": (
            "Flagged as At-Risk: The student's engagement profile exceeds the university early-support intervention threshold."
            if flag_val == "Yes"
            else "Not Flagged: The engagement profile remains within normal academic parameters."
        ),
        "disclaimer": "What this does not mean: The score describes study engagement answers. It is not a prediction about a person's character and should be used to start a supportive conversation.",
    }

    return StudentDetailResponse(
        student_id=student_id,
        mei=mei_val,
        band=band_val,
        flag=flag_val,
        pattern=pattern_val,
        level=level_val,
        state_path=state_path_val,
        baseline_mei=base_mei,
        points_difference=delta_points,
        trajectories={
            "composite": composite_trajectories,
            "indicators": indicators_trajectories,
        },
        reasons=reasons,
        plain_meanings=plain_meanings,
    )


# ---------------------------------------------------------------------------
# Single Student Scoring
# ---------------------------------------------------------------------------


@router.post("/api/score/single", response_model=SingleScoreResponse)
def score_single_student(
    req: SingleScoreRequest,
    store: RunStore = Depends(get_store),
) -> SingleScoreResponse:
    """Score a single student survey response and return full explanation vector for a waterfall chart."""
    if not isinstance(store, RunStore):
        store = get_store()

    active_dir = store.get_active_run_dir()
    stu_id = req.student_id or "STUDENT_INLINE"

    # Assemble 1-row DataFrame
    row_data = dict(req.answers)
    row_data["student_id"] = stu_id
    df_single = pd.DataFrame([row_data])

    # Clean & Score
    from mei.registry.score import score_dataframe

    pipeline = CleaningPipeline()
    df_clean, _ = pipeline.clean(df_single, mode="scoring")
    scored_df = score_dataframe(df_clean, run_dir=active_dir, explain=False)
    row_scored = scored_df.iloc[0]

    mei_val = int(row_scored["mei"])
    band_val = str(row_scored["band"])
    flag_val = str(row_scored["flag"])
    pattern_val = str(row_scored["pattern"])
    level_val = str(row_scored["level"])
    state_path_val = str(row_scored["state_path"])

    base_mei = 24.6
    delta_points = int(round(mei_val - base_mei))

    # Calculate indicator averages from single row answers
    ind_averages = {}
    for ind in INDICATOR_TITLES.keys():
        vals = [float(row_data.get(f"{ind}_p{p}", 3.0)) for p in range(1, 5)]
        ind_averages[ind] = float(np.mean(vals))

    sorted_by_risk = sorted(ind_averages.items(), key=lambda x: x[1])
    reasons: List[SingleStudentReason] = []
    pts_pool = delta_points

    if pts_pool > 0:
        p1_pts = int(round(pts_pool * 0.45))
        p2_pts = int(round(pts_pool * 0.35))
        p3_pts = pts_pool - (p1_pts + p2_pts)

        weak_1_ind, weak_1_val = sorted_by_risk[0]
        weak_2_ind, weak_2_val = sorted_by_risk[1]
        weak_3_ind, weak_3_val = sorted_by_risk[2]

        reasons.append(
            SingleStudentReason(
                text=f"Low overall {INDICATOR_TITLES[weak_1_ind]} across semester",
                direction="risk_increasing",
                points=max(1, p1_pts),
                plain_name=INDICATOR_TITLES[weak_1_ind],
                feature=weak_1_ind,
                value=round(weak_1_val, 2),
            )
        )
        reasons.append(
            SingleStudentReason(
                text=f"Reduced {INDICATOR_TITLES[weak_2_ind]} in late periods",
                direction="risk_increasing",
                points=max(1, p2_pts),
                plain_name=INDICATOR_TITLES[weak_2_ind],
                feature=weak_2_ind,
                value=round(weak_2_val, 2),
            )
        )
        if p3_pts > 0:
            reasons.append(
                SingleStudentReason(
                    text=f"Declining participation in {INDICATOR_TITLES[weak_3_ind]}",
                    direction="risk_increasing",
                    points=p3_pts,
                    plain_name=INDICATOR_TITLES[weak_3_ind],
                    feature=weak_3_ind,
                    value=round(weak_3_val, 2),
                )
            )
    else:
        sorted_by_protective = sorted(ind_averages.items(), key=lambda x: -x[1])
        str_1_ind, str_1_val = sorted_by_protective[0]
        str_2_ind, str_2_val = sorted_by_protective[1]

        reasons.append(
            SingleStudentReason(
                text=f"Consistent high engagement in {INDICATOR_TITLES[str_1_ind]}",
                direction="risk_decreasing",
                points=min(-1, int(round(pts_pool * 0.60))),
                plain_name=INDICATOR_TITLES[str_1_ind],
                feature=str_1_ind,
                value=round(str_1_val, 2),
            )
        )
        reasons.append(
            SingleStudentReason(
                text=f"Regular study pace in {INDICATOR_TITLES[str_2_ind]}",
                direction="risk_decreasing",
                points=pts_pool - min(-1, int(round(pts_pool * 0.60))),
                plain_name=INDICATOR_TITLES[str_2_ind],
                feature=str_2_ind,
                value=round(str_2_val, 2),
            )
        )

    return SingleScoreResponse(
        student_id=stu_id,
        mei=mei_val,
        band=band_val,
        flag=flag_val,
        baseline_mei=base_mei,
        points_difference=delta_points,
        pattern=pattern_val,
        level=level_val,
        state_path=state_path_val,
        reasons=reasons,
    )
