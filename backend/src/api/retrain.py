"""
Model Retraining & Isolation Engine for Academic Intelligence Framework.
Coordinates:
  - Outcome detection and dataset validation (leakage, class balance, minimum cohort)
  - Isolated execution environment preventing overwrites to thesis synthetic benchmarks
  - Asynchronous background execution with live progress reporting
  - Small-data safety adjustments and multi-criteria champion evaluation

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import HTTPException, status

from backend.src.config import config
from backend.src.api.validator import (
    COLUMN_SYNONYMS,
    OUTCOME_SYNONYMS,
    is_wide_format,
    convert_wide_to_long,
    normalize_column_name,
)

logger = logging.getLogger("AcademicIntelligence.Retrain")

CHAMPIONS_SCOPE_MODELS = [
    "catboost",
    "gru",
    "logistic_elasticnet",
    "random_forest",
    "lightgbm",
]


def has_outcomes(df: pd.DataFrame) -> bool:
    """
    Returns True if the dataframe contains an outcome/target column
    matching 'attrited' or any recognized synonym.
    """
    for col in df.columns:
        norm = normalize_column_name(col)
        if norm == "attrited" or norm in OUTCOME_SYNONYMS["attrited"]:
            return True
    return False


def _find_column_by_canonical(df: pd.DataFrame, canonical_key: str) -> Optional[str]:
    """Finds raw column name matching a canonical synonym key."""
    syns = OUTCOME_SYNONYMS.get(canonical_key, []) + COLUMN_SYNONYMS.get(canonical_key, [])
    for col in df.columns:
        cleaned = re.sub(r"[^a-z0-9]", "_", str(col).strip().lower()).strip("_")
        if cleaned == canonical_key or cleaned in syns:
            return col
    return None


def extract_and_validate_outcomes(
    df_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Processes uploaded raw dataframe (long or wide format), extracts observation records
    and student-level outcomes, checks for data leakage and validity.

    Returns:
      (df_long_clean, df_outcomes, blocking_issues, warnings)
    """
    blocking_issues: List[str] = []
    warnings: List[str] = []

    # 1. Convert wide format if detected
    if is_wide_format(df_raw):
        try:
            df_long, conv_notes = convert_wide_to_long(df_raw)
            warnings.extend(conv_notes)
        except Exception as e:
            blocking_issues.append(f"Failed to convert wide format table: {e}")
            return pd.DataFrame(), pd.DataFrame(), blocking_issues, warnings
    else:
        df_long = df_raw.copy()

    # 2. Identify core columns
    sid_col = _find_column_by_canonical(df_long, "student_id")
    sem_col = _find_column_by_canonical(df_long, "semester")
    att_col = _find_column_by_canonical(df_long, "attrited")
    att_sem_col = _find_column_by_canonical(df_long, "attrition_semester")
    cohort_col = _find_column_by_canonical(df_long, "cohort_year")

    if sid_col is None:
        blocking_issues.append("Missing mandatory 'student_id' column.")
    if sem_col is None:
        blocking_issues.append("Missing mandatory 'semester' column.")
    if att_col is None:
        blocking_issues.append("Missing mandatory outcome column ('attrited' or synonyms: attrition, dropout, label).")

    if blocking_issues:
        return pd.DataFrame(), pd.DataFrame(), blocking_issues, warnings

    assert sid_col is not None and sem_col is not None and att_col is not None

    # Rename canonical identifiers
    df_long = df_long.rename(columns={
        sid_col: "student_id",
        sem_col: "semester",
        att_col: "attrited_raw",
    })

    if att_sem_col:
        df_long = df_long.rename(columns={att_sem_col: "attrition_semester_raw"})
    if cohort_col:
        df_long = df_long.rename(columns={cohort_col: "cohort_year_raw"})

    # Normalize metrics
    for m in ["gpa", "exam_marks", "attendance_pct", "assignment_delay_days", "late_assignments_count"]:
        col_m = _find_column_by_canonical(df_long, m)
        if col_m and col_m != m:
            df_long = df_long.rename(columns={col_m: m})

    # Ensure required numeric columns exist (fill with reasonable defaults if missing)
    if "gpa" not in df_long.columns:
        df_long["gpa"] = 3.0
        warnings.append("GPA column not found; defaulted to 3.0.")
    if "exam_marks" not in df_long.columns:
        df_long["exam_marks"] = 70.0
        warnings.append("Exam marks column not found; defaulted to 70.0.")
    if "attendance_pct" not in df_long.columns:
        df_long["attendance_pct"] = 80.0
        warnings.append("Attendance percentage column not found; defaulted to 80.0.")
    if "assignment_delay_days" not in df_long.columns:
        df_long["assignment_delay_days"] = 1.0
        warnings.append("Assignment delay days column not found; defaulted to 1.0.")
    if "late_assignments_count" not in df_long.columns:
        df_long["late_assignments_count"] = 0
        warnings.append("Late assignments count column not found; defaulted to 0.")

    # 3. Validate Student-Level Outcomes
    outcomes_rows: List[Dict[str, Any]] = []
    leakage_students: List[str] = []
    invalid_label_students: List[str] = []
    missing_att_sem_count = 0

    student_groups = df_long.groupby("student_id")
    for sid, group in student_groups:
        s_sid = str(sid).strip()
        
        # Determine outcome label
        raw_vals = group["attrited_raw"].dropna().unique()
        if len(raw_vals) == 0:
            invalid_label_students.append(s_sid)
            continue
        
        # Convert label to integer 0 or 1
        try:
            label_int = int(round(float(raw_vals[0])))
            if label_int not in (0, 1):
                invalid_label_students.append(s_sid)
                continue
        except Exception:
            invalid_label_students.append(s_sid)
            continue

        if len(raw_vals) > 1 and len(set(int(round(float(v))) for v in raw_vals)) > 1:
            warnings.append(f"Student '{s_sid}' had conflicting outcome labels across rows; used majority/latest label {label_int}.")

        # Determine last semester observed
        try:
            sems = pd.to_numeric(group["semester"], errors="coerce").dropna().astype(int).tolist()
            last_sem = max(sems) if sems else 1
        except Exception:
            last_sem = 1

        # Determine attrition semester
        attrition_sem = None
        if "attrition_semester_raw" in group.columns:
            att_sem_vals = group["attrition_semester_raw"].dropna().unique()
            if len(att_sem_vals) > 0 and pd.notna(att_sem_vals[0]):
                try:
                    attrition_sem = float(att_sem_vals[0])
                except Exception:
                    attrition_sem = None

        if label_int == 1:
            if attrition_sem is None:
                attrition_sem = float(last_sem + 1)
                missing_att_sem_count += 1
            else:
                # Check Data Leakage: rows after recorded attrition semester
                post_attrition_rows = [s for s in sems if s > attrition_sem]
                if post_attrition_rows:
                    leakage_students.append(s_sid)
        else:
            attrition_sem = None

        # Determine cohort year
        cohort_yr = 2024
        if "cohort_year_raw" in group.columns:
            cy_vals = group["cohort_year_raw"].dropna().unique()
            if len(cy_vals) > 0:
                try:
                    cohort_yr = int(float(cy_vals[0]))
                except Exception:
                    cohort_yr = 2024

        outcomes_rows.append({
            "student_id": s_sid,
            "cohort_year": cohort_yr,
            "attrited": label_int,
            "last_semester_observed": last_sem,
            "attrition_semester": attrition_sem,
        })

    if invalid_label_students:
        blocking_issues.append(
            f"Invalid outcome labels detected for {len(invalid_label_students)} student(s). "
            "Attrition outcomes must be binary values (0 = enrolled/graduated, 1 = attrited/dropped out)."
        )

    if leakage_students:
        blocking_issues.append(
            f"Data leakage detected for {len(leakage_students)} student(s): "
            "Observations were recorded for semesters occurring AFTER the student's documented attrition semester."
        )

    if missing_att_sem_count > 0:
        warnings.append(
            f"Attrition semester was unspecified for {missing_att_sem_count} attrited student(s); "
            "defaulted to (last observed semester + 1)."
        )

    if "cohort_year_raw" not in df_long.columns:
        warnings.append("Cohort year column was not provided; defaulted to 2024 for all students.")

    df_outcomes = pd.DataFrame(outcomes_rows)

    # Clean df_long columns
    cols_to_keep = [
        "student_id", "semester", "gpa", "exam_marks",
        "attendance_pct", "assignment_delay_days", "late_assignments_count"
    ]
    df_long_clean = df_long[[c for c in cols_to_keep if c in df_long.columns]].copy()
    df_long_clean["student_id"] = df_long_clean["student_id"].astype(str)
    df_long_clean["semester"] = pd.to_numeric(df_long_clean["semester"], errors="coerce").fillna(1).astype(int)

    # Filter out any records that exceed student's last_semester_observed to prevent corrupted samples
    return df_long_clean, df_outcomes, blocking_issues, warnings


def inspect_dataset_for_retrain(df_raw: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates uploaded data against criteria for model retraining.
    Returns schema-compliant dictionary for POST /train/inspect.
    """
    if not has_outcomes(df_raw):
        n_students = df_raw["student_id"].nunique() if "student_id" in df_raw.columns else len(df_raw)
        return {
            "has_outcomes": False,
            "n_students": int(n_students),
            "n_records": len(df_raw),
            "n_attrited": 0,
            "attrition_rate": 0.0,
            "can_retrain": False,
            "blocking_issues": [
                "No outcome column found in uploaded file (expected 'attrited' or synonyms: 'attrition', 'dropout', 'dropped_out', 'withdrawn', 'label')."
            ],
            "warnings": [],
        }

    df_long, df_outcomes, blocking_issues, warnings = extract_and_validate_outcomes(df_raw)

    n_students = len(df_outcomes)
    n_records = len(df_long)
    n_attrited = int(df_outcomes["attrited"].sum()) if n_students > 0 else 0
    n_non_attrited = n_students - n_attrited
    attrition_rate = float(n_attrited / n_students) if n_students > 0 else 0.0

    # Rule checks
    if n_students < 100:
        blocking_issues.append(
            f"Insufficient student cohort size: uploaded dataset has {n_students} students (minimum 100 required for stable training)."
        )

    if n_attrited < 20:
        blocking_issues.append(
            f"Insufficient positive outcome cases: uploaded dataset has {n_attrited} attrited student(s) (minimum 20 required)."
        )

    if n_non_attrited < 20:
        blocking_issues.append(
            f"Insufficient negative outcome cases: uploaded dataset has {n_non_attrited} non-attrited student(s) (minimum 20 required)."
        )

    # Advisory warnings
    if 100 <= n_students < 300:
        warnings.append(
            f"Cohort size ({n_students}) is below the recommended 300 students for high-confidence multi-semester model convergence."
        )

    if n_students > 0 and (attrition_rate < 0.05 or attrition_rate > 0.60):
        warnings.append(
            f"Extreme class distribution: attrition rate is {attrition_rate:.1%} (recommended baseline is between 5% and 60%)."
        )

    can_retrain = len(blocking_issues) == 0

    return {
        "has_outcomes": True,
        "n_students": n_students,
        "n_records": n_records,
        "n_attrited": n_attrited,
        "attrition_rate": round(attrition_rate, 4),
        "can_retrain": can_retrain,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
    }


class RetrainJobManager:
    """Singleton manager controlling asynchronous isolated retraining jobs."""

    _instance: Optional[RetrainJobManager] = None
    _lock = threading.Lock()

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._active_job_id: Optional[str] = None

    @classmethod
    def get_instance(cls) -> RetrainJobManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = RetrainJobManager()
            return cls._instance

    @property
    def active_job_id(self) -> Optional[str]:
        return self._active_job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                # Update elapsed time if still active
                if job["state"] in ("queued", "running"):
                    job["elapsed_s"] = round(time.time() - job["start_time"], 1)
                return dict(job)

        # Check on-disk persistence if not in memory
        disk_path = config.root_dir / "artifacts" / "user_dataset" / job_id / "job_status.json"
        if disk_path.exists():
            try:
                with open(disk_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def submit_job(
        self,
        df_long: pd.DataFrame,
        df_outcomes: pd.DataFrame,
        budget: str = "quick",
        scope: str = "champions",
        upload_bytes: Optional[bytes] = None,
        upload_filename: Optional[str] = None,
    ) -> str:
        with self._lock:
            if self._active_job_id is not None:
                current_active = self._jobs.get(self._active_job_id)
                if current_active and current_active["state"] in ("queued", "running"):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=(
                            f"Another retraining job ({self._active_job_id}) is currently in progress. "
                            "Please wait for it to finish or check /jobs/{id} before starting a new run."
                        ),
                    )

            job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            self._active_job_id = job_id
            self._jobs[job_id] = {
                "job_id": job_id,
                "state": "queued",
                "phase": "validate",
                "progress_pct": 5,
                "message": "Retraining job queued. Initializing isolated directory workspace...",
                "current_model": None,
                "models_done": 0,
                "models_total": 0,
                "elapsed_s": 0.0,
                "start_time": time.time(),
                "result": None,
                "error": None,
            }

        # Save local-only upload copy if provided
        if upload_bytes:
            try:
                upload_dir = config.root_dir / "data" / "uploads" / job_id
                upload_dir.mkdir(parents=True, exist_ok=True)
                fname = upload_filename or "uploaded_dataset.csv"
                with open(upload_dir / fname, "wb") as f:
                    f.write(upload_bytes)
            except Exception as e:
                logger.warning(f"Could not persist upload backup to data/uploads/{job_id}: {e}")

        # Spawn worker thread
        t = threading.Thread(
            target=self._run_job_worker,
            args=(job_id, df_long, df_outcomes, budget, scope),
            daemon=True,
            name=f"RetrainWorker-{job_id}",
        )
        t.start()
        return job_id

    def _persist_job_state(self, job_id: str, state_dict: Dict[str, Any]) -> None:
        try:
            job_dir = config.root_dir / "artifacts" / "user_dataset" / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            with open(job_dir / "job_status.json", "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to persist job status to disk: {e}")

    def _update_job(self, job_id: str, **kwargs) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)
                if self._jobs[job_id]["state"] in ("queued", "running"):
                    self._jobs[job_id]["elapsed_s"] = round(
                        time.time() - self._jobs[job_id]["start_time"], 1
                    )
                self._persist_job_state(job_id, self._jobs[job_id])

    def _run_job_worker(
        self,
        job_id: str,
        df_long: pd.DataFrame,
        df_outcomes: pd.DataFrame,
        budget: str,
        scope: str,
    ) -> None:
        logger.info(f"Retrain worker started for job_id={job_id} | Budget={budget} | Scope={scope}")
        self._update_job(
            job_id,
            state="running",
            phase="splits",
            progress_pct=10,
            message="Partitioning student cohort with zero-leakage student grouping...",
        )

        try:
            from backend.src.data.task import create_prefix_task_samples
            from backend.src.features.pipeline import run_feature_pipeline
            from backend.src.training.orchestrator import run_training_pipeline
            from backend.src.evaluation.evaluator import run_full_evaluation
            from backend.src.evaluation.leaderboard import build_leaderboard
            from backend.src.explain.orchestrator import run_explainability_pipeline

            with config.isolated_run(job_id) as out_dir:
                # 1. Determine small-data safety parameters
                n_students = len(df_outcomes)
                max_sem = int(df_long["semester"].max()) if not df_long.empty else 6
                effective_max_window = min(6, max_sem)
                min_window = 2 if max_sem >= 2 else 1

                if n_students < 150:
                    n_cv_folds = 2
                    test_ratio = 0.15
                    val_ratio = 0.15
                elif n_students < 300:
                    n_cv_folds = 3
                    test_ratio = 0.15
                    val_ratio = 0.15
                else:
                    n_cv_folds = 5
                    test_ratio = 0.15
                    val_ratio = 0.15

                # 2. Build task prefix samples
                self._update_job(
                    job_id,
                    phase="features",
                    progress_pct=15,
                    message="Extracting multi-horizon observation prefix task windows...",
                )
                df_task = create_prefix_task_samples(
                    df_long,
                    df_outcomes,
                    min_window=min_window,
                    max_window=effective_max_window,
                )

                if df_task.empty:
                    raise ValueError(
                        f"Unable to generate prefix task samples. Dataset has maximum semester={max_sem}, "
                        "which is insufficient to construct multi-semester longitudinal prediction windows."
                    )

                # Save raw copies to isolated directory for complete audit provenance
                df_long.to_csv(out_dir / "academic_long.csv", index=False)
                df_outcomes.to_csv(out_dir / "outcomes.csv", index=False)
                df_task.to_csv(out_dir / "task_samples.csv", index=False)

                # 3. Feature Pipeline
                self._update_job(
                    job_id,
                    phase="features",
                    progress_pct=25,
                    message="Fitting AcademicPreprocessor and generating dual-paradigm feature tensors...",
                )
                run_feature_pipeline(
                    df_long=df_long,
                    df_outcomes=df_outcomes,
                    df_task=df_task,
                    seed=config.global_seed,
                    max_window=effective_max_window,
                    train_ratio=1.0 - (test_ratio + val_ratio),
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    n_cv_folds=n_cv_folds,
                    evaluate_static_baseline=False,
                )

                # 4. Model Training
                self._update_job(
                    job_id,
                    phase="training",
                    progress_pct=35,
                    message="Starting hyperparameter tuning and model training...",
                )

                target_models = CHAMPIONS_SCOPE_MODELS if scope == "champions" else None

                def training_progress(info: Dict[str, Any]):
                    m_done = info.get("models_done", 0)
                    m_total = info.get("models_total", 1)
                    curr_m = info.get("current_model", "")
                    pct = int(35 + 40 * (m_done / max(m_total, 1)))
                    self._update_job(
                        job_id,
                        phase="training",
                        progress_pct=min(75, pct),
                        current_model=curr_m,
                        models_done=m_done,
                        models_total=m_total,
                        message=f"Training {curr_m} ({m_done}/{m_total} models completed)...",
                    )

                run_training_pipeline(
                    budget=budget,
                    target_models=target_models,
                    progress_callback=training_progress,
                )

                # 5. One-Touch Test Set Evaluation & Leaderboard
                self._update_job(
                    job_id,
                    phase="evaluation",
                    progress_pct=78,
                    message="Running final test-set evaluation and composite leaderboard ranking...",
                )
                eval_results = run_full_evaluation()
                lb_results = build_leaderboard(eval_results)

                # 6. Explainability Layer
                self._update_job(
                    job_id,
                    phase="explainability",
                    progress_pct=90,
                    message="Computing TreeSHAP feature attributions and Academic Risk Index...",
                )
                run_explainability_pipeline()

                # 7. Extract Champion Summary & Save latest.json
                champ_key = lb_results.get("champion", "catboost")
                champ_model_meta = None
                for m in lb_results.get("models", []):
                    if m.get("model_key") == champ_key:
                        champ_model_meta = m
                        break

                if not champ_model_meta and lb_results.get("models"):
                    champ_model_meta = lb_results["models"][0]
                    champ_key = champ_model_meta["model_key"]

                test_roc = float(champ_model_meta.get("test_roc_auc", 0.0)) if champ_model_meta else 0.0
                test_pr = float(champ_model_meta.get("test_pr_auc", 0.0)) if champ_model_meta else 0.0
                test_f2 = float(champ_model_meta.get("test_f2_score", 0.0)) if champ_model_meta else 0.0
                lead_time = float(champ_model_meta.get("test_lead_time", 2.0)) if champ_model_meta else 2.0

                seq_champ = eval_results.get("key_models", {}).get("sequence_champion", "gru")

                latest_meta = {
                    "job_id": job_id,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "n_students": n_students,
                    "n_attrited": int(df_outcomes["attrited"].sum()),
                    "champion": champ_key,
                    "champion_label": champ_model_meta.get("label", champ_key) if champ_model_meta else champ_key,
                    "sequence_champion": seq_champ,
                    "test_roc_auc": round(test_roc, 4),
                    "test_pr_auc": round(test_pr, 4),
                    "test_f2_score": round(test_f2, 4),
                    "lead_time_semesters": round(lead_time, 1),
                    "budget": budget,
                    "scope": scope,
                    "data_mode": "REAL",
                }

                config.save_latest_user_dataset_metadata(latest_meta)

                # Save local copy inside job folder too
                with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
                    json.dump(latest_meta, f, indent=2)

                self._update_job(
                    job_id,
                    state="succeeded",
                    phase=None,
                    progress_pct=100,
                    message="Retraining on user data completed successfully. Isolated benchmark ready.",
                    result=latest_meta,
                    error=None,
                )
                logger.info(f"Retrain job {job_id} succeeded! Champion: {champ_key} (ROC-AUC={test_roc:.4f})")

        except Exception as e:
            logger.error(f"Retraining job {job_id} failed with error: {e}", exc_info=True)
            self._update_job(
                job_id,
                state="failed",
                progress_pct=100,
                message=f"Retraining job failed: {e}",
                error=str(e),
            )
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None
