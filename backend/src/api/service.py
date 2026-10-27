"""
Inference and Analytics Service for Academic Intelligence API.
Caches and coordinates:
  - Preprocessor (Joblib)
  - Champion Model (CatBoost)
  - Platt Calibrator
  - Academic Risk Index & Clinical Banding Engine
  - SHAP TreeExplainer
  - Deterministic Diagnostic Narrative Generator
  - Fusion Framework Contract Builder

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import io
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from backend.src.config import config
from backend.src.models.base import ModelRegistry
import backend.src.models  # Trigger model registrations
from backend.src.features.builder import build_tabular_engineered_features
from backend.src.explain.risk_index import AcademicRiskIndexEngine, DEFAULT_RISK_BANDS
from backend.src.explain.shap_explainer import TabularTreeSHAPExplainer, FEATURE_METADATA
from backend.src.explain.sentence_generator import generate_explanation_narrative
from backend.src.explain.fusion_contract import FusionAcademicRiskPayload, FactorContribution

logger = logging.getLogger("AcademicIntelligence.ApiService")


class InferenceService:
    """
    Thread-safe, high-performance inference and analytics coordinator.
    Lazy-loads and caches model artifacts for sub-50ms inference latency.
    """

    _instance: Optional["InferenceService"] = None

    def __init__(self):
        self._is_initialized = False
        self.model_champ: Any = None
        self.calibrator: Any = None
        self.preprocessor: Any = None
        self.explainer: Optional[TabularTreeSHAPExplainer] = None
        self.risk_engine: AcademicRiskIndexEngine = AcademicRiskIndexEngine(DEFAULT_RISK_BANDS)
        self.feature_cols: List[str] = []
        
        # In-memory download cache for batch exports (token -> {csv, json, timestamp})
        self._export_cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_instance(cls) -> "InferenceService":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.initialize()
        return cls._instance

    def initialize(self) -> None:
        """Loads preprocessor, champion model, calibrator, and SHAP explainer."""
        if self._is_initialized:
            return

        logger.info("Initializing Academic Intelligence Inference Service...")
        t0 = time.perf_counter()

        # 1. Load Preprocessor
        preproc_path = config.models_dir / "preprocessor.joblib"
        if not preproc_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {preproc_path}")
        self.preprocessor = joblib.load(preproc_path)

        # 2. Load Champion Model (CatBoost)
        catboost_dir = config.models_dir / "Category_C" / "catboost"
        if not catboost_dir.exists():
            raise FileNotFoundError(f"Champion model directory not found at {catboost_dir}")
        self.model_champ = ModelRegistry.load_model("catboost", catboost_dir)

        # 3. Load Calibrator
        calibrator_path = catboost_dir / "calibrator.joblib"
        if not calibrator_path.exists():
            raise FileNotFoundError(f"Calibrator not found at {calibrator_path}")
        self.calibrator = joblib.load(calibrator_path)

        # 4. Feature Columns Specification
        df_tab = pd.read_csv(config.processed_data_dir / "tabular_features.csv", nrows=5)
        meta_cols = ["sample_id", "student_id", "window_t", "target"]
        self.feature_cols = [c for c in df_tab.columns if c not in meta_cols]

        # 5. Initialize TreeSHAP Explainer
        self.explainer = TabularTreeSHAPExplainer(self.model_champ, self.feature_cols)

        self._is_initialized = True
        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(f"Inference Service successfully initialized in {elapsed:.2f} ms")

    # ── Single Student Prediction ─────────────────────────────────────────────

    def predict_single_student(
        self,
        student_id: str,
        semesters_data: List[Dict[str, Any]],
        warnings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference and explainability on a single student sequence.
        Returns complete diagnostic payload, calibrated risk index, and fusion contract.
        """
        self.initialize()
        t0 = time.perf_counter()

        # Build raw DataFrame for single student
        rows = []
        for s in semesters_data:
            rows.append({
                "student_id": student_id,
                "semester": int(s["semester"]),
                "gpa": s.get("gpa"),
                "exam_marks": s.get("exam_marks"),
                "attendance_pct": s.get("attendance_pct"),
                "assignment_delay_days": s.get("assignment_delay_days"),
                "late_assignments_count": s.get("late_assignments_count"),
            })
        df_raw = pd.DataFrame(rows).sort_values("semester").reset_index(drop=True)

        # Transform using saved zero-leakage preprocessor
        df_clean = self.preprocessor.transform(df_raw)

        # Evaluate risk at the latest available semester
        latest_sem = int(df_clean["semester"].max())
        sample_id = f"{student_id}_t{latest_sem}"
        df_task = pd.DataFrame([{
            "sample_id": sample_id,
            "student_id": student_id,
            "window_t": latest_sem,
            "target_attrite_next_2_sems": 0,
        }])

        # Extract engineered trend features
        df_feat = build_tabular_engineered_features(df_clean, df_task)
        X = df_feat[self.feature_cols].values.astype(np.float32)

        # Predict raw & calibrated probability
        raw_prob = float(self.model_champ.predict_proba(X)[0])
        cal_prob = float(self.calibrator.transform(np.array([raw_prob]))[0])

        # Compute Academic Risk Index & Clinical Band
        ari = float(self.risk_engine.compute_risk_index(np.array([cal_prob]))[0])
        bands = self.risk_engine.batch_assign_bands(np.array([ari]))
        band_name = bands[0].name

        # SHAP local attribution
        self.explainer.explain(X)
        local_exp = self.explainer.get_local_explanation(sample_idx=0, top_k=3)

        factors = [
            FactorContribution(
                name=f["name"],
                display_name=f["display_name"],
                direction=f["direction"],
                contribution=f["contribution"],
                feature_value=f["feature_value"],
                unit=f["unit"],
                narrative=f["display_name"]
            )
            for f in local_exp["top_factors"]
        ]

        # Plain language diagnostic narrative
        raw_feat_dict = df_feat.iloc[0].to_dict()
        narrative = generate_explanation_narrative(
            ari,
            band_name,
            local_exp["top_factors"],
            raw_feat_dict
        )

        # Fusion Framework Contract
        fusion_payload = FusionAcademicRiskPayload(
            participant_id=student_id,
            sample_id=sample_id,
            window_semester=latest_sem,
            academic_risk_index=round(ari, 2),
            risk_band=band_name,
            top_factors=factors,
            model_name="catboost",
            model_version="1.0.0",
            calibrated=True,
            explanation_text=narrative,
            provenance_tag="SYNTHETIC"
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        nfr1_ok = elapsed_ms < 2000.0  # < 2 seconds NFR1

        return {
            "student_id": student_id,
            "sample_id": sample_id,
            "window_semester": latest_sem,
            "academic_risk_index": round(ari, 2),
            "risk_band": band_name,
            "calibrated_probability": round(cal_prob, 4),
            "top_factors": [f.model_dump() for f in factors],
            "explanation_text": narrative,
            "fusion_payload": fusion_payload.model_dump(),
            "inference_time_ms": round(elapsed_ms, 2),
            "nfr1_compliant": nfr1_ok,
            "warnings": warnings or [],
        }

    # ── Batch Upload Prediction ───────────────────────────────────────────────

    def predict_batch(
        self,
        df_long: pd.DataFrame,
        warnings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes batch inference across multi-student cohort data.
        Evaluates each student at their latest recorded semester window.
        """
        self.initialize()
        t0 = time.perf_counter()

        # Transform using saved zero-leakage preprocessor
        df_clean = self.preprocessor.transform(df_long)

        # Identify latest semester per student
        latest_sems = df_clean.groupby("student_id")["semester"].max().reset_index()
        df_tasks = pd.DataFrame({
            "sample_id": latest_sems["student_id"] + "_t" + latest_sems["semester"].astype(str),
            "student_id": latest_sems["student_id"],
            "window_t": latest_sems["semester"],
            "target_attrite_next_2_sems": 0,
        })

        # Build engineered tabular features
        df_feat = build_tabular_engineered_features(df_clean, df_tasks)
        X = df_feat[self.feature_cols].values.astype(np.float32)

        # Batch predict & calibrate
        raw_probs = self.model_champ.predict_proba(X)
        cal_probs = self.calibrator.transform(raw_probs)
        ari_scores = self.risk_engine.compute_risk_index(cal_probs)
        bands = self.risk_engine.batch_assign_bands(ari_scores)

        # Batch TreeSHAP attributions
        self.explainer.explain(X)

        predictions: List[Dict[str, Any]] = []
        band_counts = {"Low": 0, "Moderate": 0, "High": 0, "Critical": 0}

        for idx, sid in enumerate(df_feat["student_id"]):
            ari = float(ari_scores[idx])
            band_name = bands[idx].name
            band_counts[band_name] = band_counts.get(band_name, 0) + 1

            local_exp = self.explainer.get_local_explanation(sample_idx=idx, top_k=3)
            factors = [
                FactorContribution(
                    name=f["name"],
                    display_name=f["display_name"],
                    direction=f["direction"],
                    contribution=f["contribution"],
                    feature_value=f["feature_value"],
                    unit=f["unit"],
                    narrative=f["display_name"]
                )
                for f in local_exp["top_factors"]
            ]

            raw_feat_dict = df_feat.iloc[idx].to_dict()
            narrative = generate_explanation_narrative(
                ari,
                band_name,
                local_exp["top_factors"],
                raw_feat_dict
            )

            sample_id = str(df_feat.iloc[idx]["sample_id"])
            window_sem = int(df_feat.iloc[idx]["window_t"])

            fusion_payload = FusionAcademicRiskPayload(
                participant_id=sid,
                sample_id=sample_id,
                window_semester=window_sem,
                academic_risk_index=round(ari, 2),
                risk_band=band_name,
                top_factors=factors,
                model_name="catboost",
                model_version="1.0.0",
                calibrated=True,
                explanation_text=narrative,
                provenance_tag="SYNTHETIC"
            )

            predictions.append({
                "student_id": sid,
                "sample_id": sample_id,
                "window_semester": window_sem,
                "academic_risk_index": round(ari, 2),
                "risk_band": band_name,
                "calibrated_probability": round(float(cal_probs[idx]), 4),
                "top_factors": [f.model_dump() for f in factors],
                "explanation_text": narrative,
                "fusion_payload": fusion_payload.model_dump(),
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_students = len(predictions)
        ms_per_student = elapsed_ms / max(n_students, 1)

        # Store in ephemeral export cache
        token = str(uuid.uuid4())
        self._cache_export(token, predictions)

        return {
            "success": True,
            "total_students": n_students,
            "total_records": len(df_long),
            "summary": {
                "total_students": n_students,
                "total_records": len(df_long),
                "average_risk_index": round(float(np.mean(ari_scores)), 2) if n_students > 0 else 0.0,
                "risk_band_counts": band_counts,
            },
            "predictions": predictions,
            "download_token": token,
            "download_url_csv": f"/predict/download/{token}?format=csv",
            "download_url_json": f"/predict/download/{token}?format=json",
            "inference_time_total_ms": round(elapsed_ms, 2),
            "inference_time_per_student_ms": round(ms_per_student, 2),
            "nfr1_compliant": ms_per_student < 2000.0,
            "warnings": warnings or [],
        }

    # ── Export & Download ─────────────────────────────────────────────────────

    def _cache_export(self, token: str, predictions: List[Dict[str, Any]]) -> None:
        """Stores CSV and JSON serialized strings in memory."""
        # Clean older cache entries (keep last 50)
        if len(self._export_cache) > 50:
            oldest_keys = list(self._export_cache.keys())[:20]
            for k in oldest_keys:
                self._export_cache.pop(k, None)

        # 1. JSON
        json_str = json.dumps(predictions, indent=2)

        # 2. CSV
        flat_rows = []
        for p in predictions:
            f1 = p["top_factors"][0]["display_name"] if len(p["top_factors"]) > 0 else ""
            f2 = p["top_factors"][1]["display_name"] if len(p["top_factors"]) > 1 else ""
            f3 = p["top_factors"][2]["display_name"] if len(p["top_factors"]) > 2 else ""
            flat_rows.append({
                "student_id": p["student_id"],
                "sample_id": p["sample_id"],
                "window_semester": p["window_semester"],
                "academic_risk_index": p["academic_risk_index"],
                "risk_band": p["risk_band"],
                "calibrated_probability": p["calibrated_probability"],
                "top_factor_1": f1,
                "top_factor_2": f2,
                "top_factor_3": f3,
                "explanation_text": p["explanation_text"],
            })
        df_export = pd.DataFrame(flat_rows)
        csv_str = df_export.to_csv(index=False)

        self._export_cache[token] = {
            "csv": csv_str,
            "json": json_str,
            "timestamp": time.time(),
        }

    def get_export_data(self, token: str, format_type: str = "csv") -> Optional[str]:
        """Retrieves cached export string by token."""
        entry = self._export_cache.get(token)
        if not entry:
            return None
        return entry.get(format_type.lower())

    # ── Model & Leaderboard Metadata ──────────────────────────────────────────

    def get_leaderboard(self) -> Dict[str, Any]:
        """Loads leaderboard, category champions, and rankings."""
        lb_path = config.leaderboard_dir / "leaderboard.json"
        cat_champs_path = config.leaderboard_dir / "category_champions.json"
        
        with open(lb_path, "r", encoding="utf-8") as f:
            lb_data = json.load(f)
        with open(cat_champs_path, "r", encoding="utf-8") as f:
            cat_data = json.load(f)

        lb_data["category_champions"] = cat_data.get("categories", {})
        return lb_data

    def get_model_details(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Finds and compiles full model details from artifacts/models/*/{model_key}."""
        target_dir = None
        for cat_dir in config.models_dir.iterdir():
            if cat_dir.is_dir() and cat_dir.name.startswith("Category_"):
                candidate = cat_dir / model_key
                if candidate.is_dir():
                    target_dir = candidate
                    break

        if not target_dir:
            return None

        def _safe_load_json(file_path: Path) -> Dict[str, Any]:
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}

        summary = _safe_load_json(target_dir / "summary.json")
        best_params = _safe_load_json(target_dir / "best_params.json")
        training_curves = _safe_load_json(target_dir / "training_curves.json")
        trials = _safe_load_json(target_dir / "trials.json")
        threshold_config = _safe_load_json(target_dir / "threshold_config.json")
        seed_metrics = _safe_load_json(target_dir / "seed_metrics.json")

        return {
            "model_key": model_key,
            "category": summary.get("category", ""),
            "category_label": summary.get("category_label", ""),
            "label": summary.get("label", model_key),
            "is_proposal_model": summary.get("is_proposal_model", False),
            "is_sequence_model": summary.get("is_sequence_model", False),
            "status": summary.get("status", "COMPLETED"),
            "validation_metrics": summary.get("validation_metrics", {}),
            "stability_summary": summary.get("stability_summary", seed_metrics),
            "calibration": summary.get("calibration", {}),
            "threshold": threshold_config or summary.get("threshold", {}),
            "best_params": best_params,
            "training_curves": training_curves if isinstance(training_curves, list) else [],
            "trials_summary": trials if isinstance(trials, dict) else {},
            "runtime_seconds": summary.get("runtime_seconds", 0.0),
            "artifact_dir": str(target_dir),
        }

    def get_research_answer(self) -> Dict[str, Any]:
        """Loads formal research question adjudication from Phase 6."""
        path = config.leaderboard_dir / "research_question_answer.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_data_summary(self) -> Dict[str, Any]:
        """Returns cohort distribution statistics and plot catalogue."""
        plots_list = []
        plot_categories = ["data_audit", "evaluation", "explainability"]
        
        plot_descriptions = {
            "attrition_by_type.png": "Observed attrition proportions across 7 latent behavioral student archetypes.",
            "cohort_gpa_trends.png": "Longitudinal GPA evolution over 8 semesters for retained vs attrited students.",
            "correlation_heatmap.png": "Cross-correlation matrix among academic, attendance, and assignment velocity indicators.",
            "distributions.png": "Kernel density distributions for GPA, exam marks, attendance, and assignment delay.",
            "missingness_map.png": "Longitudinal missingness pattern and attrition cessation map.",
            "trajectories_by_type.png": "Trajectory profiles illustrating sudden drops, steady declines, and volatile students.",
            "calibration_curves.png": "Reliability diagrams comparing raw vs Platt calibrated probability estimates.",
            "confusion_matrices.png": "Confusion matrices at recall-constrained operating threshold (Recall >= 80%).",
            "lead_time_distribution.png": "Prognostic lead time distribution in semesters prior to academic termination.",
            "metric_heatmap.png": "Comprehensive performance matrix across all 28 registered models.",
            "metric_vs_complexity.png": "Pareto frontier analysis: predictive utility vs computational runtime.",
            "pr_curves_overlay.png": "Precision-Recall curves highlighting minority class detection quality.",
            "risk_index_distribution.png": "Distribution of continuous Academic Risk Index (0-100) across cohort.",
            "roc_curves_overlay.png": "ROC curves comparing sequence models, boosting baselines, and snapshot models.",
            "feature_deletion_faithfulness.png": "Empirical feature deletion test confirming monotonic risk index drop.",
            "risk_band_distribution.png": "Cohort population breakdown across Low, Moderate, High, and Critical risk bands.",
            "shap_summary_global.png": "Global mean absolute SHAP importance showing engagement features lead prognostic signal.",
            "temporal_attention_heatmap.png": "Sequence model temporal integrated gradients across semesters 1 to 6.",
            "waterfall_sample_high_risk.png": "Local SHAP waterfall attribution for an exemplary high-risk student."
        }

        for cat in plot_categories:
            p_dir = config.plots_dir / cat
            if p_dir.is_dir():
                for p_file in sorted(p_dir.glob("*.png")):
                    fname = p_file.name
                    plots_list.append({
                        "key": p_file.stem,
                        "title": p_file.stem.replace("_", " ").title(),
                        "category": cat,
                        "filename": fname,
                        "url_path": f"/static/plots/{cat}/{fname}",
                        "description": plot_descriptions.get(fname, "Academic Intelligence Framework visual asset.")
                    })

        return {
            "data_mode": "SYNTHETIC",
            "total_students": 5000,
            "total_observations": 21985,
            "prefix_prediction_samples": 21985,
            "base_attrition_rate_pct": 23.6,
            "num_semesters": 8,
            "gpa_scale": "0.0 - 4.0",
            "variables": [
                {"name": "gpa", "scale": "0.0 - 4.0", "description": "Semester Grade Point Average"},
                {"name": "exam_marks", "scale": "0 - 100%", "description": "Coursework & Examination Average"},
                {"name": "attendance_pct", "scale": "0 - 100%", "description": "Lecture & Tutorial Attendance Rate"},
                {"name": "assignment_delay_days", "scale": "0.0 - 45.0 days", "description": "Average Assignment Submission Delay"},
                {"name": "late_assignments_count", "scale": "0 - 15 count", "description": "Number of Late Assignment Submissions"}
            ],
            "trajectory_archetypes": [
                "stable-high", "stable-average", "slow-decline", "sudden-drop", 
                "recovering", "volatile", "struggling-from-start"
            ],
            "plots_catalog": plots_list
        }

    def get_global_explanations(self) -> Dict[str, Any]:
        """Loads global SHAP importance, semester attributions, and faithfulness metrics."""
        imp_path = config.explanations_dir / "global_feature_importance.csv"
        sem_path = config.explanations_dir / "semester_feature_attribution_matrix.csv"
        faith_path = config.explanations_dir / "faithfulness_deletion_results.json"
        stab_path = config.explanations_dir / "explanation_stability_results.json"

        df_imp = pd.read_csv(imp_path)
        df_sem = pd.read_csv(sem_path)
        
        with open(faith_path, "r", encoding="utf-8") as f:
            faith_data = json.load(f)
        with open(stab_path, "r", encoding="utf-8") as f:
            stab_data = json.load(f)

        return {
            "model_key": "catboost",
            "explainer_type": "TreeSHAP",
            "features": df_imp.to_dict(orient="records"),
            "semester_attributions": df_sem.to_dict(orient="records"),
            "faithfulness_audit": faith_data,
            "stability_audit": stab_data
        }

    def get_student_explanation_by_id(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Looks up precomputed test explanation for student_id or computes dynamically."""
        self.initialize()

        # 1. Look in fusion_payloads.json if present
        payloads_path = config.explanations_dir / "fusion_payloads.json"
        if payloads_path.exists():
            # Quick check if file is accessible
            with open(payloads_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                payloads = data.get("payloads", [])
                for p in payloads:
                    if p.get("participant_id") == student_id:
                        return {
                            "student_id": student_id,
                            "sample_id": p.get("sample_id"),
                            "window_semester": p.get("window_semester"),
                            "academic_risk_index": p.get("academic_risk_index"),
                            "risk_band": p.get("risk_band"),
                            "top_factors": p.get("top_factors", []),
                            "explanation_text": p.get("explanation_text"),
                            "model_name": p.get("model_name"),
                            "calibrated": p.get("calibrated", True),
                            "provenance_tag": p.get("provenance_tag", "SYNTHETIC"),
                            "timestamp_utc": p.get("timestamp_utc", datetime.now(timezone.utc).isoformat())
                        }

        # 2. Dynamic check in academic_long.csv
        long_path = config.synthetic_data_dir / "academic_long.csv"
        if long_path.exists():
            df_long = pd.read_csv(long_path)
            student_df = df_long[df_long["student_id"] == student_id]
            if not student_df.empty:
                sems = student_df.to_dict(orient="records")
                res = self.predict_single_student(student_id, sems)
                return {
                    "student_id": student_id,
                    "sample_id": res["sample_id"],
                    "window_semester": res["window_semester"],
                    "academic_risk_index": res["academic_risk_index"],
                    "risk_band": res["risk_band"],
                    "top_factors": res["top_factors"],
                    "explanation_text": res["explanation_text"],
                    "model_name": "catboost",
                    "calibrated": True,
                    "provenance_tag": "SYNTHETIC",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat()
                }

        return None
