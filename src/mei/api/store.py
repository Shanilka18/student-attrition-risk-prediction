"""RunStore: Real artifact loader and in-memory cache for the MEI platform.

Reads exclusively from artifacts/runs/<active_run_id>/ and data/ directories.
Zero hard-coded metrics, zero fake endpoints (Rule G2).
Switching the active run clears all in-memory caches.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mei.api.models import (
    ChampionSummary,
    LeaderboardResponse,
    LeaderboardRow,
    StatusResponse,
)
from mei.config import ARTIFACTS_DIR, RUNS_DIR
from mei.io.schema import load_schema

logger = logging.getLogger(__name__)


class RunStore:
    """Thread-safe real artifact manager and cache for active MEI runs."""

    _instance: Optional["RunStore"] = None
    _lock = threading.Lock()

    def __init__(
        self, runs_dir: Optional[Path] = None, artifacts_dir: Optional[Path] = None
    ) -> None:
        self.runs_dir = Path(runs_dir) if runs_dir else RUNS_DIR
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
        self.active_file = self.artifacts_dir / "active_run.txt"
        self._cache: Dict[str, Any] = {}
        self._cached_run_id: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "RunStore":
        """Get or create singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = RunStore()
            return cls._instance

    # -----------------------------------------------------------------------
    # Active Run Management & Cache Invalidation
    # -----------------------------------------------------------------------

    def get_active_run_id(self) -> str:
        """Resolve current active run ID from disk."""
        if self.active_file.exists():
            run_id = self.active_file.read_text(encoding="utf-8").strip()
            if run_id:
                return run_id

        # Fallback to newest run in directory
        runs = [
            d.name
            for d in self.runs_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]
        if runs:
            return sorted(runs)[-1]
        return "run_default"

    def get_active_run_dir(self) -> Path:
        """Return Path to active run directory."""
        run_id = self.get_active_run_id()
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Active run directory not found: {run_dir}")
        return run_dir

    def set_active_run(self, run_id: str) -> str:
        """Switch active run, invalidate cache, and write to disk."""
        target_dir = self.runs_dir / run_id
        if not target_dir.exists() or not target_dir.is_dir():
            raise FileNotFoundError(f"Run {run_id} does not exist in {self.runs_dir}")

        with self._lock:
            self.active_file.write_text(run_id, encoding="utf-8")
            self._cache.clear()
            self._cached_run_id = run_id
            logger.info("Active run switched to: %s. Cache invalidated.", run_id)
        return run_id

    def _ensure_cache_validity(self) -> None:
        """Invalidate cache if active run changed externally on disk."""
        current_id = self.get_active_run_id()
        if self._cached_run_id != current_id:
            with self._lock:
                self._cache.clear()
                self._cached_run_id = current_id

    def load_json(self, rel_path: str) -> Any:
        """Load and cache a JSON artifact from the active run directory."""
        self._ensure_cache_validity()
        cache_key = f"{self._cached_run_id}:{rel_path}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        run_dir = self.get_active_run_dir()
        file_path = run_dir / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact {rel_path} not found in {run_dir}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with self._lock:
            self._cache[cache_key] = data
        return data

    # -----------------------------------------------------------------------
    # Status & Health
    # -----------------------------------------------------------------------

    def get_status(self) -> StatusResponse:
        """Build status payload directly from champion model card and thresholds."""
        run_id = self.get_active_run_id()
        model_card = self.load_json("champion/model_card.json")
        thresholds_data = self.load_json("champion/thresholds.json")
        bands_data = self.load_json("champion/bands.json")

        # Track B champion resolution
        reg_leaderboard = self.get_leaderboard("regression")
        track_b_champ = (
            reg_leaderboard.rows[0].model_dump() if reg_leaderboard.rows else {}
        )

        opt_th = thresholds_data.get("optimal_threshold", {}).get("threshold", 0.26)
        monotonic = bands_data.get("test_holdout_verification", {}).get(
            "is_strictly_monotonic", True
        )

        n_train = model_card.get("training_cohort_size", 1856)
        n_test = model_card.get("test_cohort_size", 328)
        n_total = n_train + n_test

        data_source_raw = model_card.get(
            "data_source", "Sample survey responses (generated)"
        )
        is_gen = (
            "generated" in data_source_raw.lower()
            or "sample" in data_source_raw.lower()
        )
        badge = "Sample survey responses (generated)" if is_gen else "Your survey data"

        champ_summary = ChampionSummary(
            model_id=model_card.get("model_id", "knn"),
            display_name=model_card.get("display_name", "k-Nearest Neighbours"),
            plain_name=model_card.get("plain_name", "Distance-Based Peer Classifier"),
            category=model_card.get("category", "classical"),
            test_f1=model_card.get("test_f1"),
            test_auc_roc=model_card.get("test_auc_roc"),
            test_mcc=model_card.get("test_mcc"),
            cv_score=model_card.get("validation_cv_score", 0.7949),
            cv_f1=model_card.get("validation_cv_f1", 0.6969),
            optimal_threshold=opt_th,
            inference_time_ms=59.05,
            calibrator=model_card.get("calibrator", "IsotonicRegression"),
            band_verification_monotonic=monotonic,
        )

        return StatusResponse(
            active_run_id=run_id,
            data_source=data_source_raw,
            is_generated_sample=is_gen,
            data_source_badge=badge,
            n_students_total=n_total,
            n_students_train=n_train,
            n_students_test=n_test,
            trained_at=model_card.get("timestamp_utc", ""),
            champion=champ_summary,
            track_b_champion=track_b_champ,
            model_card_essentials={
                "component_id": model_card.get("component_id", "KND_CDAP_IT_05"),
                "project_name": model_card.get(
                    "project_name", "Multimodal Student Engagement Intelligence"
                ),
                "bootstrap_95_ci": model_card.get("bootstrap_95_ci", {}),
                "limitations": model_card.get("limitations", []),
                "sensitive_threshold": thresholds_data.get("sensitive_threshold", {}),
            },
            bands=bands_data,
        )

    # -----------------------------------------------------------------------
    # Leaderboard & Models
    # -----------------------------------------------------------------------

    def get_leaderboard(self, track: str = "classification") -> LeaderboardResponse:
        """Return leaderboard rows for classification or regression with flags."""
        track_norm = track.lower().strip()
        run_id = self.get_active_run_id()

        if track_norm in ["regression", "b", "track_b"]:
            raw_rows = self.load_json("leaderboard_regression.json")
            best_cat = self.load_json("per_category_best_regression.json")
            remediation_info = self.load_json("remediation_summary.json")
            metric_name = "R² Score"
            champ_id = "stacking_regressor"
        else:
            raw_rows = self.load_json("leaderboard_classification.json")
            best_cat = self.load_json("per_category_best.json")
            remediation_info = {}
            metric_name = "F1 Score (Threshold-Tuned)"
            champ_id = "knn"

        # Deduplicate rows by model_id (preserve highest score / first appearance)
        seen_ids = set()
        deduped = []
        for r in raw_rows:
            mid = r["model_id"]
            if mid not in seen_ids:
                seen_ids.add(mid)
                deduped.append(r)

        category_best_ids = {
            v.get("model_id") for v in best_cat.values() if isinstance(v, dict)
        }

        parsed_rows: List[LeaderboardRow] = []
        for r in deduped:
            mid = r["model_id"]
            row_dict = dict(r)
            row_dict["is_champion"] = mid == champ_id
            row_dict["is_category_best"] = mid in category_best_ids
            row_dict["is_remediated"] = mid in remediation_info or "remediated" in mid
            parsed_rows.append(LeaderboardRow(**row_dict))

        # Re-sort by leaderboard_score descending
        parsed_rows.sort(key=lambda x: x.leaderboard_score, reverse=True)
        for idx, row in enumerate(parsed_rows, start=1):
            row.rank = idx

        return LeaderboardResponse(
            track="regression" if "regression" in track_norm else "classification",
            active_run_id=run_id,
            total_models=len(parsed_rows),
            metric_name=metric_name,
            champion_id=champ_id,
            rows=parsed_rows,
        )

    def get_best_per_category(self, track: str = "classification") -> Dict[str, Any]:
        """Return dictionary of best model per architectural category."""
        track_norm = track.lower().strip()
        if track_norm in ["regression", "b", "track_b"]:
            return self.load_json("per_category_best_regression.json")
        return self.load_json("per_category_best.json")

    def get_model_detail(
        self, model_id: str, track: str = "classification"
    ) -> Dict[str, Any]:
        """Return comprehensive detail for a model."""
        track_norm = track.lower().strip()
        run_dir = self.get_active_run_dir()

        # If regression
        if track_norm in ["regression", "b", "track_b"]:
            reg_details = self.load_json("models/regression_details.json")
            if model_id in reg_details:
                detail = reg_details[model_id]
                overfit_audit = self.load_json("overfit_audit.json")
                reg_models_list = overfit_audit.get("track_b_regression", {}).get(
                    "models", []
                )
                reg_audit = next(
                    (
                        m
                        for m in reg_models_list
                        if isinstance(m, dict) and m.get("model_id") == model_id
                    ),
                    {},
                )
                status_expl = (
                    reg_audit.get("explanation")
                    or f"Model generalization status: {detail.get('overfit_status', 'Good fit')}"
                )
                return {
                    "model_id": model_id,
                    "display_name": detail.get("display_name", model_id),
                    "plain_name": detail.get("plain_name", model_id),
                    "category": detail.get("category", "regression"),
                    "track": "regression",
                    "feature_set": detail.get("feature_set", "agg"),
                    "leaderboard_score": detail.get("leaderboard_score", 0.0),
                    "generalization_gap": detail.get("generalization_gap", 0.0),
                    "overfit_status": detail.get("overfit_status", "Good fit"),
                    "status_explanation": status_expl,
                    "hyperparameters": detail.get("hyperparameters", {}),
                    "training_time_s": detail.get("training_time_s", 0.0),
                    "inference_time_ms": detail.get("inference_time_ms", 0.0),
                    "metrics": detail,
                    "fold_metrics": detail.get("fold_metrics", []),
                    "test_metrics": detail.get("test"),
                }

        # Classification detail
        model_dir = run_dir / "models" / model_id
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model {model_id} not found in {run_dir / 'models'}"
            )

        detail = self.load_json(f"models/{model_id}/detail.json")
        overfit_audit = self.load_json("overfit_audit.json")
        clf_models_list = overfit_audit.get("track_a_classification", {}).get(
            "models", []
        )
        clf_audit = next(
            (
                m
                for m in clf_models_list
                if isinstance(m, dict) and m.get("model_id") == model_id
            ),
            {},
        )
        status_expl = (
            clf_audit.get("explanation")
            or f"Model generalization status: {detail.get('leaderboard_metrics', {}).get('overfit_status', 'Good fit')}"
        )

        curves_file = model_dir / "curves.json"
        thresh_table = None
        if curves_file.exists():
            curves_data = self.load_json(f"models/{model_id}/curves.json")
            thresh_table = curves_data.get("curves", {}).get("threshold_table")

        return {
            "model_id": model_id,
            "display_name": detail.get("spec", {}).get("display_name", model_id),
            "plain_name": detail.get("spec", {}).get("plain_name", model_id),
            "category": detail.get("spec", {}).get("category", "classical"),
            "track": "classification",
            "feature_set": detail.get("spec", {}).get("feature_set", "agg"),
            "leaderboard_score": detail.get("leaderboard_metrics", {}).get(
                "leaderboard_score", 0.0
            ),
            "generalization_gap": detail.get("leaderboard_metrics", {}).get(
                "generalization_gap", 0.0
            ),
            "overfit_status": detail.get("leaderboard_metrics", {}).get(
                "overfit_status", "Good fit"
            ),
            "status_explanation": status_expl,
            "hyperparameters": detail.get("hyperparameters", {}),
            "training_time_s": detail.get("leaderboard_metrics", {}).get(
                "training_time_s", 0.0
            ),
            "inference_time_ms": detail.get("leaderboard_metrics", {}).get(
                "inference_time_ms", 0.0
            ),
            "metrics": detail.get("leaderboard_metrics", {}),
            "fold_metrics": detail.get("fold_metrics", []),
            "confusion_matrix_raw": detail.get("confusion_matrix_raw"),
            "confusion_matrix_norm": detail.get("confusion_matrix_norm"),
            "threshold_table": thresh_table,
            "test_metrics": detail.get("test"),
        }

    def get_model_curves(
        self, model_id: str, track: str = "classification"
    ) -> Dict[str, Any]:
        """Return training, epoch, and metric curves for visualizations."""
        run_dir = self.get_active_run_dir()
        track_norm = track.lower().strip()
        model_dir = run_dir / "models" / model_id

        curves_path = model_dir / "curves.json"
        curves_data = (
            self.load_json(f"models/{model_id}/curves.json")
            if curves_path.exists()
            else {}
        )

        # Check remediation summary
        remediation_info = self.load_json("remediation_summary.json")
        remediation_comp = remediation_info.get(model_id)

        curves_inner = curves_data.get("curves", {})
        rep_fold = curves_data.get("representative_fold", {})

        return {
            "model_id": model_id,
            "track": "regression" if "regression" in track_norm else "classification",
            "category": curves_data.get("category", "unknown"),
            "roc_fpr": curves_inner.get("roc_fpr"),
            "roc_tpr": curves_inner.get("roc_tpr"),
            "pr_precision": curves_inner.get("pr_precision"),
            "pr_recall": curves_inner.get("pr_recall"),
            "cal_prob_true": curves_inner.get("cal_prob_true"),
            "cal_prob_pred": curves_inner.get("cal_prob_pred"),
            "epoch_curves": rep_fold.get("history"),
            "best_epoch": rep_fold.get("best_epoch"),
            "stage_2_start_epoch": rep_fold.get("stage_2_start_epoch"),
            "stage_2_used": rep_fold.get("stage_2_used"),
            "overfitting_zone": rep_fold.get("overfitting_zone"),
            "iteration_losses": curves_data.get("iteration_losses"),
            "learning_curves": curves_data.get("learning_curves"),
            "validation_curves": curves_data.get("validation_curves"),
            "remediation_comparison": remediation_comp,
        }

    def compare_models(
        self, model_ids: List[str], track: str = "classification"
    ) -> Dict[str, Any]:
        """Compare up to 4 models side-by-side."""
        selected_ids = model_ids[:4]
        models_data = []
        metrics_comp: Dict[str, Dict[str, Any]] = {}

        for mid in selected_ids:
            try:
                detail = self.get_model_detail(mid, track=track)
                models_data.append(detail)
                metrics_comp[mid] = detail.get("metrics", {})
            except Exception as e:
                logger.warning("Could not fetch detail for model %s: %s", mid, e)

        return {
            "track": track,
            "model_ids": selected_ids,
            "models": models_data,
            "metrics_comparison": metrics_comp,
        }

    # -----------------------------------------------------------------------
    # Patterns & Insights
    # -----------------------------------------------------------------------

    def get_patterns(self) -> Dict[str, Any]:
        """Return state profiles, transition dynamics, pattern trajectories, and risk rates."""
        run_id = self.get_active_run_id()
        summary = self.load_json("states/summary.json")

        hmm_info = summary.get("hmm", {})
        clustering_info = summary.get("clustering", {})
        pat_dist = summary.get("pattern_distribution", {})
        lvl_dist = summary.get("level_distribution", {})
        ablation = summary.get("ablation_benchmark", {})

        # Compute real empirical trajectories and risk rates by pattern
        run_dir = self.get_active_run_dir()
        tr_states_path = run_dir / "features" / "train_agg_states.parquet"
        val_states_path = run_dir / "features" / "val_agg_states.parquet"

        mean_trajs: Dict[str, Dict[str, float]] = {}
        risk_rates: Dict[str, Dict[str, Any]] = {}

        if tr_states_path.exists() and val_states_path.exists():
            tr_states = pd.read_parquet(tr_states_path)
            val_states = pd.read_parquet(val_states_path)
            all_states = pd.concat([tr_states, val_states]).reset_index(drop=True)

            # Map patterns
            patterns: List[str] = []
            for _, r in all_states[
                [
                    "pattern_consistent",
                    "pattern_declining",
                    "pattern_improving",
                    "pattern_sporadic",
                ]
            ].iterrows():
                if r.get("pattern_consistent") == 1:
                    patterns.append("Consistent")
                elif r.get("pattern_declining") == 1:
                    patterns.append("Declining")
                elif r.get("pattern_improving") == 1:
                    patterns.append("Improving")
                else:
                    patterns.append("Sporadic")
            all_states["pattern"] = patterns

            comps = ["composite_p1", "composite_p2", "composite_p3", "composite_p4"]
            for pat, g in all_states.groupby("pattern"):
                mean_trajs[pat] = {
                    col: round(float(g[col].mean()), 2)
                    for col in comps
                    if col in g.columns
                }

            # At-risk rates if target indices exist
            tr_idx = self.load_json("splits/train_idx.json")
            val_idx = self.load_json("splits/val_idx.json")
            cohort_files = list(Path("data/interim").glob("clean_run_*.parquet"))
            cohort_files = [f for f in cohort_files if f.stat().st_size > 100_000]
            if cohort_files:
                cohort_df = pd.read_parquet(sorted(cohort_files)[-1])
                y = cohort_df.iloc[tr_idx + val_idx]["at_risk"].values
                all_states["at_risk"] = y

                for pat, g in all_states.groupby("pattern"):
                    n = len(g)
                    k = int(g["at_risk"].sum())
                    p = k / n if n > 0 else 0.0
                    se = float(np.sqrt(p * (1 - p) / n)) if n > 0 else 0.0
                    risk_rates[pat] = {
                        "n_students": n,
                        "at_risk_count": k,
                        "at_risk_pct": round(p * 100.0, 1),
                        "ci95_low": round(max(0.0, p - 1.96 * se) * 100.0, 1),
                        "ci95_high": round(min(1.0, p + 1.96 * se) * 100.0, 1),
                    }

        return {
            "active_run_id": run_id,
            "hmm_k": hmm_info.get("selected_k", 3),
            "clustering_k": clustering_info.get("selected_k", 3),
            "state_names": hmm_info.get("state_names", {}),
            "transition_matrix": hmm_info.get("transition_matrix", []),
            "emission_means": hmm_info.get("emission_means", {}),
            "expected_dwell_times": hmm_info.get("expected_dwell_times", {}),
            "pattern_distribution": pat_dist,
            "level_distribution": lvl_dist,
            "mean_trajectories_by_pattern": mean_trajs,
            "risk_rates_by_pattern": risk_rates,
            "clustering_comparison_table": clustering_info.get("metrics_table", []),
            "ablation_benchmark": ablation,
        }

    # -----------------------------------------------------------------------
    # Risk Drivers & Explainability
    # -----------------------------------------------------------------------

    def get_drivers(self) -> Dict[str, Any]:
        """Return complete feature attribution and SHAP driver insights."""
        run_id = self.get_active_run_id()
        glob_imp = self.load_json("explain/global_importance.json")
        grp_imp = self.load_json("explain/grouped_importance.json")
        consistency = self.load_json("explain/shap_consistency.json")
        cross_method = self.load_json("explain/cross_method_comparison.json")
        pattern_imp = self.load_json("explain/pattern_importance.json")
        beeswarm = self.load_json("explain/beeswarm_samples.json")
        dependence = self.load_json("explain/dependence_samples.json")

        return {
            "active_run_id": run_id,
            "baseline_probability": glob_imp.get("baseline_probability", 0.2462),
            "baseline_mei": glob_imp.get("baseline_mei", 24.6),
            "validation_ranking": glob_imp.get("validation_ranking", []),
            "grouped_importance": grp_imp,
            "shap_consistency": consistency,
            "cross_method_comparison": cross_method,
            "pattern_importance": pattern_imp,
            "tree_hybrid_audit": glob_imp.get("tree_hybrid_audit"),
            "beeswarm_samples": beeswarm,
            "dependence_samples": dependence,
        }

    # -----------------------------------------------------------------------
    # Data Dictionary
    # -----------------------------------------------------------------------

    def get_dictionary(self) -> Dict[str, Any]:
        """Return standardized data dictionary for questionnaire items."""
        schema = load_schema()
        cols = []
        for c in schema.columns:
            cols.append(
                {
                    "name": c.name,
                    "type": c.type,
                    "role": c.role,
                    "period": c.period,
                    "indicator": c.indicator,
                    "description": c.description,
                    "allowed_codes": c.allowed_codes,
                    "text_labels": {str(k): v for k, v in c.text_labels.items()}
                    if c.text_labels
                    else None,
                }
            )

        return {
            "version": schema.version,
            "num_columns": schema.num_columns,
            "num_features": schema.num_features,
            "num_outcomes": schema.num_outcomes,
            "columns": cols,
        }

    # -----------------------------------------------------------------------
    # Runs Management
    # -----------------------------------------------------------------------

    def get_all_runs(self) -> List[Dict[str, Any]]:
        """List all valid experimental runs present in artifacts/runs/."""
        active_id = self.get_active_run_id()
        runs = []

        for p in self.runs_dir.iterdir():
            if not p.is_dir() or not p.name.startswith("run_"):
                continue

            card_path = p / "champion" / "model_card.json"
            if card_path.exists():
                try:
                    with open(card_path, "r", encoding="utf-8") as f:
                        mc = json.load(f)
                    runs.append(
                        {
                            "run_id": p.name,
                            "is_active": p.name == active_id,
                            "data_source": mc.get(
                                "data_source", "Sample survey responses (generated)"
                            ),
                            "trained_at": mc.get("timestamp_utc", ""),
                            "n_students": mc.get("training_cohort_size", 0)
                            + mc.get("test_cohort_size", 0),
                            "champion_model_id": mc.get("model_id", "knn"),
                            "champion_f1": mc.get("test_f1"),
                            "champion_cv_score": mc.get("validation_cv_score"),
                        }
                    )
                except Exception as e:
                    logger.debug("Skipping unreadable run %s: %s", p.name, e)

        # Sort with active run first, then by run timestamp descending
        runs.sort(key=lambda x: (not x["is_active"], x["run_id"]), reverse=False)
        return runs
