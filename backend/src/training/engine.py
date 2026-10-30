"""
Single Model Training, Multi-Seed Validation, and Artifact Serialization Engine.
Orchestrates:
  1. Resumability check
  2. Optuna hyperparameter tuning (grouped CV on train folds)
  3. Final training on full training set with early stopping on validation
  4. Multi-seed stability runs (3 random seeds) computing mean ± std
  5. Probability calibration (Platt/Isotonic on validation)
  6. Decision threshold optimization (Max-F1 / Recall-Constrained)
  7. Atomic artifact saving under artifacts/models/<category>/<model_key>/
  8. Graceful fault tolerance (logs failure reason without crashing)

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import json
import logging
import platform
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import catboost
import lightgbm
import numpy as np
import optuna
import sklearn
import torch
import xgboost

from backend.src.config import config
from backend.src.models.base import ModelRegistry
from backend.src.training.calibration import ProbabilityCalibrator
from backend.src.training.metrics import compute_validation_metrics
from backend.src.training.threshold import optimize_decision_threshold
from backend.src.training.tuner import tune_model_hyperparameters

logger = logging.getLogger("AcademicIntelligence.TrainingEngine")


def _gather_environment_metadata(fit_duration_seconds: float) -> Dict[str, Any]:
    """Records system, hardware, and library versions for reproducibility."""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "device": str(config.device),
        "cuda_available": torch.cuda.is_available(),
        "library_versions": {
            "torch": torch.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
            "optuna": optuna.__version__,
        },
        "preprocessing_reference": "artifacts/models/preprocessor.joblib",
        "total_runtime_seconds": round(fit_duration_seconds, 2),
    }


def train_and_validate_model(
    model_key: str,
    data: Dict[str, Any],
    cv_folds: List[Tuple[np.ndarray, np.ndarray]],
    category: str,
    n_tuning_trials: int = 5,
    deep_epochs: int = 5,
    seeds: Optional[List[int]] = None,
    threshold_strategy: str = "recall_constrained",
    min_recall: float = 0.80,
    calibration_method: str = "platt",
    output_base_dir: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Executes the complete training, calibration, and validation lifecycle for one model.
    """
    if seeds is None:
        seeds = [42, 123, 999]

    base_dir = output_base_dir or config.models_dir
    model_dir = base_dir / f"Category_{category}" / model_key
    summary_path = model_dir / "summary.json"

    # Step 0: Resumability Check
    if summary_path.exists() and not force:
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                saved_summary = json.load(f)
            logger.info(f"[{model_key}] Finished model found. Resuming from disk: {summary_path}")
            return saved_summary
        except Exception as e:
            logger.warning(f"[{model_key}] Stale summary could not be read ({e}). Retraining...")

    t0_overall = time.perf_counter()
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]

        # Step 1: Hyperparameter Tuning via Optuna on training folds
        logger.info(f"[{model_key}] Step 1/5: Tuning hyperparameters ({n_tuning_trials} trials)...")
        best_params, trial_logs = tune_model_hyperparameters(
            model_key=model_key,
            X_train=X_train,
            y_train=y_train,
            cv_folds=cv_folds,
            n_trials=n_tuning_trials,
            deep_epochs=deep_epochs,
            seed=seeds[0],
        )

        # Step 2: Multi-seed Training for Stability Evaluation
        logger.info(f"[{model_key}] Step 2/5: Multi-seed stability training across seeds {seeds}...")
        seed_runs: List[Dict[str, Any]] = []
        primary_model = None
        primary_raw_val_probs: Optional[np.ndarray] = None
        primary_training_history: List[Dict[str, Any]] = []

        for idx, seed in enumerate(seeds):
            seed_params = {**best_params, "random_state": seed, "seed": seed}
            if category in ("D", "E") and deep_epochs > 0:
                seed_params["epochs"] = min(int(seed_params.get("epochs", deep_epochs)), deep_epochs)

            model_inst = ModelRegistry.build(model_key, params=seed_params)
            model_inst.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            raw_val_probs = model_inst.predict_proba(X_val)
            val_metrics = compute_validation_metrics(y_val, raw_val_probs, threshold=0.5)

            seed_runs.append({
                "seed": seed,
                "fit_time_seconds": round(model_inst.fit_time_seconds, 2),
                "metrics": val_metrics,
            })

            if idx == 0:
                primary_model = model_inst
                primary_raw_val_probs = raw_val_probs
                primary_training_history = getattr(model_inst, "training_history", [])

        # Calculate mean and std for stability metrics
        metric_keys = ["roc_auc", "pr_auc", "brier_score", "f1_score", "recall", "precision", "accuracy"]
        stability_summary: Dict[str, Any] = {}
        for m in metric_keys:
            vals = [run["metrics"][m] for run in seed_runs]
            stability_summary[f"{m}_mean"] = round(float(np.mean(vals)), 4)
            stability_summary[f"{m}_std"] = round(float(np.std(vals)), 4)

        # Step 3: Probability Calibration on Validation Partition
        logger.info(f"[{model_key}] Step 3/5: Fitting probability calibrator ({calibration_method})...")
        calibrator = ProbabilityCalibrator(method=calibration_method)
        calibrator.fit(y_val, primary_raw_val_probs)
        calibrated_val_probs = calibrator.transform(primary_raw_val_probs)

        # Step 4: Decision Threshold Selection
        logger.info(f"[{model_key}] Step 4/5: Optimizing decision threshold ({threshold_strategy})...")
        raw_thresh = optimize_decision_threshold(
            y_val, primary_raw_val_probs, strategy=threshold_strategy, min_recall=min_recall
        )
        cal_thresh = optimize_decision_threshold(
            y_val, calibrated_val_probs, strategy=threshold_strategy, min_recall=min_recall
        )

        # Final validated metrics at the chosen calibrated threshold
        final_val_metrics = compute_validation_metrics(
            y_val, calibrated_val_probs, threshold=cal_thresh["optimal_threshold"]
        )

        total_runtime = time.perf_counter() - t0_overall
        env_meta = _gather_environment_metadata(total_runtime)

        # Step 5: Save All Artifacts
        logger.info(f"[{model_key}] Step 5/5: Persisting artifacts to {model_dir}...")

        # 1. Model artifact
        primary_model.save(model_dir)

        # 2. Calibrator
        calibrator.save(model_dir / "calibrator.joblib")

        # 3. Best hyperparameters
        with open(model_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)

        # 4. Training curves
        with open(model_dir / "training_curves.json", "w", encoding="utf-8") as f:
            json.dump(primary_training_history, f, indent=2)

        # 5. Optuna trial logs
        with open(model_dir / "trials.json", "w", encoding="utf-8") as f:
            json.dump(trial_logs, f, indent=2)

        # 6. Multi-seed stability
        with open(model_dir / "seed_metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "seeds": seeds,
                "stability_summary": stability_summary,
                "individual_runs": seed_runs,
            }, f, indent=2)

        # 7. Threshold configurations
        with open(model_dir / "threshold_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "raw_threshold_optimization": raw_thresh,
                "calibrated_threshold_optimization": cal_thresh,
                "selected_threshold": cal_thresh["optimal_threshold"],
            }, f, indent=2)

        # 8. Environment & hardware metadata
        with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(env_meta, f, indent=2)

        # 9. Consolidated summary
        summary = {
            "model_key": model_key,
            "category": category,
            "category_label": primary_model.metadata()["category_label"],
            "label": primary_model.LABEL,
            "is_proposal_model": primary_model.IS_PROPOSAL_MODEL,
            "is_sequence_model": primary_model.IS_SEQUENCE_MODEL,
            "status": "COMPLETED",
            "validation_metrics": final_val_metrics,
            "stability_summary": stability_summary,
            "calibration": calibrator.summary(),
            "threshold": {
                "strategy": threshold_strategy,
                "optimal_threshold": cal_thresh["optimal_threshold"],
                "metrics": cal_thresh["metrics_at_threshold"],
            },
            "runtime_seconds": round(total_runtime, 2),
            "artifact_dir": str(model_dir),
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"[{model_key}] COMPLETED in {total_runtime:.1f}s | "
            f"Val AUROC: {final_val_metrics['roc_auc']:.4f} (std={stability_summary['roc_auc_std']:.4f}) | "
            f"F1: {final_val_metrics['f1_score']:.4f} | Recall: {final_val_metrics['recall']:.1%}"
        )
        return summary

    except Exception as exc:
        err_msg = str(exc)
        stack = traceback.format_exc()
        logger.error(f"[{model_key}] FAILED: {err_msg}\n{stack}")

        error_data = {
            "model_key": model_key,
            "category": category,
            "status": "FAILED",
            "error_message": err_msg,
            "traceback": stack,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(model_dir / "error.json", "w", encoding="utf-8") as f:
            json.dump(error_data, f, indent=2)

        return error_data
