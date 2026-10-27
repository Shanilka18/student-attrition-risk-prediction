"""
Optuna Hyperparameter Tuning Engine with Grouped K-Fold Cross-Validation.
Evaluates search spaces defined in configs/models.yaml using only training data folds.
Logs every trial with hyperparameter choices, objective score, and runtime.

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score

from backend.src.models.base import ModelRegistry

# Reduce noisy Optuna console logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("AcademicIntelligence.Tuner")


def _slice_input_data(X: Any, indices: np.ndarray) -> Any:
    """Helper to slice either tabular array or sequence dict by fold indices."""
    if isinstance(X, dict):
        return {
            "tensors": X["tensors"][indices],
            "padding_mask": X["padding_mask"][indices],
            "seq_lengths": X["seq_lengths"][indices],
        }
    return X[indices]


def tune_model_hyperparameters(
    model_key: str,
    X_train: Any,
    y_train: np.ndarray,
    cv_folds: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 5,
    timeout_seconds: Optional[int] = 300,
    deep_epochs: int = 5,
    seed: int = 42,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Performs Bayesian hyperparameter optimization using Optuna TPESampler.
    Evaluates candidate parameter configurations via grouped CV on training data.
    """
    search_space = ModelRegistry.get_search_space(model_key)
    default_params = ModelRegistry.get_default_params(model_key)

    if not search_space:
        logger.info(f"{model_key}: No hyperparameter search space defined. Using defaults.")
        return default_params, [{"trial": 0, "params": default_params, "value": 0.0, "status": "DEFAULT"}]

    trial_history: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        t_start = time.perf_counter()
        candidate_params: Dict[str, Any] = {}

        # Sample from configured search space
        for param_name, spec in search_space.items():
            p_type = spec.get("type")
            if p_type == "float":
                candidate_params[param_name] = trial.suggest_float(
                    param_name,
                    float(spec["low"]),
                    float(spec["high"]),
                    log=bool(spec.get("log", False)),
                )
            elif p_type == "int":
                candidate_params[param_name] = trial.suggest_int(
                    param_name,
                    int(spec["low"]),
                    int(spec["high"]),
                    log=bool(spec.get("log", False)),
                )
            elif p_type == "categorical":
                candidate_params[param_name] = trial.suggest_categorical(
                    param_name, spec["choices"]
                )

        # For sequence models during tuning, enforce faster tuning epochs
        merged_params = {**default_params, **candidate_params}
        meta = ModelRegistry.get_meta(model_key)
        if meta.get("category") in ("D", "E"):
            merged_params["epochs"] = min(int(merged_params.get("epochs", deep_epochs)), deep_epochs)
            merged_params["early_stopping_patience"] = 3

        # Evaluate across CV folds inside training partition
        fold_scores: List[float] = []
        for trn_idx, val_idx in cv_folds:
            X_tr_f = _slice_input_data(X_train, trn_idx)
            y_tr_f = y_train[trn_idx]
            X_va_f = _slice_input_data(X_train, val_idx)
            y_va_f = y_train[val_idx]

            try:
                model = ModelRegistry.build(model_key, params=merged_params)
                model.fit(X_tr_f, y_tr_f, X_val=X_va_f, y_val=y_va_f)
                val_probs = model.predict_proba(X_va_f)
                score = float(roc_auc_score(y_va_f, val_probs))
            except Exception as e:
                logger.warning(f"Trial {trial.number} fold failed ({model_key}): {e}")
                score = 0.5
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        t_dur = round(time.perf_counter() - t_start, 2)

        trial_history.append({
            "trial": trial.number,
            "params": candidate_params,
            "mean_cv_roc_auc": round(mean_score, 4),
            "fold_scores": [round(s, 4) for s in fold_scores],
            "duration_seconds": t_dur,
        })
        return mean_score

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Enqueue default parameters as trial 0 baseline
    default_overlap = {k: default_params[k] for k in search_space if k in default_params}
    if default_overlap:
        study.enqueue_trial(default_overlap)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        catch=(Exception,),
    )

    best_sampled = study.best_params
    best_params = {**default_params, **best_sampled}
    logger.info(
        f"{model_key}: Tuning complete ({len(trial_history)} trials). "
        f"Best CV ROC-AUC: {study.best_value:.4f} with params: {best_sampled}"
    )

    return best_params, trial_history
