"""Hyperparameter tuning engine powered by Optuna for MEI candidate models."""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mei.evaluation.metrics import f1_score
from mei.models.registry import ModelSpec, create_base_estimator

logger = logging.getLogger(__name__)

# Suppress Optuna verbose trial logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def sample_hyperparameters(
    trial: optuna.Trial, search_space: Dict[str, Any]
) -> Dict[str, Any]:
    """Sample hyperparameters from an Optuna trial based on search space definitions."""
    params: Dict[str, Any] = {}
    for param_name, config in search_space.items():
        ptype = config.get("type")
        if ptype == "float":
            low = float(config["low"])
            high = float(config["high"])
            log = bool(config.get("log", False))
            step = config.get("step")
            if step is not None:
                params[param_name] = trial.suggest_float(
                    param_name, low, high, step=float(step)
                )
            else:
                params[param_name] = trial.suggest_float(param_name, low, high, log=log)
        elif ptype == "int":
            low = int(config["low"])
            high = int(config["high"])
            step = int(config.get("step", 1))
            params[param_name] = trial.suggest_int(param_name, low, high, step=step)
        elif ptype == "categorical":
            choices = config["choices"]
            params[param_name] = trial.suggest_categorical(param_name, choices)
    return params


def tune_model(
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    seed: int = 42,
    max_trials: Optional[int] = None,
    time_budget_seconds: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Tune candidate model hyperparameters using Optuna on 5-fold cross-validation.

    If n_tuning_trials == 0 or search_space is empty, returns fixed_params immediately.
    """
    n_trials = max_trials if max_trials is not None else spec.n_tuning_trials
    budget = (
        time_budget_seconds
        if time_budget_seconds is not None
        else spec.time_budget_seconds
    )

    if n_trials <= 0 or not spec.search_space:
        return dict(spec.fixed_params), {
            "n_trials_completed": 0,
            "best_cv_score": np.nan,
            "tuning_time_seconds": 0.0,
            "timed_out": False,
        }

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    start_time = time.time()
    timed_out = False

    def objective(trial: optuna.Trial) -> float:
        nonlocal timed_out
        if (time.time() - start_time) > budget:
            timed_out = True
            trial.study.stop()
            return -1.0

        sampled_params = sample_hyperparameters(trial, spec.search_space)
        full_params = dict(spec.fixed_params)
        full_params.update(sampled_params)

        fold_scores = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_v, y_v = X_train[val_idx], y_train[val_idx]

            est = create_base_estimator(
                spec,
                hyperparams=full_params,
                random_seed=seed + trial.number,
                use_class_weights=True,
            )

            if spec.needs_scaling:
                pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
            else:
                pipe = Pipeline([("clf", est)])

            try:
                pipe.fit(X_tr, y_tr)
                preds = pipe.predict(X_v)
                f1 = float(f1_score(y_v, preds, pos_label=1, zero_division=0))
                fold_scores.append(f1)
            except Exception as e:
                logger.debug("Trial %d failed on fold: %s", trial.number, str(e))
                return -1.0

        return float(np.mean(fold_scores)) if fold_scores else -1.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=float(budget),
            catch=(Exception,),
        )
    except Exception as e:
        logger.warning("Optuna optimization error for %s: %s", spec.id, str(e))

    tuning_time = time.time() - start_time
    best_params = dict(spec.fixed_params)

    if len(study.trials) > 0 and study.best_trial.value > -1.0:
        best_params.update(study.best_params)
        best_score = float(study.best_value)
    else:
        best_score = np.nan

    tuning_summary = {
        "n_trials_completed": len(study.trials),
        "best_cv_score": best_score,
        "tuning_time_seconds": float(tuning_time),
        "timed_out": timed_out,
    }

    return best_params, tuning_summary
