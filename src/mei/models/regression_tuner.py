"""Hyperparameter tuning engine powered by Optuna for MEI Track B regression models."""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mei.models.regression_registry import RegressionModelSpec, create_base_regressor

logger = logging.getLogger(__name__)

# Suppress Optuna verbose trial logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def sample_regression_hyperparameters(
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


def tune_regression_model(
    spec: RegressionModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 3,
    seed: int = 42,
    max_trials: Optional[int] = None,
    time_budget_seconds: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Tune candidate regression model hyperparameters using Optuna on inner cross-validation.

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
            "best_r2": None,
            "tuning_time_seconds": 0.0,
        }

    start_time = time.time()
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_splits = list(kf.split(X_train, y_train))

    def objective(trial: optuna.Trial) -> float:
        # Check elapsed time budget
        if time.time() - start_time > budget:
            raise optuna.TrialPruned()

        trial_params = sample_regression_hyperparameters(trial, spec.search_space)
        fold_scores = []

        for tr_idx, val_idx in cv_splits:
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_v, y_v = X_train[val_idx], y_train[val_idx]

            try:
                base_est = create_base_regressor(
                    spec, hyperparams=trial_params, random_seed=seed
                )
                if spec.needs_scaling:
                    model = Pipeline(
                        [("scaler", StandardScaler()), ("regressor", base_est)]
                    )
                else:
                    model = base_est

                model.fit(X_tr, y_tr)
                preds = model.predict(X_v)
                preds_clipped = np.clip(preds, 0.0, 100.0)
                score = r2_score(y_v, preds_clipped)
                fold_scores.append(score)
            except Exception as e:
                logger.debug(f"Trial failed for {spec.id}: {e}")
                return -999.0

        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=budget,
            catch=(Exception,),
        )
    except Exception as e:
        logger.warning(f"Optuna optimization terminated for {spec.id}: {e}")

    duration = time.time() - start_time
    best_params = dict(spec.fixed_params)

    if study.best_trial is not None and study.best_value > -900:
        best_params.update(study.best_params)
        best_score = float(study.best_value)
    else:
        best_score = None

    summary = {
        "n_trials_completed": len(study.trials),
        "best_r2": best_score,
        "tuning_time_seconds": duration,
        "best_params": best_params,
    }

    return best_params, summary
