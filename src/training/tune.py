"""Optuna hyperparameter tuner for ERI model zoo.

Usage:
    from src.training.tune import tune_model
    best_params = tune_model(model_class, track, X_train, y_train,
                              view_type='tree', n_trials=30, seed=42)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, KFold

from src.utils.logging_config import setup_logger
from src.utils.paths import get_project_root

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = setup_logger("tune")

ARTIFACTS_PARAMS = get_project_root() / "artifacts" / "params"
ARTIFACTS_PARAMS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Search space definitions
# ---------------------------------------------------------------------------

def _suggest_params(trial: optuna.Trial, model_name: str, track: str) -> Dict[str, Any]:
    """Return hyper-parameter dict for a given model via Optuna suggest_*."""
    p: Dict[str, Any] = {}

    if model_name in ("logistic_l2", "logistic_l1", "logistic_elasticnet"):
        p["C"] = trial.suggest_float("C", 1e-3, 10.0, log=True)
        if model_name == "logistic_elasticnet":
            p["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 0.9)

    elif model_name == "linear_svm":
        p["C"] = trial.suggest_float("C", 1e-3, 10.0, log=True)

    elif model_name == "knn_classifier":
        p["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 21, step=2)
        p["metric"] = trial.suggest_categorical("metric", ["euclidean", "manhattan"])

    elif model_name == "knn_regressor":
        p["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 21, step=2)

    elif model_name in ("decision_tree_classifier", "decision_tree_regressor"):
        p["max_depth"] = trial.suggest_int("max_depth", 3, 12)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 5, 50)

    elif model_name in ("mlp_shallow_classifier", "mlp_shallow_regressor"):
        size = trial.suggest_categorical("hidden_size", [64, 128, 256])
        p["hidden_layer_sizes"] = (size,)
        p["learning_rate_init"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    elif model_name in ("ridge", "lasso"):
        p["alpha"] = trial.suggest_float("alpha", 1e-3, 100.0, log=True)

    elif model_name == "elasticnet_reg":
        p["alpha"] = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
        p["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 0.9)

    elif model_name in ("random_forest", "extra_trees"):
        p["n_estimators"] = trial.suggest_categorical("n_estimators", [100, 200, 300])
        p["max_depth"] = trial.suggest_categorical("max_depth", [None, 6, 10, 15])
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 5, 30)

    elif model_name == "adaboost":
        p["n_estimators"] = trial.suggest_categorical("n_estimators", [100, 150, 200])
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)

    elif model_name in ("gradient_boosting", "hist_gradient_boosting"):
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        p["max_depth"] = trial.suggest_int("max_depth", 3, 8)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 10, 50)

    elif model_name == "xgboost":
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        p["max_depth"] = trial.suggest_int("max_depth", 3, 8)
        p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        p["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
        p["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True)

    elif model_name == "lightgbm":
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        p["num_leaves"] = trial.suggest_int("num_leaves", 20, 80)
        p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        p["min_child_samples"] = trial.suggest_int("min_child_samples", 10, 50)

    elif model_name == "catboost":
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        p["depth"] = trial.suggest_int("depth", 4, 8)
        p["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)

    elif model_name == "deep_mlp":
        n_layers = trial.suggest_int("n_layers", 2, 4)
        layer_size = trial.suggest_categorical("layer_size", [64, 128, 256])
        p["hidden_sizes"] = tuple([layer_size] * n_layers)
        p["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
        p["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        p["max_epochs"] = 50  # reduced for tuning

    elif model_name == "tabular_resnet":
        p["dim"] = trial.suggest_categorical("dim", [64, 128, 256])
        p["n_blocks"] = trial.suggest_int("n_blocks", 2, 4)
        p["dropout"] = trial.suggest_float("dropout", 0.1, 0.4)
        p["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        p["max_epochs"] = 50

    elif model_name == "tabnet":
        p["n_d"] = trial.suggest_categorical("n_d", [16, 32, 64])
        p["n_steps"] = trial.suggest_int("n_steps", 2, 5)
        p["max_epochs"] = 50

    elif model_name == "cnn_1d":
        p["n_filters"] = trial.suggest_categorical("n_filters", [64, 128, 256])
        p["dropout"] = trial.suggest_float("dropout", 0.1, 0.4)
        p["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        p["max_epochs"] = 50

    return p


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def _make_objective(
    model_class,
    track: str,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    view_type: str,
    n_inner_folds: int,
    seed: int,
    model_extra_kwargs: Dict[str, Any],
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective function for inner-CV tuning."""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name, track)
        scores = []

        cv = (StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
              if track == "C"
              else KFold(n_splits=n_inner_folds, shuffle=True, random_state=seed))

        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            try:
                model = model_class(
                    track=track,
                    params=params,
                    seed=seed + fold_idx,
                    **model_extra_kwargs,
                )
                model.fit(X_tr, y_tr)

                if track == "C":
                    from sklearn.metrics import roc_auc_score
                    proba = model.predict_proba(X_va)[:, 1]
                    score = roc_auc_score(y_va, proba)
                else:
                    from sklearn.metrics import r2_score
                    pred = model.predict(X_va)
                    score = r2_score(y_va, pred)
                scores.append(score)
            except Exception as e:
                logger.warning(f"  Trial fold {fold_idx} failed: {e}")
                scores.append(0.0 if track == "C" else -999.0)

        return float(np.mean(scores))

    return objective


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def tune_model(
    model_class,
    track: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    view_type: str = "tree",
    n_trials: int = 30,
    n_inner_folds: int = 3,
    seed: int = 42,
    model_extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Optuna tuning and return best params dict.

    Parameters
    ----------
    model_class : class
        The model wrapper class (e.g. XGBoostModel).
    track : str
        "C" or "R".
    X_train : ndarray
        Training features (inner CV splits here).
    y_train : ndarray
        Training labels / targets.
    model_name : str
        e.g. "xgboost" — used for search-space selection and file naming.
    view_type : str
        "tree" or "scaled".
    n_trials : int
        Optuna trial budget.
    n_inner_folds : int
        Number of inner CV folds for each trial.
    seed : int
        Random seed for reproducibility.
    model_extra_kwargs : dict, optional
        Extra constructor kwargs (e.g. ``hidden_sizes`` for DeepMLP).

    Returns
    -------
    dict
        Best hyper-parameter dictionary (to pass as ``params=`` to model ctor).
    """
    model_extra_kwargs = model_extra_kwargs or {}
    direction = "maximize"

    logger.info(f"Tuning {model_name} | Track {track} | {n_trials} trials | "
                f"inner {n_inner_folds}-fold CV")

    objective = _make_objective(
        model_class=model_class,
        track=track,
        X=X_train,
        y=y_train,
        model_name=model_name,
        view_type=view_type,
        n_inner_folds=n_inner_folds,
        seed=seed,
        model_extra_kwargs=model_extra_kwargs,
    )

    sampler = optuna.samplers.TPESampler(seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                       catch=(Exception,))

    best_params = study.best_params if study.best_trial is not None else {}
    best_value = study.best_value if study.best_trial is not None else float("nan")

    logger.info(f"  {model_name} best CV score: {best_value:.4f} | params: {best_params}")

    # Persist to artifacts/params/
    out_path = ARTIFACTS_PARAMS / f"{model_name}_{track}.json"
    out_path.write_text(
        json.dumps({
            "model_name": model_name,
            "track": track,
            "n_trials": n_trials,
            "best_value": best_value,
            "best_params": best_params,
            "all_trials": len(study.trials),
        }, indent=2),
        encoding="utf-8",
    )
    logger.info(f"  Saved tuning results → {out_path}")
    return best_params
