"""Optuna Bayesian hyperparameter tuner for PyTorch deep models."""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from mei.models.deep.architectures import build_deep_model
from mei.models.deep.trainer import DeepEngagementDataset, train_two_stage
from mei.models.registry import ModelSpec

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_deep_model(
    spec: ModelSpec,
    X_dict: Dict[str, Any],
    y: np.ndarray,
    n_trials: int = 10,
    time_budget: int = 120,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Conduct 3-fold inner cross-validated hyperparameter optimization using Optuna.

    Operates strictly within the training split of Fold 0 to prevent test/validation leakage.
    Trials are capped at 40 epochs each for rapid exploration.
    """
    if n_trials <= 0:
        logger.info(
            f"Tuning bypassed for {spec.id} (n_trials={n_trials}); using default parameters."
        )
        return {}, {"n_trials": 0, "best_val_loss": 0.0, "time_budget": time_budget}

    # Inner 3-fold CV split
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    cv_splits = list(inner_cv.split(np.zeros(len(y)), y))

    def objective(trial: optuna.Trial) -> float:
        sampled_params: Dict[str, Any] = {}
        for param_name, param_cfg in spec.search_space.items():
            p_type = param_cfg.get("type", "float")
            if p_type == "categorical":
                sampled_params[param_name] = trial.suggest_categorical(
                    param_name, param_cfg["choices"]
                )
            elif p_type == "int":
                step = param_cfg.get("step", 1)
                sampled_params[param_name] = trial.suggest_int(
                    param_name,
                    int(param_cfg["low"]),
                    int(param_cfg["high"]),
                    step=step,
                )
            elif p_type == "float":
                log = param_cfg.get("log", False)
                sampled_params[param_name] = trial.suggest_float(
                    param_name,
                    float(param_cfg["low"]),
                    float(param_cfg["high"]),
                    log=log,
                )

        fold_losses: list[float] = []

        for in_tr_idx, in_val_idx in cv_splits:
            y_tr, y_v = y[in_tr_idx], y[in_val_idx]

            # Prepare data partitions
            if spec.feature_set == "sequence":
                raw_seq_tr = X_dict["sequence"][in_tr_idx]
                raw_seq_v = X_dict["sequence"][in_val_idx]

                # Fit scaler across time steps
                n_tr, t, d = raw_seq_tr.shape
                scaler = StandardScaler()
                scaled_seq_tr = (
                    scaler.fit_transform(raw_seq_tr.reshape(-1, d))
                    .reshape(n_tr, t, d)
                    .astype(np.float32)
                )
                n_v = raw_seq_v.shape[0]
                scaled_seq_v = (
                    scaler.transform(raw_seq_v.reshape(-1, d))
                    .reshape(n_v, t, d)
                    .astype(np.float32)
                )

                ds_tr = DeepEngagementDataset(X_seq=scaled_seq_tr, y=y_tr)
                ds_val = DeepEngagementDataset(X_seq=scaled_seq_v, y=y_v)
                model = build_deep_model(
                    spec.id, seq_input_dim=d, hyperparams=sampled_params
                )

            elif spec.feature_set == "hybrid_multimodal":
                raw_seq_tr = X_dict["sequence"][in_tr_idx]
                raw_seq_v = X_dict["sequence"][in_val_idx]
                raw_tab_tr = X_dict["agg_states"][in_tr_idx]
                raw_tab_v = X_dict["agg_states"][in_val_idx]

                n_tr, t, d = raw_seq_tr.shape
                seq_scaler = StandardScaler()
                scaled_seq_tr = (
                    seq_scaler.fit_transform(raw_seq_tr.reshape(-1, d))
                    .reshape(n_tr, t, d)
                    .astype(np.float32)
                )
                scaled_seq_v = (
                    seq_scaler.transform(raw_seq_v.reshape(-1, d))
                    .reshape(len(raw_seq_v), t, d)
                    .astype(np.float32)
                )

                tab_scaler = StandardScaler()
                scaled_tab_tr = tab_scaler.fit_transform(raw_tab_tr).astype(np.float32)
                scaled_tab_v = tab_scaler.transform(raw_tab_v).astype(np.float32)

                ds_tr = DeepEngagementDataset(
                    X_tab=scaled_tab_tr, X_seq=scaled_seq_tr, y=y_tr
                )
                ds_val = DeepEngagementDataset(
                    X_tab=scaled_tab_v, X_seq=scaled_seq_v, y=y_v
                )
                model = build_deep_model(
                    spec.id,
                    seq_input_dim=d,
                    tab_input_dim=scaled_tab_tr.shape[1],
                    hyperparams=sampled_params,
                )

            else:  # agg or agg_states
                raw_tab_tr = X_dict[spec.feature_set][in_tr_idx]
                raw_tab_v = X_dict[spec.feature_set][in_val_idx]

                scaler = StandardScaler()
                scaled_tab_tr = scaler.fit_transform(raw_tab_tr).astype(np.float32)
                scaled_tab_v = scaler.transform(raw_tab_v).astype(np.float32)

                ds_tr = DeepEngagementDataset(X_tab=scaled_tab_tr, y=y_tr)
                ds_val = DeepEngagementDataset(X_tab=scaled_tab_v, y=y_v)
                model = build_deep_model(
                    spec.id,
                    tab_input_dim=scaled_tab_tr.shape[1],
                    hyperparams=sampled_params,
                )

            batch_size = int(sampled_params.get("batch_size", 64))
            lr = float(sampled_params.get("lr", 1e-3))
            weight_decay = float(sampled_params.get("weight_decay", 1e-4))

            # Train with cap of 25 epochs in stage 1 for fast inner CV tuning
            train_res = train_two_stage(
                model=model,
                train_dataset=ds_tr,
                val_dataset=ds_val,
                batch_size=batch_size,
                lr_stage1=lr,
                lr_stage2=lr * 0.1,
                weight_decay=weight_decay,
                max_epochs_stage1=25,
                patience_stage1=6,
                max_epochs_stage2=0,
                patience_stage2=0,
                seed=seed,
            )
            fold_losses.append(train_res["best_val_loss"])

        return float(np.mean(fold_losses))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    try:
        study.optimize(objective, n_trials=n_trials, timeout=time_budget)
        best_params = study.best_params
        summary = {
            "n_trials": len(study.trials),
            "best_inner_val_loss": float(study.best_value),
            "best_params": best_params,
        }
    except Exception as e:
        logger.warning(
            f"Optuna tuning encountered issue for {spec.id}: {e}; falling back to defaults."
        )
        best_params = {}
        summary = {"error": str(e), "n_trials": 0}

    return best_params, summary
