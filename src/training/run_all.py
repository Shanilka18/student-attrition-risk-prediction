"""Fault-tolerant model zoo runner for ERI Project.

Trains every model in the zoo on the same splits with the same preprocessing.
Supports Track C (classification) and Track R (regression).

Usage:
    python -m src.training.run_all --track C --models all --quick
    python -m src.training.run_all --track R --models xgboost,lightgbm
    python -m src.training.run_all --track C --no-tune

Arguments:
    --track         C or R (required)
    --models        Comma-separated list of model names, or 'all'
    --quick         3 Optuna trials, 30-epoch deep models, for fast testing
    --no-tune       Skip Optuna (use defaults)
    --n-trials      Override Optuna trial budget (default: 30)
    --open-test     Evaluate on sealed test set (finalists only, use with care)
    --seed          Random seed (default: 42)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

# ── project path setup ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logging_config import setup_logger
from src.utils.paths import get_project_root
from src.utils.seed import set_seed
from src.preprocessing.pipeline import (
    LeakageProofPreprocessor,
    get_tree_pipeline,
    get_scaled_pipeline,
    validate_feature_boundary,
)
from src.preprocessing.split import get_sealed_test_set
from src.models.base import BaseTabularModel
from src.models.linear_shallow import get_linear_shallow_models
from src.models.tree_ensembles import get_tree_ensemble_models
from src.models.deep_tabular import get_deep_tabular_models
from src.training.tune import tune_model
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_roc_curve_data,
    compute_pr_curve_data,
    compute_calibration_curve_data,
)

logger = setup_logger("run_all")
ROOT = get_project_root()

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

DIRS = {
    "models_C": ROOT / "artifacts" / "models" / "C",
    "models_R": ROOT / "artifacts" / "models" / "R",
    "history": ROOT / "artifacts" / "history",
    "params": ROOT / "artifacts" / "params",
    "preds": ROOT / "artifacts" / "preds",
    "curves": ROOT / "artifacts" / "curves",
    "metrics": ROOT / "artifacts" / "metrics",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_model_registry(track: str, seed: int, quick: bool) -> Dict[str, BaseTabularModel]:
    """Return dict of model_name → model instance."""
    linear = get_linear_shallow_models(track, seed=seed)
    trees = get_tree_ensemble_models(track, seed=seed)
    deep = get_deep_tabular_models(track, seed=seed)

    if quick:
        # Reduce epochs for deep models during quick/test runs
        for m in deep.values():
            m.params.setdefault("max_epochs", 10)
            m.params.setdefault("patience", 5)

    registry = {}
    registry.update(linear)
    registry.update(trees)
    registry.update(deep)
    return registry


# ---------------------------------------------------------------------------
# Feature view selection
# ---------------------------------------------------------------------------

_TREE_MODELS = {
    "random_forest", "extra_trees", "adaboost", "gradient_boosting",
    "hist_gradient_boosting", "xgboost", "lightgbm", "catboost",
    "decision_tree_classifier", "decision_tree_regressor",
    "gaussian_nb",  # NB works better without scaling
}


def _view_for(model_name: str) -> str:
    return "tree" if model_name in _TREE_MODELS else "scaled"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_data(track: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Load train.csv, val.csv, and split_ids.json."""
    proc = ROOT / "data" / "processed"
    train_df = pd.read_csv(proc / "train.csv")
    val_df = pd.read_csv(proc / "val.csv")
    with open(proc / "split_ids.json", "r") as f:
        split_ids = json.load(f)
    return train_df, val_df, split_ids


def _get_target_col(track: str) -> str:
    return "attrition" if track == "C" else "latent_risk_score"


def _prepare_xy(df: pd.DataFrame, track: str, view_type: str, preprocessor=None,
                is_train: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[Any]]:
    """Extract and preprocess X, y from a dataframe.

    If is_train=True, fits and transforms.
    Otherwise, transforms only using a fitted preprocessor.
    Returns (X_arr, y_arr, fitted_preprocessor_or_None).
    """
    target_col = _get_target_col(track)
    y = df[target_col].values.astype(float)

    # Remove target + all forbidden columns before feature boundary validation
    feature_df = validate_feature_boundary(df)

    if is_train:
        if view_type == "tree":
            prep = get_tree_pipeline()
        else:
            prep = get_scaled_pipeline()
        X = prep.fit_transform(feature_df, is_train_fold=True)
        return X, y, prep
    else:
        X = preprocessor.transform(feature_df)
        return X, y, preprocessor


# ---------------------------------------------------------------------------
# Single-model training & evaluation
# ---------------------------------------------------------------------------

def run_one_model(
    model_name: str,
    model: BaseTabularModel,
    track: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    do_tune: bool,
    n_trials: int,
    open_test: bool,
    seed: int,
) -> Dict[str, Any]:
    """Train, evaluate, persist one model. Returns a results dict."""
    view_type = _view_for(model_name)
    logger.info(f">> [{model_name}] track={track} view={view_type}")

    result: Dict[str, Any] = {
        "model_name": model_name, "track": track, "view_type": view_type,
        "status": "success", "error": None,
    }

    # ── Preprocess ──────────────────────────────────────────────────────────
    X_train, y_train, preprocessor = _prepare_xy(train_df, track, view_type, is_train=True)
    X_val, y_val, _ = _prepare_xy(val_df, track, view_type,
                                   preprocessor=preprocessor, is_train=False)

    # ── Optuna Tuning (inner-fold on train only) ─────────────────────────────
    best_params: Dict[str, Any] = {}
    if do_tune:
        try:
            # Map model class
            model_class = type(model)
            extra_kwargs: Dict[str, Any] = {}
            if hasattr(model, "penalty"):
                extra_kwargs["penalty"] = model.penalty
            if hasattr(model, "hidden_sizes"):
                extra_kwargs["hidden_sizes"] = model.hidden_sizes
            if hasattr(model, "dim"):
                extra_kwargs["dim"] = model.dim

            best_params = tune_model(
                model_class=model_class,
                track=track,
                X_train=X_train,
                y_train=y_train,
                model_name=model_name,
                view_type=view_type,
                n_trials=n_trials,
                seed=seed,
                model_extra_kwargs=extra_kwargs,
            )
            # Rebuild model with tuned params
            model = model_class(
                track=track,
                params=best_params,
                seed=seed,
                **extra_kwargs,
            )
        except Exception as e:
            logger.warning(f"  Tuning failed for {model_name}: {e}. Using defaults.")
            best_params = {}

    # ── Fit ──────────────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    history = model.get_history()
    history_path = DIRS["history"] / f"{model_name}_{track}_history.json"
    history_path.write_text(json.dumps(history.to_dict(), indent=2), encoding="utf-8")
    logger.info(f"  History saved -> {history_path.name}")

    # ── Predict ──────────────────────────────────────────────────────────────
    splits_data: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    for split_name, X_s, y_s in [("train", X_train, y_train), ("val", X_val, y_val)]:
        y_pred = model.predict(X_s)
        if track == "C":
            y_prob = model.predict_proba(X_s)[:, 1]
        else:
            y_prob = y_pred  # placeholder for regression
        splits_data.append((split_name, y_pred, y_prob, y_s))

    # Optionally open test set
    if open_test:
        test_df = get_sealed_test_set(
            allow_test_access=True,
            caller_description=f"run_all.py final eval — {model_name} {track}",
        )
        X_test, y_test, _ = _prepare_xy(test_df, track, view_type,
                                          preprocessor=preprocessor, is_train=False)
        y_pred_test = model.predict(X_test)
        y_prob_test = model.predict_proba(X_test)[:, 1] if track == "C" else y_pred_test
        splits_data.append(("test", y_pred_test, y_prob_test, y_test))

    # ── Compute Metrics ───────────────────────────────────────────────────────
    all_metrics: List[Dict[str, Any]] = []
    n_features = X_train.shape[1]

    for split_name, y_pred, y_prob, y_true in splits_data:
        if track == "C":
            m = compute_classification_metrics(y_true, y_pred, y_prob, split=split_name, seed=seed)
        else:
            m = compute_regression_metrics(y_true, y_pred, split=split_name,
                                           n_features=n_features, seed=seed)
        m["model"] = model_name
        m["track"] = track
        all_metrics.append(m)

        # Save predictions
        preds_path = DIRS["preds"] / f"{model_name}_{track}_{split_name}_preds.npy"
        np.save(str(preds_path), y_prob)
        logger.info(f"  {split_name} [{track}] | {'ROC-AUC' if track=='C' else 'R²'}: "
                    f"{m.get('roc_auc', m.get('r2', '?')):.4f}")

    # Save per-model metrics JSON
    metrics_path = DIRS["metrics"] / f"{model_name}_{track}_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    # ── Curves (Track C only) ─────────────────────────────────────────────
    if track == "C":
        val_pred_idx = next(i for i, (s, *_) in enumerate(splits_data) if s == "val")
        _, _, val_prob, val_y = splits_data[val_pred_idx]
        try:
            roc = compute_roc_curve_data(val_y, val_prob)
            pr = compute_pr_curve_data(val_y, val_prob)
            cal = compute_calibration_curve_data(val_y, val_prob)
            for name, data in [("roc", roc), ("pr", pr), ("calibration", cal)]:
                curve_path = DIRS["curves"] / f"{model_name}_{track}_{name}_curve.json"
                curve_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"  Curve computation failed: {e}")

    # ── Persist Model ─────────────────────────────────────────────────────
    model_dir = DIRS[f"models_{track}"]
    if hasattr(model, "_net") or hasattr(model, "_model") and isinstance(
            getattr(model, "_model", None), torch.nn.Module):
        # PyTorch model — save state_dict
        net = getattr(model, "_net", getattr(model, "_model", None))
        if net is not None and hasattr(net, "state_dict"):
            pt_path = model_dir / f"{model_name}.pt"
            torch.save(net.state_dict(), str(pt_path))
            logger.info(f"  Model saved -> {pt_path.name}")
    else:
        jl_path = model_dir / f"{model_name}.joblib"
        joblib.dump(model, str(jl_path), compress=3)
        logger.info(f"  Model saved -> {jl_path.name}")

    result["metrics"] = all_metrics
    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    track: str,
    models: List[str],
    do_tune: bool,
    n_trials: int,
    open_test: bool,
    seed: int,
    quick: bool,
) -> None:
    set_seed(seed)
    logger.info(f"=== ERI Model Zoo -- Track {track} ======================")
    logger.info(f"  Models: {models}  |  tune={do_tune}  |  trials={n_trials}  |  quick={quick}")

    train_df, val_df, _ = _load_data(track)
    registry = _build_model_registry(track, seed, quick)

    if "all" in models:
        selected = list(registry.keys())
    else:
        selected = [m for m in models if m in registry]
        unknown = [m for m in models if m not in registry]
        if unknown:
            logger.warning(f"Unknown models (skipped): {unknown}")

    logger.info(f"  Running {len(selected)} models: {selected}")

    all_results: List[Dict] = []
    failures: List[str] = []

    for model_name in selected:
        model = registry[model_name]
        try:
            result = run_one_model(
                model_name=model_name,
                model=model,
                track=track,
                train_df=train_df,
                val_df=val_df,
                do_tune=do_tune,
                n_trials=n_trials,
                open_test=open_test,
                seed=seed,
            )
            all_results.append(result)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[X] [{model_name}] FAILED: {e}\n{tb}")
            failures.append(model_name)
            all_results.append({
                "model_name": model_name, "track": track,
                "status": "failed", "error": str(e),
            })

    # ── Combine all metrics into all_splits_metrics.csv ──────────────────
    rows = []
    for r in all_results:
        if r.get("status") == "success" and r.get("metrics"):
            for m in r["metrics"]:
                rows.append(m)

    if rows:
        combined_df = pd.DataFrame(rows)
        # Ensure exact column ordering: model, track, split, samples, then metrics
        priority_cols = ["model", "track", "split", "samples"]
        other_cols = [c for c in combined_df.columns if c not in priority_cols]
        combined_df = combined_df[priority_cols + other_cols]
        csv_path = DIRS["metrics"] / "all_splits_metrics.csv"
        combined_df.to_csv(str(csv_path), index=False)
        logger.info(f"Combined metrics -> {csv_path}")

    # ── Training report ───────────────────────────────────────────────────
    report = {
        "track": track,
        "total_models": len(selected),
        "successful": len(selected) - len(failures),
        "failed": failures,
        "results": [
            {"model": r["model_name"], "status": r["status"], "error": r.get("error")}
            for r in all_results
        ],
    }
    report_path = ROOT / "artifacts" / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(f"=== Done: {len(selected) - len(failures)}/{len(selected)} succeeded. "
                f"Failures: {failures or 'none'}")

    if failures:
        logger.warning(f"Failed models: {failures}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ERI Model Zoo Runner")
    parser.add_argument("--track", required=True, choices=["C", "R"],
                        help="C=classification, R=regression")
    parser.add_argument("--models", default="all",
                        help="Comma-separated model names or 'all'")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 Optuna trials, reduced deep model epochs")
    parser.add_argument("--no-tune", dest="no_tune", action="store_true",
                        help="Skip Optuna tuning (use defaults)")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Optuna trial budget per model (default: 30)")
    parser.add_argument("--open-test", action="store_true",
                        help="Evaluate on sealed test set (use only for finalists)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models_list = [m.strip() for m in args.models.split(",")]
    n_trials = 3 if args.quick else args.n_trials

    run_all(
        track=args.track,
        models=models_list,
        do_tune=not args.no_tune,
        n_trials=n_trials,
        open_test=args.open_test,
        seed=args.seed,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
