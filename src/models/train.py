"""Unified Model Training and Optuna Hyperparameter Optimization Engine.

Implements:
- Common interface for REGRESSION (PRI 0-100) and CLASSIFICATION (Low/Moderate/High).
- Optuna Bayesian optimization strictly on training CV folds with fixed seeds.
- Automated feature set comparison (raw 15 items vs. 5 factor scores).
- Metric tracking: CV mean+std, train time, inference latency, model size, and overfitting gap.
- CLI interface supporting `--track`, `--models`, `--trials`, and `--quick` mode.
- Model persistence to /artifacts/models/{track}/{model_name}.joblib with audit metadata.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, KFold
import yaml

from src.data.splits import load_split
from src.features.build_features import (
    RAW_ITEM_COLS,
    FACTOR_COLS,
    compute_factor_scores,
    extract_feature_set,
    extract_targets,
)
from src.models.registry import (
    MODEL_CATEGORIES,
    build_model,
    sample_optuna_params,
    list_registered_models,
)
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("models.train")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    _OPTUNA_AVAILABLE = False


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    track: str,
) -> Dict[str, float]:
    """Compute primary and secondary evaluation metrics."""
    if track == "regression":
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "primary": round(rmse, 4),  # Lower is better
        }
    else:
        acc = float(accuracy_score(y_true, y_pred))
        b_acc = float(balanced_accuracy_score(y_true, y_pred))
        f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        prec_mac = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_mac = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        return {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(b_acc, 4),
            "macro_f1": round(f1_mac, 4),
            "macro_precision": round(prec_mac, 4),
            "macro_recall": round(rec_mac, 4),
            "primary": round(f1_mac, 4),  # Higher is better
        }


def cross_validate_model(
    model_name: str,
    params: Dict[str, Any],
    X_raw: pd.DataFrame,
    X_factors: pd.DataFrame,
    y: pd.Series,
    feature_set: str,
    track: str,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Execute leak-free K-Fold cross validation on training data only."""
    X = X_raw if feature_set == "raw_items" else X_factors

    if track == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_gen = cv.split(X, y)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_gen = cv.split(X)

    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(split_gen):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        model = build_model(model_name, track=track, params=params, seed=seed + fold_idx)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        metrics = compute_metrics(y_va.values, preds, track=track)
        fold_metrics.append(metrics)

    # Aggregate mean and standard deviation
    metric_keys = fold_metrics[0].keys()
    mean_metrics = {k: round(float(np.mean([m[k] for m in fold_metrics])), 4) for k in metric_keys}
    std_metrics = {f"{k}_std": round(float(np.std([m[k] for m in fold_metrics])), 4) for k in metric_keys}

    return mean_metrics, std_metrics


def train_single_model(
    model_name: str,
    track: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_trials: int = 15,
    quick: bool = False,
    seed: int = 42,
    output_dir: str = "artifacts/models",
) -> Optional[Dict[str, Any]]:
    """Tune and train a single model on training split and evaluate against validation set."""
    category = MODEL_CATEGORIES.get(model_name, "Unknown")
    is_reg = track == "regression"
    logger.info("Training [%s] '%s' (%s track, trials=%d)...", category, model_name, track, n_trials)

    # Prepare feature sets
    train_factors = compute_factor_scores(train_df)[FACTOR_COLS]
    val_factors = compute_factor_scores(val_df)[FACTOR_COLS]
    train_raw = train_df[RAW_ITEM_COLS]
    val_raw = val_df[RAW_ITEM_COLS]

    train_targets = extract_targets(train_df)
    val_targets = extract_targets(val_df)
    y_train = train_targets["pri_score"] if is_reg else train_targets["risk_category"]
    y_val = val_targets["pri_score"] if is_reg else val_targets["risk_category"]

    start_time = time.time()

    # Optuna Objective
    def objective(trial: Any) -> float:
        f_set = trial.suggest_categorical("feature_set", ["raw_items", "factor_scores"])
        params = sample_optuna_params(model_name, trial)
        cv_means, _ = cross_validate_model(
            model_name=model_name,
            params=params,
            X_raw=train_raw,
            X_factors=train_factors,
            y=y_train,
            feature_set=f_set,
            track=track,
            n_splits=3 if quick else 5,
            seed=seed,
        )
        return cv_means["primary"]

    best_params: Dict[str, Any] = {}
    best_feature_set = "factor_scores"

    # Optimize or use defaults for dummy baseline
    if model_name == "dummy" or not _OPTUNA_AVAILABLE or n_trials <= 1:
        best_params = {}
        best_feature_set = "factor_scores"
    else:
        direction = "minimize" if is_reg else "maximize"
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials, timeout=120)
        best_params = study.best_params.copy()
        best_feature_set = best_params.pop("feature_set", "factor_scores")

    # Final Cross-Validation with Best Parameters
    cv_means, cv_stds = cross_validate_model(
        model_name=model_name,
        params=best_params,
        X_raw=train_raw,
        X_factors=train_factors,
        y=y_train,
        feature_set=best_feature_set,
        track=track,
        n_splits=5,
        seed=seed,
    )

    # Fit final model on complete training set
    X_train_final = train_raw if best_feature_set == "raw_items" else train_factors
    X_val_final = val_raw if best_feature_set == "raw_items" else val_factors

    final_model = build_model(model_name, track=track, params=best_params, seed=seed)
    final_model.fit(X_train_final, y_train)

    train_time_sec = round(time.time() - start_time, 3)

    # Train and Validation evaluations for Overfitting Gap
    train_preds = final_model.predict(X_train_final)
    val_preds = final_model.predict(X_val_final)

    train_eval = compute_metrics(y_train.values, train_preds, track=track)
    val_eval = compute_metrics(y_val.values, val_preds, track=track)

    primary_train = train_eval["primary"]
    primary_val = val_eval["primary"]
    overfitting_gap = round(abs(primary_train - primary_val), 4)

    # Inference Latency Benchmark (100 samples)
    benchmark_samples = X_val_final.head(100)
    lat_start = time.perf_counter()
    _ = final_model.predict(benchmark_samples)
    lat_end = time.perf_counter()
    latency_ms = round((lat_end - lat_start) * 1000.0, 3)

    # Persist model artifact
    track_dir = Path(output_dir) / track
    track_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_path = track_dir / f"{model_name}.joblib"
    joblib.dump(final_model, model_artifact_path)
    model_size_kb = round(os.path.getsize(model_artifact_path) / 1024.0, 2)

    metadata: Dict[str, Any] = {
        "model_name": model_name,
        "category": category,
        "track": track,
        "feature_set": best_feature_set,
        "best_params": best_params,
        "cv_metrics_mean": cv_means,
        "cv_metrics_std": cv_stds,
        "train_metrics": train_eval,
        "validation_metrics": val_eval,
        "overfitting_gap": overfitting_gap,
        "train_time_sec": train_time_sec,
        "inference_latency_ms": latency_ms,
        "model_size_kb": model_size_kb,
        "artifact_path": str(model_artifact_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    meta_path = track_dir / f"{model_name}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Model '%s' trained successfully! Val Primary: %.4f | Overfitting Gap: %.4f | Time: %.2fs",
        model_name, primary_val, overfitting_gap, train_time_sec,
    )
    return metadata


def run_training_suite(
    track: str = "both",
    models: Optional[List[str]] = None,
    n_trials: int = 15,
    quick: bool = False,
    seed: int = 42,
    output_dir: str = "artifacts/models",
    leaderboard_dir: str = "artifacts/leaderboards",
) -> Dict[str, pd.DataFrame]:
    """Execute complete training benchmark over requested tracks and models."""
    set_seed(seed)
    train_df = load_split("train")
    val_df = load_split("val")

    tracks_to_run = ["regression", "classification"] if track == "both" else [track]
    leaderboards: Dict[str, pd.DataFrame] = {}
    lb_dir = Path(leaderboard_dir)
    lb_dir.mkdir(parents=True, exist_ok=True)

    for trk in tracks_to_run:
        available_models = list_registered_models(trk)
        if models and "all" not in models:
            selected_models = [m for m in models if m in available_models]
        else:
            selected_models = available_models

        # In quick mode, select a representative subset
        if quick:
            representative_models = [
                "dummy", "ridge", "decision_tree",
                "random_forest", "hist_gradient_boosting",
            ]
            selected_models = [m for m in selected_models if m in representative_models]
            n_trials = 1

        logger.info("Starting Benchmark for Track: '%s' (%d models)...", trk, len(selected_models))
        results: List[Dict[str, Any]] = []

        for m_name in selected_models:
            try:
                res = train_single_model(
                    model_name=m_name,
                    track=trk,
                    train_df=train_df,
                    val_df=val_df,
                    n_trials=n_trials,
                    quick=quick,
                    seed=seed,
                    output_dir=output_dir,
                )
                if res:
                    results.append(res)
            except Exception as e:
                logger.warning("Skipping model '%s' due to installation/training error: %s", m_name, str(e))

        # Build Leaderboard DataFrame
        lb_records: List[Dict[str, Any]] = []
        for r in results:
            rec = {
                "model_name": r["model_name"],
                "category": r["category"],
                "best_feature_set": r["feature_set"],
                "val_primary_metric": r["validation_metrics"]["primary"],
                "overfitting_gap": r["overfitting_gap"],
                "train_time_sec": r["train_time_sec"],
                "latency_ms": r["inference_latency_ms"],
                "size_kb": r["model_size_kb"],
            }
            if trk == "regression":
                rec["val_rmse"] = r["validation_metrics"]["rmse"]
                rec["val_mae"] = r["validation_metrics"]["mae"]
                rec["val_r2"] = r["validation_metrics"]["r2"]
            else:
                rec["val_macro_f1"] = r["validation_metrics"]["macro_f1"]
                rec["val_accuracy"] = r["validation_metrics"]["accuracy"]
                rec["val_balanced_acc"] = r["validation_metrics"]["balanced_accuracy"]
            lb_records.append(rec)

        lb_df = pd.DataFrame(lb_records)
        if not lb_df.empty:
            sort_ascending = True if trk == "regression" else False
            lb_df = lb_df.sort_values(by="val_primary_metric", ascending=sort_ascending).reset_index(drop=True)

            csv_path = lb_dir / f"{trk}_leaderboard.csv"
            md_path = lb_dir / f"{trk}_leaderboard.md"
            json_path = lb_dir / f"{trk}_leaderboard.json"

            lb_df.to_csv(csv_path, index=False)
            try:
                md_content = lb_df.to_markdown(index=False)
            except Exception:
                md_content = lb_df.to_string(index=False)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            logger.info("Saved %s leaderboard to %s", trk, csv_path)
            leaderboards[trk] = lb_df

    return leaderboards


def main() -> None:
    """CLI execution entrypoint for model training suite."""
    parser = argparse.ArgumentParser(description="Psychosocial Intelligence Model Zoo Training Engine")
    parser.add_argument("--track", choices=["regression", "classification", "both"], default="both")
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'")
    parser.add_argument("--trials", type=int, default=15, help="Number of Optuna tuning trials per model")
    parser.add_argument("--quick", action="store_true", help="Quick mode for fast smoke testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    models_list = [m.strip() for m in args.models.split(",")] if args.models != "all" else None

    print("\n" + "=" * 65)
    print("PSYCHOSOCIAL RISK MODEL ZOO TRAINING ENGINE")
    print(f"Track:       {args.track}")
    print(f"Models:      {args.models}")
    print(f"Trials:      {1 if args.quick else args.trials}")
    print(f"Quick Mode:  {args.quick}")
    print(f"Seed:        {args.seed}")
    print("=" * 65 + "\n")

    leaderboards = run_training_suite(
        track=args.track,
        models=models_list,
        n_trials=args.trials,
        quick=args.quick,
        seed=args.seed,
    )

    for trk, df in leaderboards.items():
        print(f"\nLeaderboard: {trk.upper()} (Top Models)")
        print("-" * 65)
        print(df.to_string(index=False))
        print("-" * 65)


if __name__ == "__main__":
    main()
