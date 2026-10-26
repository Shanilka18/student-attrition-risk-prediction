"""Asynchronous retraining worker for user-submitted datasets with outcome labels.

Maintains strict separation between the primary synthetic research benchmark
and any secondary user-submitted datasets.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, f1_score, accuracy_score

from src.data.preprocess import PsychometricFeatureTransformer
from src.features.build_features import FACTOR_ITEMS, FACTOR_NAMES, compute_factor_scores
from src.api.storage import storage
from src.utils.logger import get_logger

logger = get_logger("api.retrain")

USER_ARTIFACTS_DIR = Path("artifacts/user_dataset")
USER_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def execute_retraining_job(
    job_id: str,
    df: pd.DataFrame,
    target_column: str = "attrition_label",
    random_seed: int = 42,
) -> None:
    """Run full retraining pipeline on user-submitted dataset in the background.

    Args:
        job_id: Unique job tracking identifier.
        df: Input DataFrame with questionnaire items and outcome column.
        target_column: Column name representing supervised target.
        random_seed: Fixed random seed for reproducibility.
    """
    logger.info("Starting background retraining job '%s' on %d records...", job_id, len(df))
    storage.update_job(job_id, status="running", progress_pct=5.0, message="Validating dataset and schema")

    try:
        # 1. Verify target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset. Available: {list(df.columns)}")

        storage.update_job(job_id, status="running", progress_pct=15.0, message="Splitting training and validation folds")
        
        # Clean target and features
        valid_df = df.dropna(subset=[target_column]).copy()
        if len(valid_df) < 20:
            raise ValueError(f"Insufficient labeled rows for retraining (found {len(valid_df)}, minimum required: 20)")

        # Determine task type
        unique_targets = valid_df[target_column].unique()
        is_classification = len(unique_targets) <= 5 or valid_df[target_column].dtype == object

        # Train/val split
        stratify = valid_df[target_column] if is_classification and len(unique_targets) > 1 else None
        try:
            train_df, val_df = train_test_split(
                valid_df,
                test_size=0.25,
                random_state=random_seed,
                stratify=stratify,
            )
        except Exception:
            train_df, val_df = train_test_split(
                valid_df,
                test_size=0.25,
                random_state=random_seed,
            )

        storage.update_job(job_id, status="running", progress_pct=30.0, message="Fitting feature engineering and psychometric pipeline")
        
        # 2. Preprocessing
        required_items = [item for items in FACTOR_ITEMS.values() for item in items]
        preprocessor = PsychometricFeatureTransformer()
        preprocessor.fit(train_df[required_items])

        train_trans = preprocessor.transform(train_df[required_items])
        val_trans = preprocessor.transform(val_df[required_items])

        factor_cols = [f"factor_{f}" for f in FACTOR_NAMES]
        X_train = train_trans[factor_cols]
        X_val = val_trans[factor_cols]

        y_train = train_df[target_column].values
        y_val = val_df[target_column].values

        storage.update_job(job_id, status="running", progress_pct=50.0, message="Training benchmark model candidate suite")

        leaderboard_records = []

        if not is_classification:
            # Regression Track
            models = {
                "ridge": Ridge(alpha=1.0, random_state=random_seed),
                "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=100, random_state=random_seed),
                "random_forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=random_seed),
                "dummy": DummyRegressor(strategy="mean"),
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                r2 = float(r2_score(y_val, preds))
                rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
                leaderboard_records.append({
                    "model_name": name,
                    "primary_metric": "r2",
                    "r2": round(r2, 4),
                    "rmse": round(rmse, 4),
                })
            
            leaderboard_records.sort(key=lambda x: x["r2"], reverse=True)
            champ_model = leaderboard_records[0]["model_name"]
            champ_metric = f"R² = {leaderboard_records[0]['r2']}"

        else:
            # Classification Track
            # Convert targets to strings if needed
            y_train_str = [str(v) for v in y_train]
            y_val_str = [str(v) for v in y_val]

            models = {
                "logistic_regression": LogisticRegression(max_iter=500, random_state=random_seed),
                "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=100, random_state=random_seed),
                "random_forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_seed),
                "dummy": DummyClassifier(strategy="most_frequent"),
            }

            for name, model in models.items():
                model.fit(X_train, y_train_str)
                preds = model.predict(X_val)
                acc = float(accuracy_score(y_val_str, preds))
                f1 = float(f1_score(y_val_str, preds, average="macro", zero_division=0))
                leaderboard_records.append({
                    "model_name": name,
                    "primary_metric": "macro_f1",
                    "macro_f1": round(f1, 4),
                    "accuracy": round(acc, 4),
                })

            leaderboard_records.sort(key=lambda x: x["macro_f1"], reverse=True)
            champ_model = leaderboard_records[0]["model_name"]
            champ_metric = f"Macro F1 = {leaderboard_records[0]['macro_f1']}"

        storage.update_job(job_id, status="running", progress_pct=85.0, message="Compiling user dataset leaderboard and champion report")

        # Assign ranks
        for idx, item in enumerate(leaderboard_records):
            item["rank"] = idx + 1
            item["is_champion"] = (idx == 0)

        results_payload = {
            "dataset_type": "user_dataset",
            "task_type": "classification" if is_classification else "regression",
            "target_column": target_column,
            "training_samples": len(train_df),
            "validation_samples": len(val_df),
            "champion_model": champ_model,
            "champion_performance": champ_metric,
            "leaderboard": leaderboard_records,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "notes": "Trained exclusively on user-uploaded dataset; isolated from synthetic benchmark pipeline.",
        }

        # Save user dataset leaderboard to disk
        out_json = USER_ARTIFACTS_DIR / f"user_leaderboard_{job_id}.json"
        import json
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2)

        storage.update_job(
            job_id,
            status="completed",
            progress_pct=100.0,
            message=f"Retraining completed successfully. Champion: {champ_model} ({champ_metric})",
            results=results_payload,
        )
        logger.info("Retraining job '%s' finished successfully. Champion: %s", job_id, champ_model)

    except Exception as e:
        logger.error("Retraining job '%s' failed: %s", job_id, e, exc_info=True)
        storage.update_job(
            job_id,
            status="failed",
            progress_pct=100.0,
            message=f"Retraining failed: {str(e)}",
            results=None,
        )
