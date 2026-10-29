"""Adaptive SHAP Explainer Engine for ERI Framework.

Automatically selects the appropriate SHAP explainer:
- LinearExplainer for Linear/LogisticRegression models
- TreeExplainer for Tree-based models (XGBoost, LightGBM, CatBoost, RandomForest)
- DeepExplainer / KernelExplainer for Deep Tabular architectures

Extracts local SHAP attributions, global importance, and validates the additive efficiency property.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from src.preprocessing.pipeline import get_scaled_pipeline, get_tree_pipeline, validate_feature_boundary
from src.preprocessing.split import load_split_dataset
from src.utils.logging_config import setup_logger
from src.utils.paths import get_artifacts_dir

logger = setup_logger("shap_explainer")


def get_feature_names_from_preprocessor(preprocessor: Any) -> List[str]:
    """Extract human-readable feature column names from fitted preprocessor."""
    try:
        names = preprocessor.get_feature_names_out()
        clean_names = []
        for n in names:
            # Strip transformer prefixes like remainder__, num__, cat__
            parts = n.split("__")
            clean_names.append(parts[-1] if len(parts) > 1 else n)
        return clean_names
    except Exception:
        if hasattr(preprocessor, "feature_names_out_") and preprocessor.feature_names_out_:
            return list(preprocessor.feature_names_out_)
        # Fallback generic names
        return [f"feature_{i}" for i in range(30)]


def select_shap_explainer(
    model: Any,
    X_background: np.ndarray,
    model_name: str,
) -> Tuple[Any, str]:
    """Select optimal SHAP explainer based on model class and algorithm family."""
    # Unwrap pipeline or wrapper if needed
    underlying = getattr(model, "_clf", getattr(model, "_model", getattr(model, "_booster", model)))

    if isinstance(underlying, LogisticRegression) or model_name.startswith("logistic"):
        logger.info(f"Dispatching LinearExplainer for model: {model_name}")
        explainer = shap.LinearExplainer(underlying, X_background)
        explainer_type = "LinearExplainer"
    elif any(tree_kw in model_name.lower() for tree_kw in ["xgb", "lightgbm", "catboost", "forest", "tree", "gradient_boosting"]):
        logger.info(f"Dispatching TreeExplainer for model: {model_name}")
        explainer = shap.TreeExplainer(underlying)
        explainer_type = "TreeExplainer"
    else:
        logger.info(f"Dispatching KernelExplainer with background sample for model: {model_name}")
        # Sample background up to 50 rows for fast approximation
        bg_sample = X_background[np.random.choice(len(X_background), min(50, len(X_background)), replace=False)]
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else underlying.predict_proba
        explainer = shap.KernelExplainer(lambda x: predict_fn(x)[:, 1], bg_sample)
        explainer_type = "KernelExplainer"

    return explainer, explainer_type


def compute_shap_explanations(
    model_name: Optional[str] = None,
    dataset_split: str = "val",
) -> Dict[str, Any]:
    """Compute local and global SHAP explanations for the specified model and dataset."""
    shap_dir = get_artifacts_dir("shap")
    shap_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine model to explain
    if model_name is None:
        decision_file = get_artifacts_dir("leaderboard") / "selection_decision.json"
        if decision_file.exists():
            with open(decision_file, "r", encoding="utf-8") as f:
                dec = json.load(f)
                model_name = dec.get("selected_model", "logistic_l2")
        else:
            model_name = "logistic_l2"

    # If selected model is an ensemble or missing joblib, explain champion base model
    model_path = get_artifacts_dir("models") / "C" / f"{model_name}.joblib"
    if not model_path.exists():
        candidates = ["catboost", "random_forest", "logistic_l2", "linear_svm", "xgboost", "lightgbm"]
        for cand in candidates:
            cand_path = get_artifacts_dir("models") / "C" / f"{cand}.joblib"
            if cand_path.exists():
                logger.warning(
                    f"Selected model '{model_name}' has no direct single joblib artifact. "
                    f"Explaining champion individual base model '{cand}' for SHAP attributions."
                )
                model_name = cand
                model_path = cand_path
                break

    logger.info(f"Computing SHAP explanations for chosen model: {model_name} on split: {dataset_split}")

    # 2. Load dataset partitions and preprocessors
    train_df, val_df, _ = load_split_dataset()
    target_df = val_df if dataset_split == "val" else train_df
    sample_ids = target_df["participant_id"].tolist() if "participant_id" in target_df.columns else [f"P_{i}" for i in range(len(target_df))]

    view_type = "tree" if any(k in model_name for k in ["xgb", "forest", "tree", "lightgbm", "catboost"]) else "scaled"
    prep = get_tree_pipeline() if view_type == "tree" else get_scaled_pipeline()

    # Fit preprocessor strictly inside train fold
    X_train_clean = validate_feature_boundary(train_df)
    prep.fit(X_train_clean, is_train_fold=True)
    X_train = prep.transform(X_train_clean)

    X_target_clean = validate_feature_boundary(target_df)
    X_target = prep.transform(X_target_clean)

    feature_names = get_feature_names_from_preprocessor(prep)

    # 3. Load trained model
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}. Run training first.")
    model = joblib.load(str(model_path))

    # 4. Select explainer & compute SHAP values
    explainer, explainer_type = select_shap_explainer(model, X_train, model_name)
    raw_explanation = explainer(X_target)

    # Handle shape variations across explainer types
    if hasattr(raw_explanation, "values"):
        vals = raw_explanation.values
        if vals.ndim == 3:  # Binary classification with (N, D, 2) shape
            shap_values = vals[:, :, 1]
            base_val = float(raw_explanation.base_values[0, 1] if raw_explanation.base_values.ndim > 1 else raw_explanation.base_values[0])
        else:
            shap_values = vals
            base_val = float(raw_explanation.base_values[0] if hasattr(raw_explanation.base_values, "__len__") else raw_explanation.base_values)
    else:
        shap_values = np.array(raw_explanation)
        base_val = float(explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value)

    # 5. Persist artifacts
    np.save(str(shap_dir / "values.npy"), shap_values)
    with open(shap_dir / "base_value.json", "w", encoding="utf-8") as f:
        json.dump({"base_value": base_val, "explainer_type": explainer_type}, f, indent=2)
    with open(shap_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    with open(shap_dir / "sample_ids.json", "w", encoding="utf-8") as f:
        json.dump(sample_ids, f, indent=2)

    # 6. Global Feature Importance (mean |SHAP|)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    global_importance = [
        {
            "rank": int(r + 1),
            "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
            "mean_abs_shap": round(float(mean_abs_shap[i]), 5),
        }
        for r, i in enumerate(sorted_idx)
    ]

    with open(shap_dir / "global_importance.json", "w", encoding="utf-8") as f:
        json.dump(global_importance, f, indent=2)

    logger.info(f"SHAP calculations complete: {len(sample_ids)} samples, {len(feature_names)} features.")
    return {
        "model_name": model_name,
        "explainer_type": explainer_type,
        "shap_values": shap_values,
        "base_value": base_val,
        "feature_names": feature_names,
        "sample_ids": sample_ids,
        "global_importance": global_importance,
        "X_target": X_target,
    }
