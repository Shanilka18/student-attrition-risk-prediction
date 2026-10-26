"""Automated Explainer Selection and Caching Engine.

Inspects model architectures and dynamically instantiates the optimal SHAP explainer:
- TreeExplainer for GBDT and Tree Ensembles (fast $O(T L D^2)$ exact polynomial evaluation).
- LinearExplainer for regularized linear models (Ridge, Lasso, ElasticNet, Logistic).
- Permutation / KernelExplainer fallback for non-differentiable / meta-ensembles.
- Caches computed attributions to /artifacts/shap for reproducible downstream reporting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
import joblib

from src.utils.logger import get_logger

logger = get_logger("explain.explainer")

# Directory constants
SHAP_DIR = Path("artifacts/shap")


def detect_explainer_type(model: Any) -> str:
    """Determine the optimal SHAP explainer class for a given model.

    Args:
        model: Fitted estimator instance or pipeline.

    Returns:
        One of 'tree', 'linear', or 'kernel'.
    """
    model_name = type(model).__name__.lower()

    # Unwrap pipeline if needed
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        estimator = model.named_steps["model"]
        model_name = type(estimator).__name__.lower()
    elif hasattr(model, "estimator"):
        estimator = model.estimator
        model_name = type(estimator).__name__.lower()

    tree_keywords = [
        "tree", "forest", "gradientboosting", "histgradientboosting",
        "xgb", "lgbm", "lightgbm", "catboost"
    ]
    linear_keywords = [
        "ridge", "lasso", "elasticnet", "linearregression",
        "logisticregression", "sgd", "linearsvc"
    ]

    for kw in tree_keywords:
        if kw in model_name:
            return "tree"

    for kw in linear_keywords:
        if kw in model_name:
            return "linear"

    return "kernel"


def build_explainer(
    model: Any,
    X_background: Union[pd.DataFrame, np.ndarray],
    explainer_type: Optional[str] = None,
    seed: int = 42,
) -> Tuple[Any, str]:
    """Instantiate the best SHAP explainer for a model with graceful fallbacks.

    Args:
        model: Fitted model estimator.
        X_background: Reference background dataset for expectation calculation.
        explainer_type: Explicit override ('tree', 'linear', 'kernel') or None for auto.
        seed: Random state for permutation sampling.

    Returns:
        Tuple of (explainer_object, selected_type_name).
    """
    import shap

    chosen_type = explainer_type or detect_explainer_type(model)
    logger.info("Attempting to build '%s' SHAP explainer for %s...", chosen_type, type(model).__name__)

    # Subsample background if large to maintain real-time responsiveness
    bg = X_background
    if isinstance(bg, pd.DataFrame):
        if len(bg) > 200:
            bg = bg.sample(n=200, random_state=seed)
    elif isinstance(bg, np.ndarray):
        if len(bg) > 200:
            rng = np.random.default_rng(seed)
            bg = bg[rng.choice(len(bg), size=200, replace=False)]

    # 1. TreeExplainer
    if chosen_type == "tree":
        try:
            explainer = shap.TreeExplainer(model, data=bg)
            logger.info("Successfully initialized TreeExplainer.")
            return explainer, "tree"
        except Exception as e:
            logger.warning("TreeExplainer initialization failed: %s. Falling back to shap.Explainer.", e)
            try:
                explainer = shap.Explainer(model, bg)
                return explainer, "explainer_generic"
            except Exception as e2:
                logger.warning("shap.Explainer failed: %s. Falling back to PermutationExplainer.", e2)

    # 2. LinearExplainer
    elif chosen_type == "linear":
        try:
            explainer = shap.LinearExplainer(model, data=bg)
            logger.info("Successfully initialized LinearExplainer.")
            return explainer, "linear"
        except Exception as e:
            logger.warning("LinearExplainer initialization failed: %s. Falling back to shap.Explainer.", e)
            try:
                explainer = shap.Explainer(model, bg)
                return explainer, "explainer_generic"
            except Exception as e2:
                logger.warning("Generic Explainer failed: %s. Falling back to PermutationExplainer.", e2)

    # 3. Permutation / Kernel Fallback
    try:
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        explainer = shap.PermutationExplainer(predict_fn, bg)
        logger.info("Successfully initialized PermutationExplainer.")
        return explainer, "permutation"
    except Exception as e:
        logger.warning("PermutationExplainer failed: %s. Falling back to KernelExplainer.", e)
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        explainer = shap.KernelExplainer(predict_fn, bg)
        logger.info("Successfully initialized KernelExplainer.")
        return explainer, "kernel"


def explain_dataset(
    explainer: Any,
    X: Union[pd.DataFrame, np.ndarray],
    track: str = "regression",
    target_class_idx: int = 0,  # Default: class 0 ('High') for scikit-learn sorted classes ['High', 'Low', 'Moderate']
) -> Dict[str, Any]:
    """Compute SHAP values for a dataset and extract primary risk attributions.

    Args:
        explainer: Initialized SHAP explainer.
        X: Dataset to explain.
        track: 'regression' or 'classification'.
        target_class_idx: Index of class to extract attributions for (default 2 = 'High').

    Returns:
        Dict containing:
            - values: 2D numpy array (N, p) representing primary attributions.
            - base_value: Scalar or float representing expected baseline.
            - full_values: Raw SHAP values array or list.
            - feature_names: List of feature names.
    """
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]

    shap_output = explainer(X) if callable(explainer) else explainer.shap_values(X)

    # shap.Explanation object vs. raw numpy array
    if hasattr(shap_output, "values"):
        vals = shap_output.values
        base = shap_output.base_values
    else:
        vals = shap_output
        base = getattr(explainer, "expected_value", 0.0)

    # Multi-class handling: vals shape can be (N, p, C) or list of C arrays of (N, p)
    if isinstance(vals, list):
        # List of arrays for each class
        if track == "classification" and len(vals) > target_class_idx:
            primary_vals = vals[target_class_idx]
            primary_base = base[target_class_idx] if isinstance(base, (list, np.ndarray)) else base
        else:
            primary_vals = vals[0]
            primary_base = base[0] if isinstance(base, (list, np.ndarray)) else base
    elif isinstance(vals, np.ndarray) and len(vals.shape) == 3:
        # 3D array (N, p, C)
        primary_vals = vals[:, :, target_class_idx]
        primary_base = base[:, target_class_idx] if isinstance(base, np.ndarray) and len(base.shape) > 1 else base
    else:
        # Standard 2D array (N, p)
        primary_vals = vals
        primary_base = base

    # Normalize scalar base_value
    if isinstance(primary_base, np.ndarray):
        primary_base = float(np.mean(primary_base))
    else:
        primary_base = float(primary_base)

    return {
        "values": np.asarray(primary_vals, dtype=float),
        "base_value": primary_base,
        "full_values": vals,
        "feature_names": feature_names,
    }


def save_shap_artifacts(
    explanation: Dict[str, Any],
    track: str,
    model_name: str,
    output_dir: Union[Path, str] = SHAP_DIR,
) -> Path:
    """Save SHAP attributions and metadata to disk.

    Args:
        explanation: Dictionary returned by explain_dataset.
        track: 'regression' or 'classification'.
        model_name: Identifier of the champion model.
        output_dir: Destination directory.

    Returns:
        Path to the saved npz archive.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    archive_path = out_path / f"shap_values_{track}_{model_name}.npz"
    meta_path = out_path / f"shap_meta_{track}_{model_name}.json"

    np.savez_compressed(
        archive_path,
        values=explanation["values"],
        base_value=explanation["base_value"],
        feature_names=np.array(explanation["feature_names"]),
    )

    meta = {
        "track": track,
        "model_name": model_name,
        "n_samples": int(explanation["values"].shape[0]),
        "n_features": int(explanation["values"].shape[1]),
        "base_value": float(explanation["base_value"]),
        "feature_names": explanation["feature_names"],
        "archive_file": archive_path.name,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved SHAP artifacts to %s and %s", archive_path, meta_path)
    return archive_path
