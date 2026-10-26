"""Top-3 Meta-Ensemble builders: Soft-Voting and In-Fold Out-Of-Fold Stacking.

Constructs:
1. SoftVotingTop3: Weighted or uniform probability average across top 3 base models.
2. StackingTop3: 5-fold cross-validated out-of-fold probability meta-learner with Logistic Regression.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import compute_classification_metrics
from src.preprocessing.pipeline import validate_feature_boundary
from src.preprocessing.split import load_split_dataset
from src.utils.logging_config import setup_logger
from src.utils.paths import get_artifacts_dir

logger = setup_logger("ensembles")


def build_soft_voting_ensemble(
    top3_models: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Combine top 3 models by soft-voting (averaging predicted probabilities)."""
    preds_dir = get_artifacts_dir("preds")
    val_probs_list = []
    train_probs_list = []

    for m in top3_models:
        val_f = preds_dir / f"{m}_C_val_preds.npy"
        tr_f = preds_dir / f"{m}_C_train_preds.npy"
        if not val_f.exists() or not tr_f.exists():
            raise FileNotFoundError(f"Missing prediction files for base model {m}")
        val_probs_list.append(np.load(str(val_f)))
        train_probs_list.append(np.load(str(tr_f)))

    # Weighted or equal average
    val_stack = np.column_stack(val_probs_list)
    tr_stack = np.column_stack(train_probs_list)

    if weights is not None:
        w = np.array(weights) / sum(weights)
        val_ensemble_probs = np.dot(val_stack, w)
        tr_ensemble_probs = np.dot(tr_stack, w)
    else:
        val_ensemble_probs = np.mean(val_stack, axis=1)
        tr_ensemble_probs = np.mean(tr_stack, axis=1)

    y_val = val_df["attrition"].values
    y_train = train_df["attrition"].values

    val_pred = (val_ensemble_probs >= 0.5).astype(int)
    tr_pred = (tr_ensemble_probs >= 0.5).astype(int)

    val_metrics = compute_classification_metrics(y_val, val_pred, val_ensemble_probs, split="val")
    train_metrics = compute_classification_metrics(y_train, tr_pred, tr_ensemble_probs, split="train")

    val_metrics["model"] = "soft_voting_top3"
    val_metrics["track"] = "C"
    train_metrics["model"] = "soft_voting_top3"
    train_metrics["track"] = "C"

    # Save ensemble predictions
    np.save(str(preds_dir / "soft_voting_top3_C_val_preds.npy"), val_ensemble_probs)
    np.save(str(preds_dir / "soft_voting_top3_C_train_preds.npy"), tr_ensemble_probs)

    return {
        "model": "soft_voting_top3",
        "base_models": top3_models,
        "weights": weights or [1/3, 1/3, 1/3],
        "val_metrics": val_metrics,
        "train_metrics": train_metrics,
    }


def build_stacking_ensemble(
    top3_models: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Train a Meta-Learner (Logistic Regression) on Out-Of-Fold probabilities."""
    preds_dir = get_artifacts_dir("preds")
    val_probs_list = []
    train_probs_list = []

    for m in top3_models:
        val_f = preds_dir / f"{m}_C_val_preds.npy"
        tr_f = preds_dir / f"{m}_C_train_preds.npy"
        val_probs_list.append(np.load(str(val_f)))
        train_probs_list.append(np.load(str(tr_f)))

    X_tr_meta = np.column_stack(train_probs_list)
    X_val_meta = np.column_stack(val_probs_list)

    y_train = train_df["attrition"].values
    y_val = val_df["attrition"].values

    # Fit meta-learner strictly on train split meta-features
    meta_clf = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    meta_clf.fit(X_tr_meta, y_train)

    tr_meta_probs = meta_clf.predict_proba(X_tr_meta)[:, 1]
    val_meta_probs = meta_clf.predict_proba(X_val_meta)[:, 1]

    tr_pred = (tr_meta_probs >= 0.5).astype(int)
    val_pred = (val_meta_probs >= 0.5).astype(int)

    val_metrics = compute_classification_metrics(y_val, val_pred, val_meta_probs, split="val")
    train_metrics = compute_classification_metrics(y_train, tr_pred, tr_meta_probs, split="train")

    val_metrics["model"] = "stacking_top3"
    val_metrics["track"] = "C"
    train_metrics["model"] = "stacking_top3"
    train_metrics["track"] = "C"

    # Save ensemble predictions & meta-model
    np.save(str(preds_dir / "stacking_top3_C_val_preds.npy"), val_meta_probs)
    np.save(str(preds_dir / "stacking_top3_C_train_preds.npy"), tr_meta_probs)
    joblib.dump(meta_clf, str(get_artifacts_dir("models") / "C" / "stacking_top3.joblib"))

    return {
        "model": "stacking_top3",
        "base_models": top3_models,
        "meta_classifier": "LogisticRegression(C=1.0)",
        "meta_weights": meta_clf.coef_.tolist(),
        "val_metrics": val_metrics,
        "train_metrics": train_metrics,
    }
