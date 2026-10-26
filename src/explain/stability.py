"""Cross-Validation SHAP Explanation Stability Diagnostics.

Assesses whether model explanations are stable and reproducible across data partitions:
- Top-k feature overlap (Jaccard similarity index) for k=3 and k=5
- Spearman rank correlation of mean |SHAP| values across folds
- Identification of unstable or sensitive features
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from src.explain.shap_explainer import get_feature_names_from_preprocessor, select_shap_explainer
from src.preprocessing.pipeline import get_scaled_pipeline, get_tree_pipeline, validate_feature_boundary
from src.preprocessing.split import get_cv_folds, load_split_dataset
from src.utils.logging_config import setup_logger
from src.utils.paths import get_artifacts_dir

logger = setup_logger("shap_stability")


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity index between two sets."""
    union = len(set_a.union(set_b))
    if union == 0:
        return 1.0
    return float(len(set_a.intersection(set_b)) / union)


def compute_shap_stability_across_folds(
    model_name: str = "logistic_l2",
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Train candidate model on CV folds and evaluate explanation stability."""
    train_df, _, _ = load_split_dataset()
    y_full = train_df["attrition"].values
    X_clean = validate_feature_boundary(train_df)

    view_type = "tree" if any(k in model_name for k in ["xgb", "forest", "tree", "lightgbm", "catboost"]) else "scaled"
    prep = get_tree_pipeline() if view_type == "tree" else get_scaled_pipeline()
    prep.fit(X_clean, is_train_fold=True)
    X_full = prep.transform(X_clean)
    feature_names = get_feature_names_from_preprocessor(prep)

    fold_mean_abs_shaps = []
    fold_top3 = []
    fold_top5 = []

    # Iterate through CV folds
    indices = list(range(len(train_df)))
    cv = get_cv_folds(train_df, indices=indices, n_splits=n_folds, n_repeats=1, seed=42)
    for fold_idx, (tr_idx, val_idx) in enumerate(cv):
        X_tr, y_tr = X_full[tr_idx], y_full[tr_idx]
        X_va = X_full[val_idx]

        # Fit model on training fold
        if model_name.startswith("logistic"):
            clf = LogisticRegression(C=1.0, solver="lbfgs", random_state=42 + fold_idx, max_iter=2000)
            clf.fit(X_tr, y_tr)
            explainer = select_shap_explainer(clf, X_tr, model_name)[0]
        else:
            clf = LogisticRegression(random_state=42 + fold_idx).fit(X_tr, y_tr)
            explainer = select_shap_explainer(clf, X_tr, model_name)[0]

        # Compute SHAP on validation fold
        exp = explainer(X_va)
        vals = exp.values if hasattr(exp, "values") else np.array(exp)
        if vals.ndim == 3:
            vals = vals[:, :, 1]

        mean_abs = np.mean(np.abs(vals), axis=0)
        fold_mean_abs_shaps.append(mean_abs)

        ranked_indices = np.argsort(mean_abs)[::-1]
        top3_feats = set([feature_names[i] for i in ranked_indices[:3]])
        top5_feats = set([feature_names[i] for i in ranked_indices[:5]])

        fold_top3.append(top3_feats)
        fold_top5.append(top5_feats)

    # 1. Pairwise Jaccard similarity across folds
    jaccard_top3_pairs = []
    jaccard_top5_pairs = []
    spearman_pairs = []

    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            j3 = jaccard_similarity(fold_top3[i], fold_top3[j])
            j5 = jaccard_similarity(fold_top5[i], fold_top5[j])
            jaccard_top3_pairs.append(j3)
            jaccard_top5_pairs.append(j5)

            # Spearman correlation of feature rankings
            rho, _ = stats.spearmanr(fold_mean_abs_shaps[i], fold_mean_abs_shaps[j])
            spearman_pairs.append(float(rho))

    # 2. Identify unstable features (high standard deviation in rank across folds)
    rank_matrix = []
    for m in fold_mean_abs_shaps:
        # 1-based ranks (1 = highest importance)
        ranks = len(m) - stats.rankdata(m) + 1
        rank_matrix.append(ranks)

    rank_matrix = np.array(rank_matrix)  # Shape (n_folds, n_features)
    rank_stds = np.std(rank_matrix, axis=0)
    unstable_indices = np.where(rank_stds > 4.0)[0]  # Rank fluctuates by > 4 positions
    unstable_features = [feature_names[i] for i in unstable_indices if i < len(feature_names)]

    avg_jaccard_3 = round(float(np.mean(jaccard_top3_pairs)), 4)
    avg_jaccard_5 = round(float(np.mean(jaccard_top5_pairs)), 4)
    avg_spearman = round(float(np.mean(spearman_pairs)), 4)

    stability_report = {
        "model_name": model_name,
        "n_folds": n_folds,
        "top3_mean_jaccard_overlap": avg_jaccard_3,
        "top5_mean_jaccard_overlap": avg_jaccard_5,
        "mean_spearman_rank_correlation": avg_spearman,
        "unstable_features": unstable_features,
        "is_stable": bool(avg_spearman >= 0.70 and avg_jaccard_5 >= 0.50),
    }

    shap_dir = get_artifacts_dir("shap")
    shap_dir.mkdir(parents=True, exist_ok=True)
    with open(shap_dir / "stability.json", "w", encoding="utf-8") as f:
        json.dump(stability_report, f, indent=2)

    logger.info(
        f"SHAP Stability Audit: Spearman={avg_spearman:.4f}, Top-5 Jaccard={avg_jaccard_5:.4f}, "
        f"Unstable features: {len(unstable_features)}"
    )
    return stability_report
