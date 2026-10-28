"""Explainability engine for the MEI research pipeline.

Implements global SHAP explanations, cross-fold consistency audits,
cross-method comparisons (Permutation Importance and Logistic Regression),
pattern-stratified attributions, and local plain-English reason generation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from mei.config import ARTIFACTS_DIR, RUNS_DIR
from mei.explain.mapping import compute_grouped_importance, get_feature_group_mapping
from mei.explain.reasons import (
    convert_shap_to_mei_points,
    generate_student_reasons,
    load_reason_templates,
)

logger = logging.getLogger("mei.explain.engine")


def get_active_run_dir(run_dir: Optional[Path] = None) -> Path:
    """Resolve the active run directory from active_run.txt if not provided."""
    if run_dir is not None:
        return Path(run_dir)
    active_file = ARTIFACTS_DIR / "active_run.txt"
    if active_file.exists():
        run_id = active_file.read_text(encoding="utf-8").strip()
        candidate = RUNS_DIR / run_id
        if candidate.exists():
            return candidate
    runs = sorted(RUNS_DIR.glob("run_*"))
    if runs:
        return runs[-1]
    raise FileNotFoundError("No active run directory found.")


def load_full_clean_cohort() -> pd.DataFrame:
    """Load the full validated clean cohort (N >= 2000) from data/interim."""
    for f in sorted(Path("data/interim").glob("clean_run_*.parquet"), reverse=True):
        df = pd.read_parquet(f)
        if len(df) >= 2000:
            return df
    raise FileNotFoundError(
        "Full clean cohort (>= 2000 rows) not found in data/interim."
    )


def get_champion_explainer(
    run_dir: Optional[Path] = None,
    background_samples: int = 30,
) -> Tuple[shap.KernelExplainer, Any, Any, List[str], float]:
    """Load the frozen champion bundle and build the calibrated SHAP KernelExplainer.

    Returns:
        (explainer, pipeline, calibrator, feature_names, baseline_probability)
    """
    active_dir = get_active_run_dir(run_dir)
    champ_dir = active_dir / "champion"

    pipeline = joblib.load(champ_dir / "pipeline.joblib")
    calibrator = joblib.load(champ_dir / "calibrator.joblib")

    with open(champ_dir / "features.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    bg_df = pd.read_parquet(champ_dir / "shap_background_200.parquet")[feature_names]

    # Summarize background with k-means medoids for fast exact KernelExplainer evaluation
    bg_summary = shap.kmeans(bg_df.values, background_samples)

    # Base prediction function returning probability of class 1 (at-risk)
    def predict_fn(x):
        return pipeline.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(predict_fn, bg_summary)
    baseline_p = float(explainer.expected_value)

    return explainer, pipeline, calibrator, feature_names, baseline_p


def compute_global_shap_audit(
    run_dir: Optional[Path] = None,
    n_val_samples: int = 250,
    n_test_samples: int = 250,
) -> Dict[str, Any]:
    """Compute global SHAP explanations on validation and holdout test data."""
    active_dir = get_active_run_dir(run_dir)
    active_dir / "champion"

    explainer, pipeline, calibrator, feature_names, baseline_p = get_champion_explainer(
        active_dir, background_samples=25
    )
    feature_mapping = get_feature_group_mapping(feature_names)

    # 1. Load validation data
    val_agg = pd.read_parquet(active_dir / "features" / "val_agg.parquet")[
        feature_names
    ]
    if len(val_agg) > n_val_samples:
        val_sample = val_agg.sample(n=n_val_samples, random_state=42).reset_index(
            drop=True
        )
    else:
        val_sample = val_agg.copy()

    logger.info(
        "Computing SHAP values for validation sample (%d rows)...", len(val_sample)
    )
    val_shap = explainer.shap_values(
        val_sample.values, nsamples=100, l1_reg="num_features(10)"
    )
    val_shap = np.asarray(val_shap)

    val_mean_abs = np.mean(np.abs(val_shap), axis=0)
    val_ranking = [
        {
            "rank": int(r + 1),
            "feature": feature_names[idx],
            "plain_name": feature_mapping[feature_names[idx]]["plain_name"],
            "indicator": feature_mapping[feature_names[idx]]["indicator"],
            "period": feature_mapping[feature_names[idx]]["period"],
            "family": feature_mapping[feature_names[idx]]["family"],
            "mean_abs_shap": float(round(val_mean_abs[idx], 6)),
            "mean_abs_points": float(round(val_mean_abs[idx] * 100.0, 2)),
        }
        for r, idx in enumerate(np.argsort(-val_mean_abs))
    ]

    # 2. Load holdout test data
    from mei.features.splits import load_test_indices

    test_idx = load_test_indices(active_dir / "splits", allow_override=True)
    cohort_df = load_full_clean_cohort()
    test_df = cohort_df.iloc[test_idx].reset_index(drop=True)
    # Transform test features
    from mei.features.pipeline import build_feature_pipeline

    feat_pipe = build_feature_pipeline("agg")
    train_val_idx = json.load(
        open(active_dir / "splits" / "train_idx.json")
    ) + json.load(open(active_dir / "splits" / "val_idx.json"))
    feat_pipe.fit(cohort_df.iloc[train_val_idx].reset_index(drop=True))
    X_test_df = feat_pipe.transform(test_df)

    # Ensure column alignment
    for col in feature_names:
        if col not in X_test_df.columns:
            X_test_df[col] = 0.0
    X_test_sample = X_test_df[feature_names]
    if len(X_test_sample) > n_test_samples:
        X_test_sample = X_test_sample.sample(
            n=n_test_samples, random_state=42
        ).reset_index(drop=True)

    logger.info(
        "Computing SHAP values for test sample (%d rows)...", len(X_test_sample)
    )
    test_shap = explainer.shap_values(
        X_test_sample.values, nsamples=100, l1_reg="num_features(10)"
    )
    test_shap = np.asarray(test_shap)
    test_mean_abs = np.mean(np.abs(test_shap), axis=0)

    test_ranking = [
        {
            "rank": int(r + 1),
            "feature": feature_names[idx],
            "plain_name": feature_mapping[feature_names[idx]]["plain_name"],
            "mean_abs_shap": float(round(test_mean_abs[idx], 6)),
            "mean_abs_points": float(round(test_mean_abs[idx] * 100.0, 2)),
        }
        for r, idx in enumerate(np.argsort(-test_mean_abs))
    ]

    # 3. Grouped importance
    val_mean_abs_dict = {
        feature_names[i]: float(val_mean_abs[i]) for i in range(len(feature_names))
    }
    grouped_importance = compute_grouped_importance(val_mean_abs_dict, feature_mapping)

    # 4. Tree model interaction values and state features importance
    tree_interactions: Dict[str, Any] = {}
    rf_dir = active_dir / "models" / "random_forest_hybrid"
    if not (rf_dir / "pipeline.joblib").exists():
        rf_dir = active_dir / "models" / "random_forest"
    if (rf_dir / "pipeline.joblib").exists():
        try:
            tree_pipe = joblib.load(rf_dir / "pipeline.joblib")
            tree_model = tree_pipe.named_steps.get("clf", tree_pipe)
            tr_states = pd.read_parquet(
                active_dir / "features" / "train_agg_states.parquet"
            )
            tree_explainer = shap.TreeExplainer(tree_model)
            sample_sub = tr_states.iloc[:100].values
            tree_shap_sub = tree_explainer.shap_values(sample_sub)
            if isinstance(tree_shap_sub, list):
                tree_shap_sub = tree_shap_sub[1]

            # Compute state vs aggregates share on hybrid tree model
            state_cols = [
                c
                for c in tr_states.columns
                if c.startswith("hmm_")
                or c.startswith("dist_cluster_")
                or c.startswith("cluster_affinity_")
            ]
            agg_cols = [c for c in tr_states.columns if c not in state_cols]
            state_idx = [
                tr_states.columns.get_loc(c)
                for c in state_cols
                if c in tr_states.columns
            ]
            agg_idx = [
                tr_states.columns.get_loc(c) for c in agg_cols if c in tr_states.columns
            ]

            state_importance = float(
                np.sum(np.mean(np.abs(tree_shap_sub[:, state_idx]), axis=0))
            )
            agg_importance = float(
                np.sum(np.mean(np.abs(tree_shap_sub[:, agg_idx]), axis=0))
            )
            total_tree_imp = state_importance + agg_importance or 1.0

            # Compute interaction values for top 5 features
            top5_tree_idx = np.argsort(-np.mean(np.abs(tree_shap_sub), axis=0))[:5]
            interaction_matrix = tree_explainer.shap_interaction_values(sample_sub[:30])
            if isinstance(interaction_matrix, list):
                interaction_matrix = interaction_matrix[1]
            top5_interactions = []
            for i, idx1 in enumerate(top5_tree_idx):
                for j, idx2 in enumerate(top5_tree_idx):
                    if idx1 < idx2:
                        val_inter = float(
                            np.mean(np.abs(interaction_matrix[:, idx1, idx2]))
                        )
                        top5_interactions.append(
                            {
                                "feature_1": tr_states.columns[idx1],
                                "feature_2": tr_states.columns[idx2],
                                "mean_abs_interaction": round(val_inter, 6),
                            }
                        )

            tree_interactions = {
                "hybrid_model_id": "random_forest_hybrid",
                "state_features_importance_share_pct": round(
                    state_importance / total_tree_imp * 100.0, 2
                ),
                "aggregates_importance_share_pct": round(
                    agg_importance / total_tree_imp * 100.0, 2
                ),
                "top_interactions": sorted(
                    top5_interactions,
                    key=lambda x: x["mean_abs_interaction"],
                    reverse=True,
                ),
            }
        except Exception as e:
            logger.warning("Could not compute TreeExplainer interaction values: %s", e)

    return {
        "baseline_probability": baseline_p,
        "baseline_mei": round(baseline_p * 100.0, 1),
        "validation_ranking": val_ranking,
        "test_ranking": test_ranking,
        "grouped_importance": grouped_importance,
        "tree_hybrid_audit": tree_interactions,
        "val_shap_matrix": val_shap,
        "val_sample_df": val_sample,
        "test_shap_matrix": test_shap,
        "test_sample_df": X_test_sample,
    }


def compute_shap_consistency_across_folds(
    run_dir: Optional[Path] = None,
    n_samples_per_fold: int = 35,
) -> Dict[str, Any]:
    """Compute SHAP ranking consistency across all 15 CV folds."""
    active_dir = get_active_run_dir(run_dir)
    cv_folds = json.load(open(active_dir / "splits" / "cv_folds.json"))
    with open(active_dir / "champion" / "features.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    tr_agg = pd.read_parquet(active_dir / "features" / "train_agg.parquet")
    val_agg = pd.read_parquet(active_dir / "features" / "val_agg.parquet")
    X_dev = pd.concat([tr_agg, val_agg]).reset_index(drop=True)[feature_names].values

    cohort_df = load_full_clean_cohort()
    tr_idx = json.load(open(active_dir / "splits" / "train_idx.json"))
    v_idx = json.load(open(active_dir / "splits" / "val_idx.json"))
    y_dev = cohort_df.iloc[tr_idx + v_idx]["at_risk"].values.astype(int)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    fold_rankings = []
    fold_top10_sets = []

    logger.info("Computing SHAP consistency across 15 folds...")
    for f_idx, fold in enumerate(cv_folds):
        train_indices = fold["train_idx"]
        val_indices = fold["val_idx"]

        # Train fold model
        fold_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=21, weights="distance")),
            ]
        )
        fold_pipe.fit(X_dev[train_indices], y_dev[train_indices])

        # Subsample validation set for speed
        if len(val_indices) > n_samples_per_fold:
            np.random.seed(42 + f_idx)
            sub_val = np.random.choice(
                val_indices, size=n_samples_per_fold, replace=False
            )
        else:
            sub_val = val_indices

        # KernelExplainer on fold validation samples
        bg_fold = shap.kmeans(X_dev[train_indices], 15)
        fold_explainer = shap.KernelExplainer(
            lambda x: fold_pipe.predict_proba(x)[:, 1], bg_fold
        )
        shap_vals = fold_explainer.shap_values(
            X_dev[sub_val], nsamples=80, l1_reg="num_features(10)"
        )
        shap_vals = np.asarray(shap_vals)
        mean_abs = np.mean(np.abs(shap_vals), axis=0)

        # Feature ranks (1 to 100)
        ranks = np.empty_like(mean_abs)
        ranks[np.argsort(-mean_abs)] = np.arange(1, len(mean_abs) + 1)
        fold_rankings.append(ranks)

        top10 = set([feature_names[i] for i in np.argsort(-mean_abs)[:10]])
        fold_top10_sets.append(top10)

    # 1. Pairwise Spearman rank correlations
    n_folds = len(fold_rankings)
    spearman_matrix = np.ones((n_folds, n_folds))
    pairwise_spearman = []
    pairwise_jaccard = []

    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            r, _ = stats.spearmanr(fold_rankings[i], fold_rankings[j])
            spearman_matrix[i, j] = r
            spearman_matrix[j, i] = r
            pairwise_spearman.append(float(r))

            # Jaccard overlap of top-10
            set_i, set_j = fold_top10_sets[i], fold_top10_sets[j]
            jaccard = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
            pairwise_jaccard.append(float(jaccard))

    # 2. Kendall's W coefficient of concordance
    # ranks_matrix shape: (m_raters=15, n_items=100)
    ranks_matrix = np.array(fold_rankings)
    m, n = ranks_matrix.shape
    R_j = ranks_matrix.sum(axis=0)
    R_bar = m * (n + 1) / 2.0
    S = float(np.sum((R_j - R_bar) ** 2))
    kendall_w = float(12.0 * S / (m**2 * (n**3 - n)))

    return {
        "n_folds": n_folds,
        "mean_spearman": float(round(np.mean(pairwise_spearman), 4)),
        "std_spearman": float(round(np.std(pairwise_spearman), 4)),
        "min_spearman": float(round(np.min(pairwise_spearman), 4)),
        "max_spearman": float(round(np.max(pairwise_spearman), 4)),
        "kendall_w": float(round(kendall_w, 4)),
        "mean_top10_jaccard": float(round(np.mean(pairwise_jaccard), 4)),
        "min_top10_jaccard": float(round(np.min(pairwise_jaccard), 4)),
        "max_top10_jaccard": float(round(np.max(pairwise_jaccard), 4)),
        "spearman_matrix": spearman_matrix.tolist(),
        "pairwise_jaccard": pairwise_jaccard,
    }


def compute_cross_method_comparison(
    run_dir: Optional[Path] = None,
    shap_mean_abs: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compare SHAP feature rankings with Permutation Importance and Logistic Regression."""
    active_dir = get_active_run_dir(run_dir)
    champ_dir = active_dir / "champion"

    pipeline = joblib.load(champ_dir / "pipeline.joblib")
    with open(champ_dir / "features.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    val_agg = pd.read_parquet(active_dir / "features" / "val_agg.parquet")[
        feature_names
    ]
    train_agg = pd.read_parquet(active_dir / "features" / "train_agg.parquet")[
        feature_names
    ]

    cohort_df = load_full_clean_cohort()
    tr_idx = json.load(open(active_dir / "splits" / "train_idx.json"))
    v_idx = json.load(open(active_dir / "splits" / "val_idx.json"))
    y_train = cohort_df.iloc[tr_idx]["at_risk"].values.astype(int)
    y_val = cohort_df.iloc[v_idx]["at_risk"].values.astype(int)

    # 1. Permutation Importance
    logger.info("Computing Permutation Importance on validation set...")
    perm_res = permutation_importance(
        pipeline, val_agg.values, y_val, n_repeats=5, random_state=42, scoring="f1"
    )
    perm_means = perm_res.importances_mean

    # 2. Standardized Logistic Regression Coefficients
    logger.info(
        "Fitting Standardized Logistic Regression for coefficient comparison..."
    )
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(train_agg.values)
    lr = LogisticRegression(penalty="l2", C=1.0, max_iter=500, random_state=42)
    lr.fit(X_tr_scaled, y_train)
    lr_coefs = np.abs(lr.coef_[0])

    # 3. Compile rankings
    if shap_mean_abs is None:
        shap_vals = np.abs(lr_coefs)  # fallback proxy if not supplied
    else:
        shap_vals = np.array([shap_mean_abs.get(f, 0.0) for f in feature_names])

    shap_ranks = np.empty_like(shap_vals)
    shap_ranks[np.argsort(-shap_vals)] = np.arange(1, len(shap_vals) + 1)

    perm_ranks = np.empty_like(perm_means)
    perm_ranks[np.argsort(-perm_means)] = np.arange(1, len(perm_means) + 1)

    lr_ranks = np.empty_like(lr_coefs)
    lr_ranks[np.argsort(-lr_coefs)] = np.arange(1, len(lr_coefs) + 1)

    r_sp, _ = stats.spearmanr(shap_ranks, perm_ranks)
    r_sl, _ = stats.spearmanr(shap_ranks, lr_ranks)
    r_pl, _ = stats.spearmanr(perm_ranks, lr_ranks)

    ranks_table = [
        {
            "feature": feature_names[i],
            "shap_rank": int(shap_ranks[i]),
            "perm_rank": int(perm_ranks[i]),
            "lr_rank": int(lr_ranks[i]),
            "shap_val": float(round(shap_vals[i], 6)),
            "perm_imp": float(round(perm_means[i], 6)),
            "lr_coef": float(round(lr_coefs[i], 6)),
        }
        for i in range(len(feature_names))
    ]

    # 4. Qualitative local linear surrogate (LIME-style) comparison for 5 example students
    lime_examples = []
    np.random.seed(42)
    sample_indices = np.random.choice(len(val_agg), size=5, replace=False)
    for s_idx in sample_indices:
        x_stu = val_agg.iloc[s_idx].values
        # Local Gaussian perturbation around student
        n_pert = 150
        pert = np.random.normal(
            loc=x_stu,
            scale=0.2 * np.std(train_agg.values, axis=0) + 1e-5,
            size=(n_pert, len(x_stu)),
        )
        dists = np.linalg.norm(pert - x_stu, axis=1)
        weights = np.exp(-dists / (np.median(dists) + 1e-8))
        preds = pipeline.predict_proba(pert)[:, 1]

        local_ridge = Ridge(alpha=1.0)
        local_ridge.fit(pert, preds, sample_weight=weights)
        top3_lime_idx = np.argsort(-np.abs(local_ridge.coef_))[:3]
        lime_examples.append(
            {
                "student_row": int(s_idx),
                "top_lime_features": [feature_names[idx] for idx in top3_lime_idx],
            }
        )

    return {
        "correlations": {
            "shap_vs_permutation": float(round(r_sp, 4)),
            "shap_vs_logistic": float(round(r_sl, 4)),
            "permutation_vs_logistic": float(round(r_pl, 4)),
        },
        "ranks_table": sorted(ranks_table, key=lambda x: x["shap_rank"]),
        "lime_qualitative_comparisons": lime_examples,
    }


def compute_pattern_shap_breakdown(
    run_dir: Optional[Path] = None,
    val_sample_df: Optional[pd.DataFrame] = None,
    val_shap_matrix: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute mean SHAP per feature within each longitudinal engagement pattern."""
    active_dir = get_active_run_dir(run_dir)
    with open(active_dir / "champion" / "features.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    feature_mapping = get_feature_group_mapping(feature_names)

    # Classify trajectories of the validation sample
    val_states_path = active_dir / "features" / "val_agg_states.parquet"
    patterns: List[str] = []
    if val_states_path.exists():
        val_states_df = pd.read_parquet(val_states_path)
        pat_cols = [
            ("pattern_consistent", "Consistent"),
            ("pattern_declining", "Declining"),
            ("pattern_improving", "Improving"),
            ("pattern_sporadic", "Sporadic"),
        ]
        if val_sample_df is not None and len(val_sample_df) < len(val_states_df):
            val_states_df = val_states_df.iloc[: len(val_sample_df)]

        for _, row in val_states_df.iterrows():
            assigned = "Consistent"
            for col, pat_name in pat_cols:
                if col in row and row[col] == 1.0:
                    assigned = pat_name
                    break
            patterns.append(assigned)
    else:
        patterns = ["Consistent"] * (
            len(val_sample_df) if val_sample_df is not None else 100
        )

    pattern_results: Dict[str, Any] = {}
    unique_patterns = ["Consistent", "Declining", "Sporadic", "Improving"]

    for pat in unique_patterns:
        mask = np.array([p == pat for p in patterns])
        n_stu = int(np.sum(mask))
        warn_small = n_stu < 30

        if n_stu > 0:
            pat_shap = val_shap_matrix[mask]
            mean_shap = np.mean(pat_shap, axis=0)  # directional impact

            # Top risk drivers (highest positive mean SHAP)
            top_risk_idx = np.argsort(-mean_shap)[:5]
            drivers = [
                {
                    "feature": feature_names[idx],
                    "plain_name": feature_mapping[feature_names[idx]]["plain_name"],
                    "mean_shap": float(round(mean_shap[idx], 6)),
                    "mean_shap_points": float(round(mean_shap[idx] * 100.0, 2)),
                }
                for idx in top_risk_idx
            ]
        else:
            drivers = []

        pattern_results[pat] = {
            "pattern_name": pat,
            "n_students": n_stu,
            "warning_small_sample": warn_small,
            "drivers": drivers,
        }

    return pattern_results


def explain_dataframe(
    df: pd.DataFrame,
    run_dir: Optional[Path] = None,
    max_explain: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Score and explain an arbitrary student survey DataFrame.

    Returns:
        (scored_df_with_reasons, list_of_full_explanation_dicts)
    """
    from mei.registry.score import score_dataframe

    active_dir = get_active_run_dir(run_dir)
    scored_df = score_dataframe(df, run_dir=active_dir)

    explainer, pipeline, calibrator, feature_names, baseline_p = get_champion_explainer(
        active_dir, background_samples=25
    )
    feature_mapping = get_feature_group_mapping(feature_names)
    templates = load_reason_templates()

    # Engineer features from input DataFrame
    from mei.features.pipeline import extract_features

    X_transformed, _ = extract_features(df, feature_set="agg")

    # Ensure full feature alignment
    for col in feature_names:
        if col not in X_transformed.columns:
            X_transformed[col] = 0.0
    X_aligned = X_transformed[feature_names]

    n_rows = len(X_aligned) if max_explain is None else min(len(X_aligned), max_explain)
    X_to_explain = X_aligned.iloc[:n_rows].values

    logger.info("Computing SHAP explanations for %d students...", n_rows)
    shap_vals = explainer.shap_values(
        X_to_explain, nsamples=100, l1_reg="num_features(10)"
    )
    shap_vals = np.asarray(shap_vals)

    full_explanations: List[Dict[str, Any]] = []
    top_reasons_col: List[List[str]] = []

    for i in range(n_rows):
        stu_id = str(scored_df.iloc[i].get("student_id", f"STU_{i:04d}"))
        mei = float(scored_df.iloc[i]["mei"])
        base_mei = round(baseline_p * 100.0, 1)

        reasons = generate_student_reasons(
            feature_values=X_to_explain[i],
            shap_values=shap_vals[i],
            feature_names=feature_names,
            top_n_risk=3,
            top_n_protective=2,
            mei_score=mei,
            baseline_mei=base_mei,
            templates=templates,
        )

        reason_texts = [f"{r['text']} ({r['points']:+d} pts)" for r in reasons]
        top_reasons_col.append(reason_texts)

        # Compact waterfall representation (top 5 features + other)
        pts_arr = convert_shap_to_mei_points(
            shap_vals[i], mei_score=mei, baseline_mei=base_mei
        )
        top_contrib_idx = np.argsort(-np.abs(pts_arr))[:5]
        waterfall_steps = [
            {
                "feature": feature_names[idx],
                "plain_name": feature_mapping[feature_names[idx]]["plain_name"],
                "points": int(pts_arr[idx]),
            }
            for idx in top_contrib_idx
        ]

        full_explanations.append(
            {
                "student_id": stu_id,
                "mei": mei,
                "baseline_mei": base_mei,
                "band": scored_df.iloc[i]["band"],
                "flag": bool(scored_df.iloc[i]["flag"]),
                "pattern": scored_df.iloc[i]["pattern"],
                "reasons": reasons,
                "waterfall_steps": waterfall_steps,
            }
        )

    # Update reasons column in scored_df
    if len(top_reasons_col) == len(scored_df):
        scored_df["top_reasons"] = top_reasons_col
    else:
        # Partial explanation
        padded = top_reasons_col + [
            [] for _ in range(len(scored_df) - len(top_reasons_col))
        ]
        scored_df["top_reasons"] = padded

    return scored_df, full_explanations
