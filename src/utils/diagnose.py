"""Comprehensive Diagnostic Engine for Model Performance & Data Signal Analysis.

Performs:
1. Ceiling test: Bayes-optimal ROC-AUC, PR-AUC, Brier score, and R² using true probabilities.
2. Signal test: Univariate AUC, Spearman rho, and Mutual Information per feature.
3. Pipeline sanity checks: Row alignment, target encoding, predict_proba slice, constant features,
   split prevalence, label shuffling test, and oracle feature test.
4. Baseline comparison: Default LogisticRegression & LightGBM on simple stratified splits
   with various feature subsets (allowed only and derived exposures).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.pipeline import (
    ALL_ALLOWED_FEATURES,
    NOMINAL_FEATURES,
    get_scaled_pipeline,
    get_tree_pipeline,
    validate_feature_boundary,
)


def run_diagnostics():
    print("=" * 80)
    print("COMPREHENSIVE MODEL PERFORMANCE & DATA SIGNAL DIAGNOSIS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 0. Load Data
    # -------------------------------------------------------------------------
    clean_truth_path = (
        PROJECT_ROOT / "data" / "synthetic" / "eri_synthetic_clean_truth_latest.csv"
        if (
            PROJECT_ROOT / "data" / "synthetic" / "eri_synthetic_clean_truth_latest.csv"
        ).is_file()
        else PROJECT_ROOT / "data" / "synthetic" / "eri_synthetic_clean_truth.csv"
    )
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    val_path = PROJECT_ROOT / "data" / "processed" / "val.csv"
    split_ids_path = PROJECT_ROOT / "data" / "processed" / "split_ids.json"
    raw_path = (
        PROJECT_ROOT / "data" / "synthetic" / "eri_synthetic_raw_latest.csv"
        if (
            PROJECT_ROOT / "data" / "synthetic" / "eri_synthetic_raw_latest.csv"
        ).is_file()
        else PROJECT_ROOT / "data" / "synthetic" / "eri_synthetic_raw.csv"
    )
    interim_clean_path = PROJECT_ROOT / "data" / "interim" / "clean.csv"

    df_clean_truth = pd.read_csv(clean_truth_path)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    with open(split_ids_path) as f:
        split_ids = json.load(f)

    # -------------------------------------------------------------------------
    # 1. CEILING TEST (Bayes-Optimal Performance)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("1. CEILING TEST (Bayes-Optimal Benchmark)")
    print("-" * 40)

    # The synthetic generator created: latent_risk_score = round(100.0 * true_prob, 2)
    # true_prob = latent_risk_score / 100.0
    # Let's verify if exact true_prob can also be computed or from latent_risk_score
    df_clean_truth["true_prob"] = df_clean_truth["latent_risk_score"] / 100.0
    train_df["true_prob"] = train_df["latent_risk_score"] / 100.0
    val_df["true_prob"] = val_df["latent_risk_score"] / 100.0

    # Evaluate on Train and Validation (DO NOT USE TEST)
    y_train = train_df["attrition"].values
    prob_train = train_df["true_prob"].values
    y_val = val_df["attrition"].values
    prob_val = val_df["true_prob"].values

    # Full population (2000 students)
    y_all = df_clean_truth["attrition"].values
    prob_all = df_clean_truth["true_prob"].values

    ceiling_metrics = {
        "full_population": {
            "n_samples": len(y_all),
            "prevalence": float(np.mean(y_all)),
            "bayes_roc_auc": float(roc_auc_score(y_all, prob_all)),
            "bayes_pr_auc": float(average_precision_score(y_all, prob_all)),
            "bayes_brier": float(brier_score_loss(y_all, prob_all)),
        },
        "train_set": {
            "n_samples": len(y_train),
            "prevalence": float(np.mean(y_train)),
            "bayes_roc_auc": float(roc_auc_score(y_train, prob_train)),
            "bayes_pr_auc": float(average_precision_score(y_train, prob_train)),
            "bayes_brier": float(brier_score_loss(y_train, prob_train)),
        },
        "val_set": {
            "n_samples": len(y_val),
            "prevalence": float(np.mean(y_val)),
            "bayes_roc_auc": float(roc_auc_score(y_val, prob_val)),
            "bayes_pr_auc": float(average_precision_score(y_val, prob_val)),
            "bayes_brier": float(brier_score_loss(y_val, prob_val)),
        },
        "regression_track_ceiling": {
            "description": "If latent_risk_score is target, true_prob * 100 is exact target (R2=1.000)",
            "train_r2_oracle": float(
                r2_score(train_df["latent_risk_score"], prob_train * 100.0)
            ),
            "val_r2_oracle": float(
                r2_score(val_df["latent_risk_score"], prob_val * 100.0)
            ),
        },
    }

    print(f"Full Population (N={len(y_all)}) Bayes Ceiling:")
    print(
        f"  Bayes-Optimal ROC-AUC : {ceiling_metrics['full_population']['bayes_roc_auc']:.4f}"
    )
    print(
        f"  Bayes-Optimal PR-AUC  : {ceiling_metrics['full_population']['bayes_pr_auc']:.4f} (Base prevalence: {ceiling_metrics['full_population']['prevalence']:.4f})"
    )
    print(
        f"  Bayes-Optimal Brier   : {ceiling_metrics['full_population']['bayes_brier']:.4f}"
    )

    print(f"\nValidation Set (N={len(y_val)}) Bayes Ceiling:")
    print(
        f"  Bayes-Optimal ROC-AUC : {ceiling_metrics['val_set']['bayes_roc_auc']:.4f}"
    )
    print(
        f"  Bayes-Optimal PR-AUC  : {ceiling_metrics['val_set']['bayes_pr_auc']:.4f} (Base prevalence: {ceiling_metrics['val_set']['prevalence']:.4f})"
    )
    print(f"  Bayes-Optimal Brier   : {ceiling_metrics['val_set']['bayes_brier']:.4f}")

    # Check generator observation noise
    with open(PROJECT_ROOT / "data" / "synthetic" / "ground_truth.json") as f:
        gt_params = json.load(f)
    obs_noise = gt_params.get("structural_coefficients", {}).get(
        "observation_noise_std", "Unknown"
    )
    ceiling_metrics["generator_observation_noise_std"] = obs_noise
    print(f"  Generator Observation Noise std: {obs_noise}")

    # Save to artifacts/diagnostics/ceiling.json
    diag_dir = PROJECT_ROOT / "artifacts" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    with open(diag_dir / "ceiling.json", "w") as f:
        json.dump(ceiling_metrics, f, indent=2)
    print(f"-> Saved ceiling metrics to: {diag_dir / 'ceiling.json'}")

    # -------------------------------------------------------------------------
    # 2. SIGNAL TEST (Univariate AUC, Spearman rho, Mutual Information)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("2. SIGNAL TEST (Feature-Level Predictiveness on Training Set)")
    print("-" * 40)

    # Candidate features to test (all allowed features plus derived exposures).
    eval_df = train_df.copy()

    # Encode ordinals and nominals numerically for correlation and MI
    ord_maps = {
        "household_income_bracket": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
        "financial_stress": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        "internet_reliability": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        "internet_affordability": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        "device_quality": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        "year_of_study": {1: 1, 2: 2, 3: 3, 4: 4},
        "baseline_gpa_band": {
            "Band 1 (< 2.00)": 1,
            "Band 2 (2.00 - 2.49)": 2,
            "Band 3 (2.50 - 2.99)": 3,
            "Band 4 (3.00 - 3.49)": 4,
            "Band 5 (3.50 - 4.00)": 5,
        },
    }
    for col, mapping in ord_maps.items():
        if col in eval_df.columns:
            eval_df[col] = eval_df[col].map(mapping).fillna(eval_df[col])

    # Convert nominals to categorical codes
    for col in NOMINAL_FEATURES:
        if col in eval_df.columns:
            eval_df[col] = eval_df[col].astype("category").cat.codes

    all_features_to_screen = [
        col
        for col in eval_df.columns
        if col
        not in [
            "participant_id",
            "gender",
            "attrition",
            "intention_to_withdraw",
            "latent_risk_score",
            "true_prob",
        ]
    ]

    signal_results = []
    y_tr = eval_df["attrition"].values

    for feat in all_features_to_screen:
        vals = eval_df[feat].values
        # Handle any NaNs
        valid_mask = ~pd.isna(vals)
        v_clean = vals[valid_mask]
        y_clean = y_tr[valid_mask]

        if len(np.unique(v_clean)) <= 1:
            auc = 0.5
            s_rho = 0.0
            mi = 0.0
        else:
            try:
                # Raw AUC
                raw_auc = roc_auc_score(y_clean, v_clean)
                # Directional adjusted AUC (discrimination capability)
                auc = max(raw_auc, 1.0 - raw_auc)
            except (ValueError, TypeError):
                auc = 0.5

            try:
                s_rho, _ = spearmanr(v_clean, y_clean)
            except (ValueError, TypeError):
                s_rho = 0.0

            try:
                mi = float(
                    mutual_info_classif(
                        v_clean.reshape(-1, 1), y_clean, random_state=42
                    )[0]
                )
            except (ValueError, TypeError):
                mi = 0.0

        is_allowed = feat in ALL_ALLOWED_FEATURES
        is_derived_exposure = feat in [
            "financial_hardship",
            "long_commute",
            "inadequate_digital_access",
            "heavy_part_time_work",
        ]

        signal_results.append(
            {
                "feature": feat,
                "auc": round(float(auc), 4),
                "spearman_rho": round(float(s_rho), 4),
                "abs_spearman": round(abs(float(s_rho)), 4),
                "mutual_info": round(float(mi), 4),
                "category": (
                    "Allowed Feature"
                    if is_allowed
                    else (
                        "Derived Exposure (Excluded)"
                        if is_derived_exposure
                        else "Other"
                    )
                ),
            }
        )

    signal_df = pd.DataFrame(signal_results)

    print("\nTop 10 Features by Univariate Discriminative AUC:")
    top_auc = signal_df.sort_values(by="auc", ascending=False).head(10)
    for idx, row in top_auc.iterrows():
        print(
            f"  {row['feature']:<28} | AUC: {row['auc']:.4f} | Rho: {row['spearman_rho']:>7.4f} | MI: {row['mutual_info']:.4f} | {row['category']}"
        )

    print("\nTop 10 Features by Absolute Spearman Correlation:")
    top_rho = signal_df.sort_values(by="abs_spearman", ascending=False).head(10)
    for idx, row in top_rho.iterrows():
        print(
            f"  {row['feature']:<28} | Rho: {row['spearman_rho']:>7.4f} | AUC: {row['auc']:.4f} | MI: {row['mutual_info']:.4f} | {row['category']}"
        )

    print("\nTop 10 Features by Mutual Information:")
    top_mi = signal_df.sort_values(by="mutual_info", ascending=False).head(10)
    for idx, row in top_mi.iterrows():
        print(
            f"  {row['feature']:<28} | MI: {row['mutual_info']:.4f} | AUC: {row['auc']:.4f} | Rho: {row['spearman_rho']:>7.4f} | {row['category']}"
        )

    # -------------------------------------------------------------------------
    # 3. PIPELINE SANITY CHECKS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("3. PIPELINE SANITY & LEAKAGE CHECKS")
    print("-" * 40)

    # 3.1 ID Alignment across cleaning & splits
    print("3.1 Checking row and ID alignment across pipeline stages:")
    raw_df = pd.read_csv(raw_path)
    clean_df = pd.read_csv(interim_clean_path)
    print(f"  Raw rows: {len(raw_df)} -> Clean interim rows: {len(clean_df)}")

    # Check if participant_ids match indices in split_ids
    train_ids_from_split = set(
        clean_df.iloc[split_ids["train_indices"]]["participant_id"]
    )
    train_ids_from_file = set(train_df["participant_id"])
    val_ids_from_split = set(clean_df.iloc[split_ids["val_indices"]]["participant_id"])
    val_ids_from_file = set(val_df["participant_id"])

    train_id_match = train_ids_from_split == train_ids_from_file
    val_id_match = val_ids_from_split == val_ids_from_file
    print(
        f"  Train set ID match between split_ids.json and train.csv: {train_id_match} ({len(train_ids_from_file)} IDs)"
    )
    print(
        f"  Val set ID match between split_ids.json and val.csv: {val_id_match} ({len(val_ids_from_file)} IDs)"
    )

    # Check for duplicate IDs or overlap between splits
    overlap_train_val = len(train_ids_from_file.intersection(val_ids_from_file))
    print(f"  Train/Val ID overlap (must be 0): {overlap_train_val}")

    # 3.2 Target Encoding & Positive Counts
    print("\n3.2 Target variable encoding:")
    train_unique = train_df["attrition"].value_counts().to_dict()
    val_unique = val_df["attrition"].value_counts().to_dict()
    print(
        f"  Train target distribution: {train_unique} (1 = Attrition: {train_unique.get(1, 0)}, {train_df['attrition'].mean()*100:.2f}%)"
    )
    print(
        f"  Val target distribution:   {val_unique} (1 = Attrition: {val_unique.get(1, 0)}, {val_df['attrition'].mean()*100:.2f}%)"
    )

    # 3.3 Constant or dropped features after Preprocessing
    print("\n3.3 Preprocessing pipeline transform sanity:")
    feature_tr_df = train_df.drop(columns=["true_prob"], errors="ignore")
    feature_val_df = val_df.drop(columns=["true_prob"], errors="ignore")
    feature_tr = validate_feature_boundary(feature_tr_df)
    feature_val = validate_feature_boundary(feature_val_df)
    prep_scaled = get_scaled_pipeline()
    prep_scaled.fit(feature_tr, is_train_fold=True)
    X_tr_scaled = prep_scaled.transform(feature_tr)
    X_val_scaled = prep_scaled.transform(feature_val)

    prep_tree = get_tree_pipeline()
    prep_tree.fit(feature_tr, is_train_fold=True)
    X_tr_tree = prep_tree.transform(feature_tr)
    X_val_tree = prep_tree.transform(feature_val)

    print(
        f"  Scaled pipeline input shape: {feature_tr.shape} -> output shape: {X_tr_scaled.shape}"
    )
    print(
        f"  Tree pipeline input shape:   {feature_tr.shape} -> output shape: {X_tr_tree.shape}"
    )
    print(
        f"  Any NaN in X_tr_scaled: {np.isnan(X_tr_scaled).any()}, X_val_scaled: {np.isnan(X_val_scaled).any()}"
    )
    print(
        f"  Any NaN in X_tr_tree:   {np.isnan(X_tr_tree).any()}, X_val_tree:   {np.isnan(X_val_tree).any()}"
    )

    # Check for zero-variance / constant features
    tr_var = np.var(X_tr_scaled, axis=0)
    zero_var_count = np.sum(tr_var == 0)
    print(
        f"  Constant (zero-variance) columns in preprocessed scaled X: {zero_var_count}"
    )

    # 3.4 Metric Calculation Slice Verification
    print("\n3.4 Metric calculation slice check:")
    lr_dummy = LogisticRegression(max_iter=500).fit(X_tr_scaled, y_train)
    probs_dummy = lr_dummy.predict_proba(X_val_scaled)
    # Check slice [:, 1] vs [:, 0]
    auc_col1 = roc_auc_score(y_val, probs_dummy[:, 1])
    auc_col0 = roc_auc_score(y_val, probs_dummy[:, 0])
    print(f"  Using probs[:, 1] ROC-AUC: {auc_col1:.4f}")
    print(f"  Using probs[:, 0] ROC-AUC: {auc_col0:.4f}")
    assert (
        auc_col1 > auc_col0
    ), "Metric calculation must strictly use column 1 for positive class probability!"

    # 3.5 Shuffled-Label Sanity Check (Should yield AUC ~ 0.50)
    print("\n3.5 Shuffled-Label Sanity Check (Null Hypothesis Control):")
    rng = np.random.default_rng(123)
    y_train_shuffled = rng.permutation(y_train)
    lr_null = LogisticRegression(max_iter=500).fit(X_tr_scaled, y_train_shuffled)
    null_auc = roc_auc_score(y_val, lr_null.predict_proba(X_val_scaled)[:, 1])
    null_pr = average_precision_score(y_val, lr_null.predict_proba(X_val_scaled)[:, 1])
    print(f"  Shuffled-Label Model Val ROC-AUC: {null_auc:.4f} (Expected: ~0.50)")
    print(
        f"  Shuffled-Label Model Val PR-AUC:  {null_pr:.4f} (Expected: ~{np.mean(y_val):.4f})"
    )

    # 3.6 Synthetic Oracle Feature Test (Model given true_prob must achieve ~ceiling)
    print("\n3.6 Synthetic Oracle Feature Test:")
    # If we add true_prob as a feature into LogisticRegression, what happens?
    X_tr_oracle = prob_train.reshape(-1, 1)
    X_val_oracle = prob_val.reshape(-1, 1)
    lr_oracle = LogisticRegression().fit(X_tr_oracle, y_train)
    oracle_val_auc = roc_auc_score(y_val, lr_oracle.predict_proba(X_val_oracle)[:, 1])
    oracle_val_pr = average_precision_score(
        y_val, lr_oracle.predict_proba(X_val_oracle)[:, 1]
    )
    oracle_val_brier = brier_score_loss(
        y_val, lr_oracle.predict_proba(X_val_oracle)[:, 1]
    )
    print(
        f"  Oracle Model Val ROC-AUC: {oracle_val_auc:.4f} (Bayes Ceiling: {ceiling_metrics['val_set']['bayes_roc_auc']:.4f})"
    )
    print(
        f"  Oracle Model Val PR-AUC:  {oracle_val_pr:.4f} (Bayes Ceiling: {ceiling_metrics['val_set']['bayes_pr_auc']:.4f})"
    )
    print(
        f"  Oracle Model Val Brier:   {oracle_val_brier:.4f} (Bayes Ceiling: {ceiling_metrics['val_set']['bayes_brier']:.4f})"
    )

    # -------------------------------------------------------------------------
    # 4. BASELINES & ABLATION STUDY
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("4. BASELINE EXPERIMENTS & FEATURE SET ABLATION")
    print("-" * 40)

    # Experiment A: Default models on ALLOWED features (Pipeline view)
    lr_allowed = LogisticRegression(max_iter=500, random_state=42).fit(
        X_tr_scaled, y_train
    )
    lgb_allowed = LGBMClassifier(random_state=42, verbose=-1).fit(X_tr_tree, y_train)

    val_probs_lr_a = lr_allowed.predict_proba(X_val_scaled)[:, 1]
    val_probs_lgb_a = lgb_allowed.predict_proba(X_val_tree)[:, 1]

    print("Experiment A: Standard Allowed Feature Set (Pipeline)")
    print(
        f"  Logistic Regression : Val ROC-AUC = {roc_auc_score(y_val, val_probs_lr_a):.4f} | PR-AUC = {average_precision_score(y_val, val_probs_lr_a):.4f}"
    )
    print(
        f"  LightGBM            : Val ROC-AUC = {roc_auc_score(y_val, val_probs_lgb_a):.4f} | PR-AUC = {average_precision_score(y_val, val_probs_lgb_a):.4f}"
    )

    # Experiment B: Add DERIVED EXPOSURES (financial_hardship, long_commute, etc.)
    # In generate.py, latent_logit directly had coefficients on these 4 exposures!
    exp_cols = [
        "financial_hardship",
        "long_commute",
        "inadequate_digital_access",
        "heavy_part_time_work",
    ]
    exp_tr = train_df[exp_cols].fillna(0).values
    exp_val = val_df[exp_cols].fillna(0).values
    X_tr_b = np.column_stack([X_tr_tree, exp_tr])
    X_val_b = np.column_stack([X_val_tree, exp_val])

    lr_b = LogisticRegression(max_iter=500, random_state=42).fit(X_tr_b, y_train)
    lgb_b = LGBMClassifier(random_state=42, verbose=-1).fit(X_tr_b, y_train)
    val_probs_lr_b = lr_b.predict_proba(X_val_b)[:, 1]
    val_probs_lgb_b = lgb_b.predict_proba(X_val_b)[:, 1]

    print("\nExperiment B: Allowed Features + 4 Direct Policy Exposures")
    print(
        f"  Logistic Regression : Val ROC-AUC = {roc_auc_score(y_val, val_probs_lr_b):.4f} | PR-AUC = {average_precision_score(y_val, val_probs_lr_b):.4f}"
    )
    print(
        f"  LightGBM            : Val ROC-AUC = {roc_auc_score(y_val, val_probs_lgb_b):.4f} | PR-AUC = {average_precision_score(y_val, val_probs_lgb_b):.4f}"
    )

    # Experiment C: Cross-Validation check on allowed features.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_lr_rocs, cv_lgb_rocs = [], []
    cv_lr_prs, cv_lgb_prs = [], []

    for fold, (trn_idx, hld_idx) in enumerate(skf.split(X_tr_scaled, y_train)):
        m_lr = LogisticRegression(max_iter=500, random_state=42).fit(
            X_tr_scaled[trn_idx], y_train[trn_idx]
        )
        m_lgb = LGBMClassifier(random_state=42, verbose=-1).fit(
            X_tr_tree[trn_idx], y_train[trn_idx]
        )

        p_lr = m_lr.predict_proba(X_tr_scaled[hld_idx])[:, 1]
        p_lgb = m_lgb.predict_proba(X_tr_tree[hld_idx])[:, 1]

        cv_lr_rocs.append(roc_auc_score(y_train[hld_idx], p_lr))
        cv_lgb_rocs.append(roc_auc_score(y_train[hld_idx], p_lgb))
        cv_lr_prs.append(average_precision_score(y_train[hld_idx], p_lr))
        cv_lgb_prs.append(average_precision_score(y_train[hld_idx], p_lgb))

    print(
        "\nExperiment D: 5-Fold Stratified Cross-Validation on Training Partition (Allowed Features Only)"
    )
    print(
        f"  Logistic Regression : Mean CV ROC-AUC = {np.mean(cv_lr_rocs):.4f} (+/- {np.std(cv_lr_rocs):.4f}) | Mean PR-AUC = {np.mean(cv_lr_prs):.4f}"
    )
    print(
        f"  LightGBM            : Mean CV ROC-AUC = {np.mean(cv_lgb_rocs):.4f} (+/- {np.std(cv_lgb_rocs):.4f}) | Mean PR-AUC = {np.mean(cv_lgb_prs):.4f}"
    )

    print("\n" + "=" * 80)
    print("DIAGNOSTIC RUN COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnostics()
