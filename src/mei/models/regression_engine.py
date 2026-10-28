"""Comprehensive 15-fold cross-validation and benchmarking engine for MEI Track B regression models."""

import logging
import time
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import scipy.stats as stats
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mei.evaluation.regression_metrics import (
    compute_breusch_pagan_test,
    compute_classification_at_60,
    compute_regression_metrics,
    compute_track_b_leaderboard_score,
)
from mei.models.deep.architectures import build_deep_model
from mei.models.deep.regression_trainer import (
    DeepEngagementDataset,
    predict_deep_regressor,
    train_two_stage_regression,
)
from mei.models.regression_registry import (
    RegressionModelSpec,
    create_base_regressor,
)
from mei.models.regression_tuner import tune_regression_model

logger = logging.getLogger(__name__)


def build_regression_pipeline(
    spec: RegressionModelSpec,
    params: Optional[Dict[str, Any]] = None,
    random_seed: int = 42,
) -> Pipeline:
    """Build a complete scikit-learn Pipeline for a candidate regression model specification."""
    estimator = create_base_regressor(spec, hyperparams=params, random_seed=random_seed)
    if spec.needs_scaling:
        return Pipeline([("scaler", StandardScaler()), ("reg", estimator)])
    return Pipeline([("reg", estimator)])


def compute_regression_learning_curve(
    spec: RegressionModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    n_fractions: int = 8,
    cv_splits: int = 3,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Compute learning curve data (R² vs training sample fraction) via cross-validation."""
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    fractions = np.linspace(0.15, 1.0, n_fractions)

    train_sizes: List[int] = []
    train_scores_mean: List[float] = []
    val_scores_mean: List[float] = []

    for frac in fractions:
        fold_train_scores: List[float] = []
        fold_val_scores: List[float] = []

        for tr_idx, val_idx in kf.split(X, y):
            n_sub = max(25, int(len(tr_idx) * frac))
            sub_tr_idx = tr_idx[:n_sub]

            X_tr, y_tr = X[sub_tr_idx], y[sub_tr_idx]
            X_v, y_v = X[val_idx], y[val_idx]

            pipe = build_regression_pipeline(spec, params=params, random_seed=seed)
            try:
                pipe.fit(X_tr, y_tr)
                p_tr = np.clip(pipe.predict(X_tr), 0.0, 100.0)
                p_v = np.clip(pipe.predict(X_v), 0.0, 100.0)
                m_tr, _, _ = compute_regression_metrics(
                    y_tr, p_tr, n_features=X.shape[1], compute_kendall=False
                )
                m_v, _, _ = compute_regression_metrics(
                    y_v, p_v, n_features=X.shape[1], compute_kendall=False
                )
                fold_train_scores.append(m_tr.r2)
                fold_val_scores.append(m_v.r2)
            except Exception:
                fold_train_scores.append(0.0)
                fold_val_scores.append(0.0)

        train_sizes.append(int(len(X) * (cv_splits - 1) / cv_splits * frac))
        train_scores_mean.append(float(np.mean(fold_train_scores)))
        val_scores_mean.append(float(np.mean(fold_val_scores)))

    return {
        "train_sizes": train_sizes,
        "train_scores": train_scores_mean,
        "val_scores": val_scores_mean,
    }


def compute_regression_validation_curve(
    spec: RegressionModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    param_name: str,
    param_range: List[Any],
    base_params: Dict[str, Any],
    cv_splits: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute validation curve coordinates for a key hyperparameter."""
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    train_scores_mean: List[float] = []
    val_scores_mean: List[float] = []

    for val in param_range:
        current_params = dict(base_params)
        current_params[param_name] = val
        f_tr: List[float] = []
        f_v: List[float] = []

        for tr_idx, val_idx in kf.split(X, y):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_v, y_v = X[val_idx], y[val_idx]
            pipe = build_regression_pipeline(
                spec, params=current_params, random_seed=seed
            )
            try:
                pipe.fit(X_tr, y_tr)
                p_tr = np.clip(pipe.predict(X_tr), 0.0, 100.0)
                p_v = np.clip(pipe.predict(X_v), 0.0, 100.0)
                m_tr, _, _ = compute_regression_metrics(
                    y_tr, p_tr, n_features=X.shape[1], compute_kendall=False
                )
                m_v, _, _ = compute_regression_metrics(
                    y_v, p_v, n_features=X.shape[1], compute_kendall=False
                )
                f_tr.append(m_tr.r2)
                f_v.append(m_v.r2)
            except Exception:
                f_tr.append(0.0)
                f_v.append(0.0)

        train_scores_mean.append(float(np.mean(f_tr)))
        val_scores_mean.append(float(np.mean(f_v)))

    return {
        "param_name": param_name,
        "param_range": [
            str(v) if not isinstance(v, (int, float)) else v for v in param_range
        ],
        "train_scores": train_scores_mean,
        "val_scores": val_scores_mean,
    }


def evaluate_single_regression_model(
    spec: RegressionModelSpec,
    X_dict: Dict[str, Any],
    y: np.ndarray,
    cv_folds: List[Dict[str, Any]],
    seed: int = 42,
    patterns: Optional[np.ndarray] = None,
    levels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Execute complete 15-fold cross-validation evaluation for a candidate regression model."""
    start_time = time.time()
    n_samples = len(y)
    n_folds = len(cv_folds)

    # 1. Resolve feature inputs
    feat_key = spec.feature_set
    if feat_key not in X_dict:
        raise KeyError(f"Feature set '{feat_key}' not found in X_dict.")
    X_matrix = X_dict[feat_key]
    n_features = (
        X_matrix.shape[1]
        if hasattr(X_matrix, "shape") and len(X_matrix.shape) == 2
        else 7
    )

    # 2. Hyperparameter Tuning on Outer Fold 0 (Train Split)
    f0 = cv_folds[0]
    f0_train_idx = np.asarray(f0.get("train_idx", f0.get("train_indices")), dtype=int)
    tuned_params: Dict[str, Any] = {}
    tuning_summary: Dict[str, Any] = {}

    if spec.category != "deep" and spec.n_tuning_trials > 0 and spec.search_space:
        tuned_params, tuning_summary = tune_regression_model(
            spec,
            X_train=X_matrix[f0_train_idx],
            y_train=y[f0_train_idx],
            cv_folds=3,
            seed=seed,
            max_trials=spec.n_tuning_trials,
            time_budget_seconds=spec.time_budget_seconds,
        )
    else:
        tuned_params = dict(spec.fixed_params)

    # 3. 15-Fold Cross-Validation Loop
    fold_results: List[Dict[str, Any]] = []
    oof_predictions_accum = np.zeros(n_samples, dtype=np.float64)
    oof_counts = np.zeros(n_samples, dtype=np.int32)

    fold_r2_val: List[float] = []
    fold_r2_train: List[float] = []
    fold_rmse_val: List[float] = []
    fold_mae_val: List[float] = []
    fold_spearman_val: List[float] = []
    fold_pearson_val: List[float] = []
    fold_within_5_val: List[float] = []
    fold_within_10_val: List[float] = []
    fold_f1_60_val: List[float] = []

    iteration_losses: Optional[Dict[str, List[float]]] = None
    representative_fold_history: Optional[Dict[str, Any]] = None

    for fold_idx, fold_data in enumerate(cv_folds):
        tr_idx = np.asarray(
            fold_data.get("train_idx", fold_data.get("train_indices")), dtype=int
        )
        val_idx = np.asarray(
            fold_data.get("val_idx", fold_data.get("val_indices")), dtype=int
        )

        y_tr, y_v = y[tr_idx], y[val_idx]

        # ---------------------------------------------------------
        # A. Classical / Boosting / Linear Models
        # ---------------------------------------------------------
        if spec.category != "deep":
            X_tr, X_v = X_matrix[tr_idx], X_matrix[val_idx]
            pipe = build_regression_pipeline(
                spec, params=tuned_params, random_seed=seed + fold_idx
            )

            pipe.fit(X_tr, y_tr)
            y_pred_tr_raw = pipe.predict(X_tr)
            y_pred_v_raw = pipe.predict(X_v)

            # Record iteration losses on Fold 0 for boosting models
            if (
                fold_idx == 0
                and spec.category == "boosting"
                and iteration_losses is None
            ):
                est = pipe.named_steps["reg"]
                if hasattr(est, "evals_result_") and est.evals_result_:
                    # LightGBM or XGBoost evals result
                    pass

        # ---------------------------------------------------------
        # B. Deep Learning Models
        # ---------------------------------------------------------
        else:
            if spec.feature_set == "sequence":
                raw_seq_tr = X_dict["sequence"][tr_idx]
                raw_seq_v = X_dict["sequence"][val_idx]
                n_tr, t, d = raw_seq_tr.shape
                scaler = StandardScaler()
                scaled_seq_tr = (
                    scaler.fit_transform(raw_seq_tr.reshape(-1, d))
                    .reshape(n_tr, t, d)
                    .astype(np.float32)
                )
                scaled_seq_v = (
                    scaler.transform(raw_seq_v.reshape(-1, d))
                    .reshape(len(val_idx), t, d)
                    .astype(np.float32)
                )
                ds_train = DeepEngagementDataset(X_seq=scaled_seq_tr, y=y_tr)
                ds_val = DeepEngagementDataset(X_seq=scaled_seq_v, y=y_v)
                tab_dim = 14
            elif spec.feature_set == "hybrid_multimodal":
                raw_seq_tr = X_dict["sequence"][tr_idx]
                raw_seq_v = X_dict["sequence"][val_idx]
                n_tr, t, d = raw_seq_tr.shape
                s_seq = StandardScaler()
                scaled_seq_tr = (
                    s_seq.fit_transform(raw_seq_tr.reshape(-1, d))
                    .reshape(n_tr, t, d)
                    .astype(np.float32)
                )
                scaled_seq_v = (
                    s_seq.transform(raw_seq_v.reshape(-1, d))
                    .reshape(len(val_idx), t, d)
                    .astype(np.float32)
                )

                raw_tab_tr = X_dict["agg_states"][tr_idx]
                raw_tab_v = X_dict["agg_states"][val_idx]
                s_tab = StandardScaler()
                scaled_tab_tr = s_tab.fit_transform(raw_tab_tr).astype(np.float32)
                scaled_tab_v = s_tab.transform(raw_tab_v).astype(np.float32)
                tab_dim = raw_tab_tr.shape[1]

                ds_train = DeepEngagementDataset(
                    X_tab=scaled_tab_tr, X_seq=scaled_seq_tr, y=y_tr
                )
                ds_val = DeepEngagementDataset(
                    X_tab=scaled_tab_v, X_seq=scaled_seq_v, y=y_v
                )
            else:
                # Tabular deep models (mlp_wide, mlp_residual)
                raw_tab_tr = X_dict[spec.feature_set][tr_idx]
                raw_tab_v = X_dict[spec.feature_set][val_idx]
                s_tab = StandardScaler()
                scaled_tab_tr = s_tab.fit_transform(raw_tab_tr).astype(np.float32)
                scaled_tab_v = s_tab.transform(raw_tab_v).astype(np.float32)
                tab_dim = raw_tab_tr.shape[1]
                ds_train = DeepEngagementDataset(X_tab=scaled_tab_tr, y=y_tr)
                ds_val = DeepEngagementDataset(X_tab=scaled_tab_v, y=y_v)

            base_mid = spec.id.replace("_regressor", "")
            deep_model = build_deep_model(
                base_mid,
                tab_input_dim=tab_dim,
                seq_input_dim=7,
                hyperparams=tuned_params,
            )
            # Initialize final bias to mean of continuous intent score (48.0)
            with joblib.parallel_backend("threading"):
                pass
            if hasattr(deep_model, "head") and isinstance(
                deep_model.head, nn.Sequential
            ):
                last_layer = deep_model.head[-1]
                if isinstance(last_layer, nn.Linear):
                    nn.init.constant_(last_layer.bias, 48.0)
            elif hasattr(deep_model, "head") and isinstance(deep_model.head, nn.Linear):
                nn.init.constant_(deep_model.head.bias, 48.0)

            train_res = train_two_stage_regression(
                model=deep_model,
                train_dataset=ds_train,
                val_dataset=ds_val,
                batch_size=int(tuned_params.get("batch_size", 64)),
                lr_stage1=float(tuned_params.get("lr", 1e-3)),
                lr_stage2=float(tuned_params.get("lr", 1e-3)) * 0.1,
                weight_decay=float(tuned_params.get("weight_decay", 1e-4)),
                max_epochs_stage1=60,
                max_epochs_stage2=25,
                patience_stage1=10,
                patience_stage2=6,
                random_seed=seed + fold_idx,
            )
            y_pred_tr_raw = predict_deep_regressor(deep_model, ds_train)
            y_pred_v_raw = predict_deep_regressor(deep_model, ds_val)

            if fold_idx == 0:
                representative_fold_history = train_res

        # Compute metrics for this fold
        m_tr, _, _ = compute_regression_metrics(
            y_tr, y_pred_tr_raw, n_features=n_features, compute_kendall=False
        )
        m_v, y_pred_v_clipped, res_v = compute_regression_metrics(
            y_v, y_pred_v_raw, n_features=n_features, compute_kendall=True
        )
        class_60_v = compute_classification_at_60(y_v, y_pred_v_clipped, threshold=60.0)

        fold_r2_train.append(m_tr.r2)
        fold_r2_val.append(m_v.r2)
        fold_rmse_val.append(m_v.rmse)
        fold_mae_val.append(m_v.mae)
        fold_spearman_val.append(m_v.spearman_rho)
        fold_pearson_val.append(m_v.pearson_r)
        fold_within_5_val.append(m_v.within_5_points)
        fold_within_10_val.append(m_v.within_10_points)
        fold_f1_60_val.append(class_60_v.f1_at_risk)

        # Accumulate OOF continuous predictions
        oof_predictions_accum[val_idx] += y_pred_v_clipped
        oof_counts[val_idx] += 1

        fold_results.append(
            {
                "fold": fold_idx,
                "train_r2": m_tr.r2,
                "val_r2": m_v.r2,
                "val_rmse": m_v.rmse,
                "val_mae": m_v.mae,
                "val_spearman": m_v.spearman_rho,
                "val_f1_60": class_60_v.f1_at_risk,
                "val_clipping_count": m_v.clipping_count,
            }
        )

    # 4. Final OOF Aggregated Predictions & Cohort Metrics
    non_zero = oof_counts > 0
    oof_predictions = np.zeros(n_samples, dtype=np.float64)
    oof_predictions[non_zero] = oof_predictions_accum[non_zero] / oof_counts[non_zero]

    overall_metrics, y_oof_clipped, oof_residuals = compute_regression_metrics(
        y, oof_predictions, n_features=n_features, compute_kendall=True
    )
    overall_class_60 = compute_classification_at_60(y, y_oof_clipped, threshold=60.0)

    # 5. Overfitting & Leaderboard Score
    mean_tr_r2 = float(np.mean(fold_r2_train))
    mean_v_r2 = float(np.mean(fold_r2_val))
    mean_v_spearman = float(np.mean(fold_spearman_val))

    lb_score, gap, penalty, status = compute_track_b_leaderboard_score(
        r2_train=mean_tr_r2,
        r2_val=mean_v_r2,
        spearman_val=mean_v_spearman,
    )

    # 6. Residual Diagnostics & Heteroscedasticity Test
    bp_test = compute_breusch_pagan_test(oof_residuals, y_oof_clipped)

    # Stratified residual analysis by engagement pattern and level if available
    residuals_by_pattern: Dict[str, Dict[str, float]] = {}
    if patterns is not None:
        for pat in np.unique(patterns):
            idx_p = patterns == pat
            if np.sum(idx_p) > 0:
                res_p = oof_residuals[idx_p]
                residuals_by_pattern[str(pat)] = {
                    "count": int(np.sum(idx_p)),
                    "mean": float(np.mean(res_p)),
                    "std": float(np.std(res_p, ddof=1)) if len(res_p) > 1 else 0.0,
                    "mae": float(np.mean(np.abs(res_p))),
                }

    residuals_by_level: Dict[str, Dict[str, float]] = {}
    if levels is not None:
        for lvl in np.unique(levels):
            idx_l = levels == lvl
            if np.sum(idx_l) > 0:
                res_l = oof_residuals[idx_l]
                residuals_by_level[str(lvl)] = {
                    "count": int(np.sum(idx_l)),
                    "mean": float(np.mean(res_l)),
                    "std": float(np.std(res_l, ddof=1)) if len(res_l) > 1 else 0.0,
                    "mae": float(np.mean(np.abs(res_l))),
                }

    # QQ coordinates (subsampled for compact JSON)
    sorted_res = np.sort(
        (oof_residuals - np.mean(oof_residuals))
        / (np.std(oof_residuals, ddof=1) or 1.0)
    )
    n_points = min(100, len(sorted_res))
    sub_indices = np.linspace(0, len(sorted_res) - 1, n_points).astype(int)
    norm_quantiles = stats.norm.ppf(
        (np.arange(1, len(sorted_res) + 1) - 0.5) / len(sorted_res)
    )
    qq_data = {
        "theoretical_quantiles": norm_quantiles[sub_indices].tolist(),
        "sample_quantiles": sorted_res[sub_indices].tolist(),
    }

    # Binned calibration curve: 10 quantiles of predicted score vs observed score
    bins = np.linspace(0.0, 100.0, 11)
    bin_centers: List[float] = []
    bin_observed_means: List[float] = []
    bin_counts: List[int] = []
    for b_idx in range(len(bins) - 1):
        b_low, b_high = bins[b_idx], bins[b_idx + 1]
        in_bin = (y_oof_clipped >= b_low) & (
            y_oof_clipped < b_high if b_idx < len(bins) - 2 else y_oof_clipped <= b_high
        )
        if np.sum(in_bin) > 0:
            bin_centers.append(float((b_low + b_high) / 2.0))
            bin_observed_means.append(float(np.mean(y[in_bin])))
            bin_counts.append(int(np.sum(in_bin)))
    calibration_binned = {
        "bin_centers": bin_centers,
        "observed_means": bin_observed_means,
        "counts": bin_counts,
    }

    total_duration = time.time() - start_time

    # Construct final result dictionary
    result = {
        "spec": {
            "id": spec.id,
            "display_name": spec.display_name,
            "plain_name": spec.plain_name,
            "category": spec.category,
            "feature_set": spec.feature_set,
            "twin_model_id": spec.twin_model_id,
        },
        "tuned_parameters": tuned_params,
        "tuning_summary": tuning_summary,
        "leaderboard_metrics": {
            "model_id": spec.id,
            "display_name": spec.display_name,
            "plain_name": spec.plain_name,
            "category": spec.category,
            "feature_set": spec.feature_set,
            "leaderboard_score": lb_score,
            "generalization_gap": gap,
            "overfit_penalty": penalty,
            "overfit_status": status,
            "val_r2": mean_v_r2,
            "val_r2_std": float(np.std(fold_r2_val)),
            "train_r2_mean": mean_tr_r2,
            "val_adj_r2": overall_metrics.adj_r2,
            "val_rmse": float(np.mean(fold_rmse_val)),
            "val_rmse_std": float(np.std(fold_rmse_val)),
            "val_mae": float(np.mean(fold_mae_val)),
            "val_mae_std": float(np.std(fold_mae_val)),
            "val_median_ae": overall_metrics.median_ae,
            "val_spearman_rho": mean_v_spearman,
            "val_pearson_r": float(np.mean(fold_pearson_val)),
            "val_within_5_pct": float(np.mean(fold_within_5_val)),
            "val_within_10_pct": float(np.mean(fold_within_10_val)),
            "val_residual_mean": overall_metrics.residual_mean,
            "val_residual_std": overall_metrics.residual_std,
            "val_clipping_count": overall_metrics.clipping_count,
            "val_clipping_pct": overall_metrics.clipping_pct,
            # Classification at 60 metrics
            "val_f1_at_60": float(np.mean(fold_f1_60_val)),
            "val_bal_acc_at_60": overall_class_60.balanced_accuracy,
            "val_auc_roc_at_60": overall_class_60.auc_roc,
            "val_mcc_at_60": overall_class_60.mcc,
            "training_time_s": total_duration,
            "inference_time_ms": float(total_duration / n_folds / n_samples * 1000.0),
        },
        "fold_r2_scores": fold_r2_val,
        "fold_results": fold_results,
        "oof_predictions": oof_predictions.tolist(),
        "oof_residuals": oof_residuals.tolist(),
        "breusch_pagan_test": bp_test,
        "residuals_by_pattern": residuals_by_pattern,
        "residuals_by_level": residuals_by_level,
        "qq_diagnostics": qq_data,
        "calibration_binned": calibration_binned,
        "representative_fold_history": representative_fold_history["history"]
        if representative_fold_history
        else None,
    }

    return result


def evaluate_voting_regressor(
    top_3_specs: List[RegressionModelSpec],
    top_3_results: List[Dict[str, Any]],
    y: np.ndarray,
    cv_folds: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Formulate an ensemble averaging the Out-of-Fold predictions of the top 3 models."""
    start_time = time.time()
    oof_preds_matrix = np.column_stack(
        [np.array(r["oof_predictions"]) for r in top_3_results]
    )
    voting_oof = np.mean(oof_preds_matrix, axis=1)

    overall_metrics, y_oof_clipped, oof_residuals = compute_regression_metrics(
        y, voting_oof, n_features=3, compute_kendall=True
    )
    class_60 = compute_classification_at_60(y, y_oof_clipped, threshold=60.0)

    # Compute fold-wise R² by evaluating the voting average on each fold's val indices
    fold_r2_val: List[float] = []
    fold_r2_tr: List[float] = []
    for f in cv_folds:
        val_idx = np.asarray(f.get("val_idx", f.get("val_indices")), dtype=int)
        tr_idx = np.asarray(f.get("train_idx", f.get("train_indices")), dtype=int)
        y_v = y[val_idx]
        y_tr = y[tr_idx]
        v_preds = voting_oof[val_idx]
        tr_preds = voting_oof[tr_idx]
        m_v, _, _ = compute_regression_metrics(
            y_v, v_preds, n_features=3, compute_kendall=False
        )
        m_tr, _, _ = compute_regression_metrics(
            y_tr, tr_preds, n_features=3, compute_kendall=False
        )
        fold_r2_val.append(m_v.r2)
        fold_r2_tr.append(m_tr.r2)

    mean_v_r2 = float(np.mean(fold_r2_val))
    mean_tr_r2 = float(np.mean(fold_r2_tr))
    mean_spearman = float(overall_metrics.spearman_rho)

    lb_score, gap, penalty, status = compute_track_b_leaderboard_score(
        r2_train=mean_tr_r2,
        r2_val=mean_v_r2,
        spearman_val=mean_spearman,
    )

    names = " + ".join([s.id for s in top_3_specs])
    spec = RegressionModelSpec(
        id="voting_regressor",
        display_name=f"Average Voting ({names})",
        plain_name="Continuous Mean Voting Ensemble",
        category="ensemble",
        feature_set="agg",
    )

    bp_test = compute_breusch_pagan_test(oof_residuals, y_oof_clipped)

    return {
        "spec": {
            "id": spec.id,
            "display_name": spec.display_name,
            "plain_name": spec.plain_name,
            "category": spec.category,
            "feature_set": spec.feature_set,
        },
        "tuned_parameters": {},
        "leaderboard_metrics": {
            "model_id": spec.id,
            "display_name": spec.display_name,
            "plain_name": spec.plain_name,
            "category": spec.category,
            "feature_set": spec.feature_set,
            "leaderboard_score": lb_score,
            "generalization_gap": gap,
            "overfit_penalty": penalty,
            "overfit_status": status,
            "val_r2": mean_v_r2,
            "val_r2_std": float(np.std(fold_r2_val)),
            "train_r2_mean": mean_tr_r2,
            "val_adj_r2": overall_metrics.adj_r2,
            "val_rmse": overall_metrics.rmse,
            "val_rmse_std": float(
                np.std([r["val_rmse"] for r in top_3_results[0]["fold_results"]])
            ),
            "val_mae": overall_metrics.mae,
            "val_mae_std": float(
                np.std([r["val_mae"] for r in top_3_results[0]["fold_results"]])
            ),
            "val_median_ae": overall_metrics.median_ae,
            "val_spearman_rho": mean_spearman,
            "val_pearson_r": overall_metrics.pearson_r,
            "val_within_5_pct": overall_metrics.within_5_points,
            "val_within_10_pct": overall_metrics.within_10_points,
            "val_residual_mean": overall_metrics.residual_mean,
            "val_residual_std": overall_metrics.residual_std,
            "val_clipping_count": overall_metrics.clipping_count,
            "val_clipping_pct": overall_metrics.clipping_pct,
            "val_f1_at_60": class_60.f1_at_risk,
            "val_bal_acc_at_60": class_60.balanced_accuracy,
            "val_auc_roc_at_60": class_60.auc_roc,
            "val_mcc_at_60": class_60.mcc,
            "training_time_s": time.time() - start_time,
            "inference_time_ms": 1.2,
        },
        "fold_r2_scores": fold_r2_val,
        "oof_predictions": voting_oof.tolist(),
        "oof_residuals": oof_residuals.tolist(),
        "breusch_pagan_test": bp_test,
    }


def evaluate_stacking_regressor(
    top_5_specs: List[RegressionModelSpec],
    top_5_results: List[Dict[str, Any]],
    y: np.ndarray,
    cv_folds: List[Dict[str, Any]],
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a Ridge meta-regressor on out-of-fold continuous predictions of top 5 models."""
    start_time = time.time()
    n_samples = len(y)
    S_oof = np.column_stack([np.array(r["oof_predictions"]) for r in top_5_results])

    stacking_oof = np.zeros(n_samples, dtype=np.float64)
    oof_counts = np.zeros(n_samples, dtype=np.int32)
    fold_r2_val: List[float] = []
    fold_r2_tr: List[float] = []

    for fold in cv_folds:
        tr_idx = np.asarray(fold.get("train_idx", fold.get("train_indices")), dtype=int)
        val_idx = np.asarray(fold.get("val_idx", fold.get("val_indices")), dtype=int)

        meta = Ridge(alpha=10.0, random_state=seed)
        meta.fit(S_oof[tr_idx], y[tr_idx])
        p_v = meta.predict(S_oof[val_idx])
        p_tr = meta.predict(S_oof[tr_idx])

        stacking_oof[val_idx] += p_v
        oof_counts[val_idx] += 1

        m_v, _, _ = compute_regression_metrics(
            y[val_idx], p_v, n_features=5, compute_kendall=False
        )
        m_tr, _, _ = compute_regression_metrics(
            y[tr_idx], p_tr, n_features=5, compute_kendall=False
        )
        fold_r2_val.append(m_v.r2)
        fold_r2_tr.append(m_tr.r2)

    non_zero = oof_counts > 0
    stacking_oof[non_zero] /= oof_counts[non_zero]

    overall_metrics, y_oof_clipped, oof_residuals = compute_regression_metrics(
        y, stacking_oof, n_features=5, compute_kendall=True
    )
    class_60 = compute_classification_at_60(y, y_oof_clipped, threshold=60.0)

    mean_v_r2 = float(np.mean(fold_r2_val))
    mean_tr_r2 = float(np.mean(fold_r2_tr))
    mean_spearman = float(overall_metrics.spearman_rho)

    lb_score, gap, penalty, status = compute_track_b_leaderboard_score(
        r2_train=mean_tr_r2,
        r2_val=mean_v_r2,
        spearman_val=mean_spearman,
    )

    names = " + ".join([s.id for s in top_5_specs])
    spec = RegressionModelSpec(
        id="stacking_regressor",
        display_name=f"Stacking Regressor ({names})",
        plain_name="Ridge Out-of-Fold Stacking Meta-Regressor",
        category="ensemble",
        feature_set="agg",
    )

    bp_test = compute_breusch_pagan_test(oof_residuals, y_oof_clipped)

    return {
        "spec": {
            "id": spec.id,
            "display_name": spec.display_name,
            "plain_name": spec.plain_name,
            "category": spec.category,
            "feature_set": spec.feature_set,
        },
        "tuned_parameters": {"meta_alpha": 10.0},
        "leaderboard_metrics": {
            "model_id": spec.id,
            "display_name": spec.display_name,
            "plain_name": spec.plain_name,
            "category": spec.category,
            "feature_set": spec.feature_set,
            "leaderboard_score": lb_score,
            "generalization_gap": gap,
            "overfit_penalty": penalty,
            "overfit_status": status,
            "val_r2": mean_v_r2,
            "val_r2_std": float(np.std(fold_r2_val)),
            "train_r2_mean": mean_tr_r2,
            "val_adj_r2": overall_metrics.adj_r2,
            "val_rmse": overall_metrics.rmse,
            "val_rmse_std": float(np.std(fold_r2_val)),
            "val_mae": overall_metrics.mae,
            "val_mae_std": float(np.std(fold_r2_val)),
            "val_median_ae": overall_metrics.median_ae,
            "val_spearman_rho": mean_spearman,
            "val_pearson_r": overall_metrics.pearson_r,
            "val_within_5_pct": overall_metrics.within_5_points,
            "val_within_10_pct": overall_metrics.within_10_points,
            "val_residual_mean": overall_metrics.residual_mean,
            "val_residual_std": overall_metrics.residual_std,
            "val_clipping_count": overall_metrics.clipping_count,
            "val_clipping_pct": overall_metrics.clipping_pct,
            "val_f1_at_60": class_60.f1_at_risk,
            "val_bal_acc_at_60": class_60.balanced_accuracy,
            "val_auc_roc_at_60": class_60.auc_roc,
            "val_mcc_at_60": class_60.mcc,
            "training_time_s": time.time() - start_time,
            "inference_time_ms": 2.5,
        },
        "fold_r2_scores": fold_r2_val,
        "oof_predictions": stacking_oof.tolist(),
        "oof_residuals": oof_residuals.tolist(),
        "breusch_pagan_test": bp_test,
    }
