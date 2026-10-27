"""
Comprehensive Academic Risk Evaluation Metrics Suite.
Computes:
  1. Classification Metrics (Accuracy, Balanced Accuracy, Precision, Recall, F1, F2,
     ROC-AUC, PR-AUC, MCC, Cohen's Kappa, Log-Loss, Specificity, Confusion Matrix)
  2. Probability & Risk-Index Quality (Brier Score, Brier Skill Score, Efron's Pseudo-R2,
     MAE, RMSE, Expected Calibration Error, Spearman Rank Correlation)
  3. Early-Warning Metrics (Lead Time in Semesters, Recall@top10%, Recall@top20%,
     Precision@k, Detected Before GPA < 2.0 Rate)
  4. Operational Efficiency (Inference Latency per Student, Disk Footprint)

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("AcademicIntelligence.EvaluationMetrics")


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Computes Expected Calibration Error (ECE) across n equal-width probability bins.
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    curve_points: List[Dict[str, float]] = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob >= low) & (y_prob <= high if i == n_bins - 1 else y_prob < high)
        bin_count = int(np.sum(mask))

        if bin_count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
            curve_points.append({
                "bin": i + 1,
                "bin_lower": round(float(low), 3),
                "bin_upper": round(float(high), 3),
                "bin_count": bin_count,
                "mean_predicted_prob": round(bin_conf, 4),
                "fraction_of_positives": round(bin_acc, 4),
            })
        else:
            curve_points.append({
                "bin": i + 1,
                "bin_lower": round(float(low), 3),
                "bin_upper": round(float(high), 3),
                "bin_count": 0,
                "mean_predicted_prob": round(float((low + high) / 2.0), 4),
                "fraction_of_positives": 0.0,
            })

    return float(ece), curve_points


def compute_early_warning_metrics(
    df_samples: pd.DataFrame,
    df_outcomes: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Calculates early-warning novelty metrics:
      - Average lead time: semesters before attrition that student is first flagged
      - Recall at top-10% and top-20% highest risk scores
      - Precision@100 and Precision@200
      - Detected before GPA < 2.0 rate vs status-quo threshold rule
    """
    df_eval = df_samples.copy()
    df_eval["pred_prob"] = y_prob
    df_eval["is_flagged"] = (y_prob >= threshold).astype(int)

    # Merge student outcomes for true attrition semester
    outcomes_map = df_outcomes.set_index("student_id").to_dict("index")

    # 1. Lead Time Calculation
    # For students who actually attrited, find their earliest flagged semester
    lead_times: List[float] = []
    flagged_before_gpa_2_count = 0
    total_attrited_in_eval = 0

    attrited_sids = set(df_outcomes[df_outcomes["attrited"] == 1]["student_id"])
    eval_attrited_sids = set(df_eval["student_id"]) & attrited_sids
    total_attrited_students = len(eval_attrited_sids)

    for sid in eval_attrited_sids:
        s_records = df_eval[df_eval["student_id"] == sid].sort_values("window_t")
        outcome_info = outcomes_map.get(sid, {})
        attr_sem = outcome_info.get("attrition_semester")

        if pd.isna(attr_sem) or attr_sem is None:
            attr_sem = outcome_info.get("last_semester_observed", 8)
        attr_sem = float(attr_sem)

        flagged_records = s_records[s_records["is_flagged"] == 1]
        if not flagged_records.empty:
            first_flag_sem = float(flagged_records.iloc[0]["window_t"])
            lead = max(0.0, attr_sem - first_flag_sem)
            lead_times.append(lead)

            # Check if GPA was still >= 2.0 when first flagged
            first_gpa = float(flagged_records.iloc[0].get("last_gpa", 0.0))
            if first_gpa >= 2.0:
                flagged_before_gpa_2_count += 1

    avg_lead_time = float(np.mean(lead_times)) if lead_times else 0.0
    detected_before_gpa_2_rate = (
        float(flagged_before_gpa_2_count / total_attrited_students)
        if total_attrited_students > 0
        else 0.0
    )

    # 2. Recall at Top-K % Percentiles
    n_samples = len(df_eval)
    y_true_all = df_eval["target"].values
    total_positives = int(np.sum(y_true_all == 1))

    # Sort descending by predicted risk probability
    sorted_indices = np.argsort(-y_prob)

    top_10_cutoff = max(1, int(0.10 * n_samples))
    top_20_cutoff = max(1, int(0.20 * n_samples))

    pos_in_top_10 = int(np.sum(y_true_all[sorted_indices[:top_10_cutoff]] == 1))
    pos_in_top_20 = int(np.sum(y_true_all[sorted_indices[:top_20_cutoff]] == 1))

    recall_at_top_10 = float(pos_in_top_10 / max(total_positives, 1))
    recall_at_top_20 = float(pos_in_top_20 / max(total_positives, 1))

    # Precision @ fixed k
    prec_at_100 = float(np.mean(y_true_all[sorted_indices[:min(100, n_samples)]] == 1))
    prec_at_200 = float(np.mean(y_true_all[sorted_indices[:min(200, n_samples)]] == 1))

    return {
        "average_lead_time_semesters": round(avg_lead_time, 2),
        "lead_time_distribution": [round(x, 1) for x in lead_times],
        "recall_at_top_10_pct": round(recall_at_top_10, 4),
        "recall_at_top_20_pct": round(recall_at_top_20, 4),
        "precision_at_100": round(prec_at_100, 4),
        "precision_at_200": round(prec_at_200, 4),
        "detected_before_gpa_below_2_rate": round(detected_before_gpa_2_rate, 4),
        "total_attrited_evaluated": total_attrited_students,
        "attrited_flagged_count": len(lead_times),
    }


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    df_samples: Optional[pd.DataFrame] = None,
    df_outcomes: Optional[pd.DataFrame] = None,
    inference_time_ms: float = 0.0,
    model_size_kb: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes all classification, risk-index quality, early warning, and efficiency metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    # 1. Classification Metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))

    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = 0.0

    try:
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        kappa = 0.0

    try:
        ll = float(log_loss(y_true, y_prob))
    except Exception:
        ll = 1.0

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    except Exception:
        tn, fp, fn, tp = 0, 0, 0, 0
        spec = 0.0

    # 2. Probability & Risk-Index Quality Metrics
    # Academic Risk Index (0-100)
    ari = y_prob * 100.0

    brier = float(brier_score_loss(y_true, y_prob))
    # Climatology reference: baseline predicting class prevalence
    p_base = float(np.mean(y_true))
    brier_ref = float(p_base * (1.0 - p_base))
    brier_skill_score = float(1.0 - (brier / brier_ref)) if brier_ref > 1e-9 else 0.0

    # Efron's Pseudo-R2: 1 - sum(y - p)^2 / sum(y - y_bar)^2
    ss_res = float(np.sum((y_true - y_prob) ** 2))
    ss_tot = float(np.sum((y_true - p_base) ** 2))
    pseudo_r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-9 else 0.0

    mae = float(np.mean(np.abs(y_true - y_prob)))
    rmse = float(np.sqrt(np.mean((y_true - y_prob) ** 2)))

    ece, cal_curve = compute_expected_calibration_error(y_true, y_prob, n_bins=10)

    try:
        spearman_corr, spearman_pval = stats.spearmanr(ari, y_true)
        spearman_corr = float(spearman_corr)
        spearman_pval = float(spearman_pval)
    except Exception:
        spearman_corr, spearman_pval = 0.0, 1.0

    # 3. Early Warning Novelty Metrics
    early_warning = {}
    if df_samples is not None and df_outcomes is not None:
        early_warning = compute_early_warning_metrics(
            df_samples=df_samples,
            df_outcomes=df_outcomes,
            y_prob=y_prob,
            threshold=threshold,
        )

    return {
        "classification": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "f2_score": round(f2, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "matthews_corrcoef": round(mcc, 4),
            "cohen_kappa": round(kappa, 4),
            "log_loss": round(ll, 4),
            "specificity": round(spec, 4),
            "threshold": round(float(threshold), 4),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        },
        "risk_index_quality": {
            "brier_score": round(brier, 4),
            "brier_skill_score": round(brier_skill_score, 4),
            "pseudo_r2": round(pseudo_r2, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "expected_calibration_error": round(ece, 4),
            "spearman_correlation": round(spearman_corr, 4),
            "spearman_pvalue": float(spearman_pval),
            "calibration_curve": cal_curve,
        },
        "early_warning": early_warning,
        "efficiency": {
            "inference_time_ms_per_student": round(float(inference_time_ms), 3),
            "model_size_kb": round(float(model_size_kb), 2),
        },
    }
