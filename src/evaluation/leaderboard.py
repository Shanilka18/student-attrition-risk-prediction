"""Leaderboard computation and model ranking for ERI Project (Track C and Track R).

Calculates composite scores, normalizes metrics across models, applies overfitting/calibration
penalties, groups models by category, enforces hard validation gates, writes detailed
score breakdowns, and formats outputs for frontend consumption.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.utils.logging_config import setup_logger
from src.utils.paths import get_artifacts_dir, get_configs_dir
from src.utils.run_context import get_current_run_info

logger = setup_logger("leaderboard")

CATEGORY_MAP = {
    # Linear / Shallow
    "logistic_l2": "Linear/Shallow",
    "logistic_l1": "Linear/Shallow",
    "logistic_elasticnet": "Linear/Shallow",
    "linear_svm": "Linear/Shallow",
    "gaussian_nb": "Linear/Shallow",
    "knn_classifier": "Linear/Shallow",
    "decision_tree_classifier": "Linear/Shallow",
    "mlp_shallow_classifier": "Linear/Shallow",
    "ridge": "Linear/Shallow",
    "lasso": "Linear/Shallow",
    "elasticnet_reg": "Linear/Shallow",
    "knn_regressor": "Linear/Shallow",
    "decision_tree_regressor": "Linear/Shallow",
    "mlp_shallow_regressor": "Linear/Shallow",
    # Tree Ensembles
    "random_forest": "Tree Ensembles",
    "extra_trees": "Tree Ensembles",
    "adaboost": "Tree Ensembles",
    "gradient_boosting": "Tree Ensembles",
    "hist_gradient_boosting": "Tree Ensembles",
    "xgboost": "Tree Ensembles",
    "lightgbm": "Tree Ensembles",
    "catboost": "Tree Ensembles",
    # Deep Tabular
    "deep_mlp": "Deep Learning",
    "tabular_resnet": "Deep Learning",
    "tabnet": "Deep Learning",
    "cnn_1d": "Deep Learning",
    # Meta Ensembles
    "soft_voting_top3": "Ensembles (voting/stacking)",
    "stacking_top3": "Ensembles (voting/stacking)",
}


def load_weights_config() -> Dict[str, Any]:
    weights_path = get_configs_dir() / "leaderboard_weights.yaml"
    with open(weights_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_hard_gates(
    row: Dict[str, Any],
    cfg: Dict[str, Any],
    prevalence: float = 0.1643,
) -> Tuple[bool, str, List[str]]:
    """Check hard disqualification gates for a classification model.

    Gates:
    1. F1 == 0 at chosen threshold -> Disqualified
    2. ROC-AUC < 0.55 -> Disqualified
    3. ECE > max_ece (e.g. 0.15) -> Disqualified
    4. Brier >= prevalence * (1 - prevalence) -> Worse than constant-prevalence baseline -> Disqualified

    Returns:
        (is_qualified, status, disqualification_reasons)
    """
    gates_cfg = cfg.get("hard_gates", {})
    min_f1 = float(gates_cfg.get("min_f1", 0.0001))
    min_roc_auc = float(gates_cfg.get("min_roc_auc", 0.55))
    max_ece = float(gates_cfg.get("max_ece", 0.15))
    check_brier_baseline = bool(gates_cfg.get("brier_worse_than_baseline", True))

    f1 = float(row.get("f1", 0.0))
    roc_auc = float(row.get("roc_auc", 0.0))
    ece = float(row.get("ece", 1.0))
    brier = float(row.get("brier_score", row.get("brier", 1.0)))

    reasons: List[str] = []

    # Check for missing metrics
    required_keys = ["pr_auc", "roc_auc", "f1", "balanced_accuracy", "brier_score", "ece"]
    for k in required_keys:
        val = row.get(k if k in row else ("brier" if k == "brier_score" else k))
        if val is None or (isinstance(val, (float, int)) and np.isnan(val)):
            return False, "incomplete", [f"Missing required metric '{k}'"]

    # Gate 1: F1
    if f1 < min_f1:
        reasons.append(f"F1 ({f1:.4f}) <= 0 at optimal threshold (no true positives predicted)")

    # Gate 2: ROC-AUC
    if roc_auc < min_roc_auc:
        reasons.append(f"ROC-AUC ({roc_auc:.4f}) < {min_roc_auc:.2f} (near-random discrimination)")

    # Gate 3: ECE
    if ece > max_ece:
        reasons.append(f"ECE ({ece:.4f}) > {max_ece:.2f} (unacceptable calibration error)")

    # Gate 4: Brier vs prevalence baseline
    brier_baseline = prevalence * (1.0 - prevalence)
    if check_brier_baseline and brier >= brier_baseline:
        reasons.append(
            f"Brier ({brier:.4f}) >= constant-prevalence baseline ({brier_baseline:.4f}, zero or negative skill)"
        )

    if reasons:
        return False, "disqualified", reasons
    return True, "passed", []


def compute_classification_composite(
    val_row: Dict[str, Any],
    train_row: Optional[Dict[str, Any]],
    weights: Dict[str, Any],
    brier_min: float = 0.05,
    brier_max: float = 0.30,
    ece_min: float = 0.01,
    ece_max: float = 0.30,
    return_breakdown: bool = False,
) -> Any:
    """Compute Track C composite score with explicit penalty and breakdown accounting."""
    # Strict missing metric check
    required_keys = ["pr_auc", "roc_auc", "f1", "balanced_accuracy"]
    for rk in required_keys:
        if rk not in val_row or val_row[rk] is None or np.isnan(float(val_row[rk])):
            raise ValueError(f"Missing required classification metric '{rk}' for model {val_row.get('model')}")

    pr_auc = float(val_row["pr_auc"])
    roc_auc = float(val_row["roc_auc"])
    f1 = float(val_row["f1"])
    bal_acc = float(val_row["balanced_accuracy"])
    raw_brier = float(val_row.get("brier_score", val_row.get("brier", 0.25)))
    raw_ece = float(val_row.get("ece", 0.15))

    if np.isnan(raw_brier) or np.isnan(raw_ece):
        raise ValueError(f"Brier or ECE is NaN for model {val_row.get('model')}")

    # Min-max normalise Brier and ECE across all evaluated models in pool
    norm_brier = (raw_brier - brier_min) / (brier_max - brier_min + 1e-8) if brier_max > brier_min else 0.0
    norm_ece = (raw_ece - ece_min) / (ece_max - ece_min + 1e-8) if ece_max > ece_min else 0.0

    def _val(x, d):
        return float(x.get("weight", d)) if isinstance(x, dict) else float(x if x is not None else d)

    w_pr = _val(weights.get("pr_auc"), 0.30)
    w_roc = _val(weights.get("roc_auc"), 0.20)
    w_f1 = _val(weights.get("f1"), 0.15)
    w_bal = _val(weights.get("balanced_accuracy"), 0.10)
    w_brier = _val(weights.get("brier_weight"), 0.15)
    w_ece = _val(weights.get("ece_weight"), 0.10)

    contrib_pr = w_pr * pr_auc
    contrib_roc = w_roc * roc_auc
    contrib_f1 = w_f1 * f1
    contrib_bal = w_bal * bal_acc
    contrib_brier = w_brier * (1.0 - norm_brier)
    contrib_ece = w_ece * (1.0 - norm_ece)

    base_score = contrib_pr + contrib_roc + contrib_f1 + contrib_bal + contrib_brier + contrib_ece

    # Overfitting gap penalty: gap between train ROC-AUC and val ROC-AUC > threshold
    overfit_gap = 0.0
    if train_row is not None:
        tr_auc = float(train_row.get("roc_auc", roc_auc))
        overfit_gap = max(0.0, tr_auc - roc_auc)

    overfit_thresh = weights.get("overfit_penalty_threshold", 0.08)
    overfit_penalty = max(0.0, overfit_gap - overfit_thresh)

    # Calibration penalty: if ECE > threshold
    ece_thresh = weights.get("ece_penalty_threshold", 0.10)
    ece_penalty = max(0.0, raw_ece - ece_thresh)

    total_penalty = overfit_penalty + ece_penalty
    composite = float(np.clip(base_score - total_penalty, 0.0, 1.0))

    breakdown = {
        "pr_auc_val": pr_auc,
        "pr_auc_norm": pr_auc,
        "pr_auc_weight": w_pr,
        "pr_auc_contrib": round(contrib_pr, 4),
        "roc_auc_val": roc_auc,
        "roc_auc_norm": roc_auc,
        "roc_auc_weight": w_roc,
        "roc_auc_contrib": round(contrib_roc, 4),
        "f1_val": f1,
        "f1_norm": f1,
        "f1_weight": w_f1,
        "f1_contrib": round(contrib_f1, 4),
        "bal_acc_val": bal_acc,
        "bal_acc_norm": bal_acc,
        "bal_acc_weight": w_bal,
        "bal_acc_contrib": round(contrib_bal, 4),
        "brier_val": raw_brier,
        "brier_norm": round(1.0 - norm_brier, 4),
        "brier_weight": w_brier,
        "brier_contrib": round(contrib_brier, 4),
        "ece_val": raw_ece,
        "ece_norm": round(1.0 - norm_ece, 4),
        "ece_weight": w_ece,
        "ece_contrib": round(contrib_ece, 4),
        "base_score": round(base_score, 4),
        "overfit_gap": round(overfit_gap, 4),
        "overfit_penalty": round(overfit_penalty, 4),
        "ece_penalty": round(ece_penalty, 4),
        "total_penalty": round(total_penalty, 4),
        "composite_score": round(composite, 4),
    }

    if return_breakdown:
        return composite, overfit_gap, total_penalty, breakdown
    return composite, overfit_gap, total_penalty


def compute_regression_composite(
    val_row: Dict[str, Any],
    train_row: Optional[Dict[str, Any]],
    weights: Dict[str, Any],
    mae_min: float = 1.0,
    mae_max: float = 50.0,
    rmse_min: float = 1.0,
    rmse_max: float = 50.0,
    return_breakdown: bool = False,
) -> Any:
    """Compute Track R composite score."""
    r2 = float(val_row.get("r2", 0.0))
    raw_mae = float(val_row.get("mae", 50.0))
    raw_rmse = float(val_row.get("rmse", 50.0))
    spearman = float(val_row.get("spearman_rho", 0.0))

    norm_mae = (raw_mae - mae_min) / (mae_max - mae_min + 1e-8) if mae_max > mae_min else 0.0
    norm_rmse = (raw_rmse - rmse_min) / (rmse_max - rmse_min + 1e-8) if rmse_max > rmse_min else 0.0

    def _val(x, d):
        return float(x.get("weight", d)) if isinstance(x, dict) else float(x if x is not None else d)

    w_r2 = _val(weights.get("r2"), 0.35)
    w_mae = _val(weights.get("mae_weight"), 0.20)
    w_rmse = _val(weights.get("rmse_weight"), 0.20)
    w_sp = _val(weights.get("spearman_rho"), 0.15)
    w_ov = _val(weights.get("overfit_weight"), 0.10)

    overfit_gap = 0.0
    if train_row is not None:
        tr_r2 = float(train_row.get("r2", r2))
        overfit_gap = max(0.0, tr_r2 - r2)

    contrib_r2 = w_r2 * max(0.0, r2)
    contrib_mae = w_mae * (1.0 - norm_mae)
    contrib_rmse = w_rmse * (1.0 - norm_rmse)
    contrib_sp = w_sp * max(0.0, spearman)
    contrib_ov = w_ov * (1.0 - np.clip(overfit_gap, 0.0, 1.0))

    base_score = contrib_r2 + contrib_mae + contrib_rmse + contrib_sp + contrib_ov

    overfit_thresh = weights.get("overfit_penalty_threshold", 0.08)
    penalty = max(0.0, overfit_gap - overfit_thresh)
    composite = float(np.clip(base_score - penalty, 0.0, 1.0))

    breakdown = {
        "r2_val": r2,
        "mae_val": raw_mae,
        "rmse_val": raw_rmse,
        "spearman_rho": spearman,
        "base_score": round(base_score, 4),
        "overfit_gap": round(overfit_gap, 4),
        "penalty": round(penalty, 4),
        "composite_score": round(composite, 4),
    }

    if return_breakdown:
        return composite, overfit_gap, penalty, breakdown
    return composite, overfit_gap, penalty


def build_leaderboard(track: str = "C") -> pd.DataFrame:
    """Build, rank, gate, and persist full leaderboard for specified track."""
    metrics_path = get_artifacts_dir("metrics") / "all_splits_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}. Run training first.")

    df_metrics = pd.read_csv(metrics_path)
    df_track = df_metrics[df_metrics["track"] == track].copy()
    if df_track.empty:
        raise ValueError(f"No metric rows found for track {track}")

    cfg = load_weights_config()
    weights = cfg.get(f"{'classification' if track == 'C' else 'regression'}_weights", {})
    run_info = get_current_run_info()

    models = df_track["model"].unique()
    val_rows = df_track[df_track["split"] == "val"].set_index("model")
    train_rows = df_track[df_track["split"] == "train"].set_index("model")

    # Load thresholds if saved
    thresholds_path = get_artifacts_dir("leaderboard") / "thresholds.json"
    tuned_thresholds = {}
    if thresholds_path.exists():
        try:
            with open(thresholds_path, "r", encoding="utf-8") as tf:
                tuned_thresholds = json.load(tf)
        except Exception:
            pass

    records = []
    breakdown_records = []

    if track == "C":
        # Calculate validation baseline prevalence
        sample_val = val_rows.iloc[0]
        val_samples = float(sample_val.get("samples", 286))
        val_pos = float(sample_val.get("positive_count", 47))
        prevalence = val_pos / val_samples if val_samples > 0 else 0.1643

        briers = [
            float(val_rows.loc[m].get("brier_score", val_rows.loc[m].get("brier", 0.25)))
            for m in models if m in val_rows.index and not np.isnan(float(val_rows.loc[m].get("brier_score", 0.25)))
        ]
        eces = [
            float(val_rows.loc[m]["ece"])
            for m in models if m in val_rows.index and not np.isnan(float(val_rows.loc[m].get("ece", 0.15)))
        ]
        b_min, b_max = min(briers) if briers else 0.05, max(briers) if briers else 0.30
        e_min, e_max = min(eces) if eces else 0.01, max(eces) if eces else 0.30

        for m in models:
            if m not in val_rows.index:
                continue
            v_dict = val_rows.loc[m].to_dict()
            t_dict = train_rows.loc[m].to_dict() if m in train_rows.index else None

            # Hard Gate Audit
            is_qualified, status, reasons = check_hard_gates(v_dict, cfg, prevalence=prevalence)

            th_val = tuned_thresholds.get(m, v_dict.get("threshold", 0.50))
            if isinstance(th_val, dict):
                th_val = th_val.get("threshold", 0.50)

            raw_brier = float(v_dict.get("brier_score", v_dict.get("brier", 0.25)))
            raw_ece = float(v_dict.get("ece", 0.15))
            brier_baseline = prevalence * (1.0 - prevalence)
            bss = float(1.0 - (raw_brier / (brier_baseline + 1e-12)))

            try:
                comp, gap, pen, b_down = compute_classification_composite(
                    v_dict, t_dict, weights, b_min, b_max, e_min, e_max, return_breakdown=True
                )
            except Exception as e:
                logger.error(f"Error computing composite for {m}: {e}")
                comp, gap, pen = 0.0, 0.0, 0.0
                b_down = {}
                is_qualified = False
                status = "incomplete"
                reasons.append(str(e))

            rec = {
                "model": m,
                "track": "C",
                "category": CATEGORY_MAP.get(m, "Other"),
                "composite_score": round(comp, 4),
                "pr_auc": round(float(v_dict.get("pr_auc", 0)), 4),
                "roc_auc": round(float(v_dict.get("roc_auc", 0)), 4),
                "f1": round(float(v_dict.get("f1", 0)), 4),
                "balanced_accuracy": round(float(v_dict.get("balanced_accuracy", 0)), 4),
                "brier_score": round(raw_brier, 4),
                "brier": round(raw_brier, 4),  # alias for backwards compatibility
                "ece": round(raw_ece, 4),
                "threshold": round(float(th_val), 4),
                "brier_skill_score": round(bss, 4),
                "qualified": is_qualified,
                "status": status,
                "disqualification_reasons": reasons,
                "overfit_gap": round(gap, 4),
                "penalty": round(pen, 4),
                "split": "validation",
                "run_id": run_info["run_id"],
                "timestamp": run_info["timestamp"],
            }
            records.append(rec)

            # Store for breakdown CSV
            breakdown_entry = {"model": m, "category": CATEGORY_MAP.get(m, "Other"), "status": status, "qualified": is_qualified}
            breakdown_entry.update(b_down)
            breakdown_entry["disqualification_reasons"] = "; ".join(reasons) if reasons else "None"
            breakdown_records.append(breakdown_entry)

    else:
        # Track R
        maes = [float(val_rows.loc[m]["mae"]) for m in models if m in val_rows.index]
        rmses = [float(val_rows.loc[m]["rmse"]) for m in models if m in val_rows.index]
        m_min, m_max = min(maes) if maes else 1.0, max(maes) if maes else 50.0
        r_min, r_max = min(rmses) if rmses else 1.0, max(rmses) if rmses else 50.0

        for m in models:
            if m not in val_rows.index:
                continue
            v_dict = val_rows.loc[m].to_dict()
            t_dict = train_rows.loc[m].to_dict() if m in train_rows.index else None

            comp, gap, pen, b_down = compute_regression_composite(
                v_dict, t_dict, weights, m_min, m_max, r_min, r_max, return_breakdown=True
            )
            rec = {
                "model": m,
                "track": "R",
                "category": CATEGORY_MAP.get(m, "Other"),
                "composite_score": round(comp, 4),
                "r2": round(float(v_dict.get("r2", 0)), 4),
                "mae": round(float(v_dict.get("mae", 0)), 4),
                "rmse": round(float(v_dict.get("rmse", 0)), 4),
                "spearman_rho": round(float(v_dict.get("spearman_rho", 0)), 4),
                "overfit_gap": round(gap, 4),
                "penalty": round(pen, 4),
                "qualified": True,
                "status": "passed",
                "disqualification_reasons": [],
                "split": "validation",
                "run_id": run_info["run_id"],
                "timestamp": run_info["timestamp"],
            }
            records.append(rec)
            breakdown_entry = {"model": m, "category": CATEGORY_MAP.get(m, "Other"), "status": "passed", "qualified": True}
            breakdown_entry.update(b_down)
            breakdown_records.append(breakdown_entry)

    lb_df = pd.DataFrame(records)

    # Hard Gates Ranking Rule:
    # Qualified models rank first, sorted by composite_score descending.
    # Disqualified / Incomplete models rank strictly AFTER qualified models.
    if "qualified" in lb_df.columns:
        lb_df = lb_df.sort_values(
            by=["qualified", "composite_score"],
            ascending=[False, False],
        ).reset_index(drop=True)
    else:
        lb_df = lb_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    lb_df["rank"] = range(1, len(lb_df) + 1)

    # Persist
    lb_dir = get_artifacts_dir("leaderboard")
    lb_dir.mkdir(parents=True, exist_ok=True)
    csv_file = lb_dir / f"leaderboard_{track}.csv"
    json_file = lb_dir / f"leaderboard_{track}.json"

    lb_df.to_csv(csv_file, index=False)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(lb_df.to_dict(orient="records"), f, indent=2)

    # Persist detailed score breakdown CSV
    if breakdown_records:
        bd_df = pd.DataFrame(breakdown_records)
        # Match order of ranked leaderboard
        bd_df = bd_df.set_index("model").loc[lb_df["model"]].reset_index()
        bd_df["rank"] = lb_df["rank"].values
        bd_csv = lb_dir / "score_breakdown.csv"
        bd_df.to_csv(bd_csv, index=False)
        logger.info(f"Detailed score breakdown written -> {bd_csv}")

    logger.info(
        f"Leaderboard for Track {track} successfully built with {len(lb_df)} models "
        f"({sum(lb_df['qualified'])} qualified, {sum(~lb_df['qualified'])} disqualified)."
    )
    return lb_df
