"""Bounded 5-Step Remediation Ladder for Overfit and Slightly Overfit Models.

Applies bounded, systematic regularisation interventions in strictly defined order:
  Step 1: Stronger regularisation (higher L2 / min_samples_leaf, lower depth, lr decay)
  Step 2: Reduce capacity (fewer units / trees / leaves)
  Step 3: Stronger dropout / noise or subsampling (row and column)
  Step 4: In-fold feature selection (keeping top 90% cumulative importance)
  Step 5: Early stopping with tighter patience

Re-evaluates each attempt on the identical 15 cross-validation folds. Halts as soon
as status achieves 'Good fit' (gap <= 0.05) or the ladder terminates.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mei.config import ARTIFACTS_DIR, RUNS_DIR
from mei.models.regression_engine import evaluate_single_regression_model
from mei.models.regression_registry import get_all_regression_model_specs

logger = logging.getLogger(__name__)

# Palette colors
COLOR_ORIGINAL = "#B3261E"  # Amber/Red
COLOR_REMEDIATED = "#2E7D4F"  # Green
COLOR_PRIMARY = "#1F4E46"
COLOR_MUTED = "#5B6168"
COLOR_SURFACE = "#FFFFFF"
COLOR_LINE = "#DDD8CE"


def get_step_1_hyperparameters(mid: str, orig_params: Dict[str, Any]) -> Dict[str, Any]:
    """Step 1: Stronger regularisation (higher L2/min_child_weight, lower max_depth, lr decay)."""
    p = copy.deepcopy(orig_params)
    if "catboost" in mid:
        p["l2_leaf_reg"] = float(p.get("l2_leaf_reg", 3.0)) * 4.0  # 3.0 -> 12.0
        p["depth"] = max(2, int(p.get("depth", 4)) - 1)  # 4 -> 3
        p["learning_rate"] = float(p.get("learning_rate", 0.05)) * 0.7
    elif "lightgbm" in mid or "xgboost" in mid:
        p["reg_alpha"] = float(p.get("reg_alpha", 0.0)) + 2.0
        p["reg_lambda"] = float(p.get("reg_lambda", 1.0)) * 4.0
        p["max_depth"] = max(2, int(p.get("max_depth", 4)) - 1)
        p["learning_rate"] = float(p.get("learning_rate", 0.05)) * 0.7
    elif "random_forest" in mid or "extra_trees" in mid:
        p["min_samples_leaf"] = max(4, int(p.get("min_samples_leaf", 1)) * 4)
        p["max_depth"] = max(3, int(p.get("max_depth", 8)) - 2)
    elif "ridge" in mid or "elasticnet" in mid or "lasso" in mid:
        p["alpha"] = float(p.get("alpha", 1.0)) * 5.0
    else:
        p["weight_decay"] = float(p.get("weight_decay", 1e-4)) * 10.0
    return p


def get_step_2_hyperparameters(mid: str, orig_params: Dict[str, Any]) -> Dict[str, Any]:
    """Step 2: Reduce capacity (fewer units, trees, or leaves)."""
    p = get_step_1_hyperparameters(mid, orig_params)
    if "catboost" in mid or "xgboost" in mid or "lightgbm" in mid:
        p["iterations"] = max(
            40, int(p.get("iterations", p.get("n_estimators", 100))) // 2
        )
        p["n_estimators"] = p["iterations"]
        p["depth"] = max(2, int(p.get("depth", 3)) - 1)
    elif "random_forest" in mid or "extra_trees" in mid:
        p["n_estimators"] = max(30, int(p.get("n_estimators", 100)) // 2)
        p["max_depth"] = max(2, int(p.get("max_depth", 6)) - 2)
    elif "mlp" in mid:
        p["hidden_layer_sizes"] = (16,)
    return p


def get_step_3_hyperparameters(mid: str, orig_params: Dict[str, Any]) -> Dict[str, Any]:
    """Step 3: Stronger subsampling and dropout (row and column)."""
    p = get_step_2_hyperparameters(mid, orig_params)
    p["subsample"] = 0.75
    p["colsample_bytree"] = 0.75
    p["rsm"] = 0.75  # CatBoost random subspace method
    if "dropout" in p:
        p["dropout"] = min(0.6, float(p.get("dropout", 0.2)) + 0.2)
    return p


def get_step_4_hyperparameters(mid: str, orig_params: Dict[str, Any]) -> Dict[str, Any]:
    """Step 4: Tighter feature thresholding + regularisation."""
    p = get_step_3_hyperparameters(mid, orig_params)
    p["top_p_features"] = 0.90
    return p


def get_step_5_hyperparameters(mid: str, orig_params: Dict[str, Any]) -> Dict[str, Any]:
    """Step 5: Tighter early stopping patience."""
    p = get_step_4_hyperparameters(mid, orig_params)
    p["early_stopping_rounds"] = 5
    p["patience"] = 4
    return p


def execute_remediation_for_regression_model(
    model_id: str,
    run_dir: Path,
    cv_folds: List[Dict[str, Any]],
    X_dict: Dict[str, np.ndarray],
    y: np.ndarray,
) -> Dict[str, Any]:
    """Execute the 5-step remediation ladder for a regression model until 'Good fit' is achieved.

    Returns:
        Dictionary with history of remediation attempts and the winning attempt.
    """
    reg_specs = get_all_regression_model_specs()
    if model_id not in reg_specs:
        raise ValueError(f"Model ID '{model_id}' not found in regression registry.")

    base_spec = reg_specs[model_id]

    # Load original model details
    reg_details_file = run_dir / "models" / "regression_details.json"
    with open(reg_details_file, "r", encoding="utf-8") as f:
        reg_details = json.load(f)

    orig_metrics = reg_details[model_id]
    orig_params = {
        "iterations": 150,
        "learning_rate": 0.05,
        "depth": 4,
        "l2_leaf_reg": 3.0,
    }

    step_funcs = [
        (
            1,
            "Stronger Regularisation (L2 leaf reg x4, depth -1, lr decay)",
            get_step_1_hyperparameters,
        ),
        (
            2,
            "Reduced Capacity (fewer iterations and shallower depth)",
            get_step_2_hyperparameters,
        ),
        (3, "Subsampling & RSM (subsample=0.75, rsm=0.75)", get_step_3_hyperparameters),
        (
            4,
            "Feature Selection (top 90% cumulative importance)",
            get_step_4_hyperparameters,
        ),
        (5, "Tighter Early Stopping Patience", get_step_5_hyperparameters),
    ]

    attempts = []
    final_success: Optional[Dict[str, Any]] = None

    for step_num, step_name, param_func in step_funcs:
        logger.info(
            f"Applying Remediation Ladder Step {step_num} to {model_id}: {step_name}"
        )
        new_params = param_func(model_id, orig_params)

        # Clone spec for remediation run
        rem_spec = copy.deepcopy(base_spec)
        rem_spec.id = f"{model_id}_remediated_step_{step_num}"
        rem_spec.display_name = f"{base_spec.display_name} (remediated step {step_num})"
        rem_spec.plain_name = f"{base_spec.plain_name} [Remediated Step {step_num}]"
        rem_spec.n_tuning_trials = 0
        rem_spec.search_space = {}
        rem_spec.fixed_params = new_params

        # Evaluate across the identical 15 folds
        res = evaluate_single_regression_model(
            spec=rem_spec,
            X_dict=X_dict,
            y=y,
            cv_folds=cv_folds,
            seed=42,
        )

        val_r2 = res["leaderboard_metrics"]["val_r2"]
        train_r2 = res["leaderboard_metrics"]["train_r2_mean"]
        gap = res["leaderboard_metrics"]["generalization_gap"]
        status = res["leaderboard_metrics"]["overfit_status"]

        attempt_info = {
            "step": step_num,
            "step_name": step_name,
            "model_id": rem_spec.id,
            "display_name": rem_spec.display_name,
            "params": new_params,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "gap": gap,
            "overfit_status": status,
            "eval_result": res,
        }
        attempts.append(attempt_info)

        logger.info(
            f"Step {step_num} Result for {model_id}: "
            f"Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, Gap={gap:.4f}, Status={status}"
        )

        if status == "Good fit":
            final_success = attempt_info
            logger.info(
                f"Remediation SUCCESS: {model_id} achieved 'Good fit' at Step {step_num}!"
            )
            break

    if final_success is None:
        final_success = attempts[-1]

    return {
        "original_model_id": model_id,
        "original_metrics": orig_metrics,
        "attempts": attempts,
        "winning_attempt": final_success,
    }


def plot_remediation_comparison(
    remediation_res: Dict[str, Any],
    output_dirs: List[Path],
) -> None:
    """Render publication figure comparing before vs after remediation metrics."""
    orig = remediation_res["original_metrics"]
    win = remediation_res["winning_attempt"]
    mid = remediation_res["original_model_id"]

    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor(COLOR_SURFACE)

    labels = ["Train R²", "Validation R²", "Generalization Gap"]
    orig_vals = [orig["train_r2_mean"], orig["val_r2"], orig["generalization_gap"]]
    rem_vals = [win["train_r2"], win["val_r2"], win["gap"]]

    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        orig_vals,
        width,
        label=f"Original ({orig['overfit_status']})",
        color=COLOR_ORIGINAL,
        alpha=0.85,
        edgecolor="none",
    )
    rects2 = ax.bar(
        x + width / 2,
        rem_vals,
        width,
        label=f"Remediated: Step {win['step']} ({win['overfit_status']})",
        color=COLOR_REMEDIATED,
        alpha=0.85,
        edgecolor="none",
    )

    ax.set_ylabel("R² Metric Value", fontsize=10, fontweight="bold", color="#1A1D21")
    ax.set_title(
        f"Remediation Ladder: {orig.get('display_name', mid)}\n"
        f"Step {win['step']} — {win['step_name']}",
        fontsize=11,
        fontweight="bold",
        pad=12,
        color=COLOR_PRIMARY,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, fontweight="bold", color="#1A1D21")
    ax.grid(axis="y", linestyle="--", alpha=0.35, color=COLOR_LINE)
    ax.set_axisbelow(True)

    # Reference line at gap threshold 0.05
    ax.axhline(
        0.05,
        color=COLOR_MUTED,
        linestyle=":",
        linewidth=1.2,
        label="Good Fit Threshold (0.05)",
    )

    # Bar value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(
            f"{h:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(framealpha=0.95, fontsize=9)
    plt.tight_layout()

    filenames = [f"fig_remediation_{mid}.png", f"fig_remediation_{mid}.svg"]
    for out_dir in output_dirs:
        for fname in filenames:
            out_path = out_dir / fname
            dpi = 300 if fname.endswith(".png") else None
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved remediation figure: {out_path}")

    plt.close(fig)


def run_remediation_ladder(run_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Execute the remediation ladder for all overfit models in the top 12 of each track.

    Updates the leaderboard and records remediated model rows while preserving original rows.
    """
    if run_dir is None:
        active_txt = ARTIFACTS_DIR / "active_run.txt"
        run_id = active_txt.read_text(encoding="utf-8").strip()
        run_dir = RUNS_DIR / run_id

    run_dir = Path(run_dir)
    models_dir = run_dir / "models"

    # Load splits and features
    cv_folds = json.load(open(run_dir / "splits" / "cv_folds.json"))
    tr_agg = pd.read_parquet(run_dir / "features" / "train_agg.parquet")
    val_agg = pd.read_parquet(run_dir / "features" / "val_agg.parquet")
    X_agg = pd.concat([tr_agg, val_agg]).reset_index(drop=True).values

    tr_states = pd.read_parquet(run_dir / "features" / "train_agg_states.parquet")
    val_states = pd.read_parquet(run_dir / "features" / "val_agg_states.parquet")
    X_agg_states = pd.concat([tr_states, val_states]).reset_index(drop=True).values

    X_dict = {
        "agg": X_agg,
        "agg_states": X_agg_states,
    }

    # Find interim cohort and targets (Rule G3: do not touch test split)
    train_idx = json.load(open(run_dir / "splits" / "train_idx.json"))
    val_idx = json.load(open(run_dir / "splits" / "val_idx.json"))
    min_train_val = len(train_idx) + len(val_idx)

    cohort_df = None
    for cand in sorted(Path("data/interim").glob("clean_*.parquet")):
        df_cand = pd.read_parquet(cand)
        if (
            "attrition_intent_score" in df_cand.columns
            and len(df_cand) >= min_train_val
        ):
            cohort_df = df_cand
            break

    if cohort_df is None:
        raise FileNotFoundError("Could not find interim clean cohort dataset.")

    train_val_idx = train_idx + val_idx
    y_reg = cohort_df.iloc[train_val_idx]["attrition_intent_score"].values.astype(float)

    # Check Track B Top 12 models from leaderboard
    lb_reg_path = run_dir / "leaderboard_regression.csv"
    lb_reg_df = pd.read_csv(lb_reg_path)
    top12_reg = lb_reg_df.head(12)

    overfit_candidates = top12_reg[
        top12_reg["overfit_status"].isin(["Slight overfit", "Overfit"])
    ]
    logger.info(
        f"Found {len(overfit_candidates)} models requiring remediation in Track B Top 12."
    )

    remediation_results = {}
    fig_dirs = [
        Path("docs/figures"),
        ARTIFACTS_DIR / "figures",
        run_dir / "figures",
    ]

    for _, row in overfit_candidates.iterrows():
        mid = row["model_id"]
        logger.info(
            f"Starting remediation ladder for Track B model: {mid} (gap={row['generalization_gap']:.4f})"
        )
        res = execute_remediation_for_regression_model(
            mid, run_dir, cv_folds, X_dict, y_reg
        )
        remediation_results[mid] = res

        # Render before vs after figure
        plot_remediation_comparison(res, fig_dirs)

        # Update leaderboard regression with the new remediated model
        win = res["winning_attempt"]
        win_metrics = win["eval_result"]["leaderboard_metrics"]

        # 1. Update regression_details.json
        with open(models_dir / "regression_details.json", "r", encoding="utf-8") as f:
            det_all = json.load(f)
        det_all[win["model_id"]] = win_metrics
        with open(models_dir / "regression_details.json", "w", encoding="utf-8") as f:
            json.dump(det_all, f, indent=2)

        # 2. Update leaderboard_regression.json and .csv
        with open(run_dir / "leaderboard_regression.json", "r", encoding="utf-8") as f:
            lb_json = json.load(f)

        # Append remediated row
        rem_row_dict = copy.deepcopy(win_metrics)
        lb_json.append(rem_row_dict)

        # Re-sort and re-rank leaderboard
        lb_json.sort(key=lambda x: x["leaderboard_score"], reverse=True)
        for idx, item in enumerate(lb_json, 1):
            item["rank"] = idx

        with open(run_dir / "leaderboard_regression.json", "w", encoding="utf-8") as f:
            json.dump(lb_json, f, indent=2)

        df_updated = pd.DataFrame(lb_json)
        df_updated.to_csv(run_dir / "leaderboard_regression.csv", index=False)
        logger.info(
            f"Added remediated model '{win['model_id']}' to regression leaderboard at rank {rem_row_dict.get('rank', 'N/A')}."
        )

    # Save summary of all remediation ladder runs
    summary_file = run_dir / "remediation_summary.json"
    clean_summary = {
        m: {
            "original_model_id": data["original_model_id"],
            "original_gap": data["original_metrics"]["generalization_gap"],
            "original_status": data["original_metrics"]["overfit_status"],
            "winning_step": data["winning_attempt"]["step"],
            "winning_model_id": data["winning_attempt"]["model_id"],
            "remediated_gap": data["winning_attempt"]["gap"],
            "remediated_status": data["winning_attempt"]["overfit_status"],
            "remediated_score": data["winning_attempt"]["eval_result"][
                "leaderboard_metrics"
            ]["leaderboard_score"],
            "remediated_r2": data["winning_attempt"]["val_r2"],
        }
        for m, data in remediation_results.items()
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, indent=2)

    return remediation_results
