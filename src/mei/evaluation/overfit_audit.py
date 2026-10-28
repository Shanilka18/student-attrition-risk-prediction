"""Overfitting Audit Module for Track A (Classification) and Track B (Regression).

Computes generalization gap, cross-validation standard deviation, seed standard
deviation, epochs-after-best (for deep models), and overfit status classification
for all candidate models across both tracks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from mei.config import ARTIFACTS_DIR, RUNS_DIR

logger = logging.getLogger(__name__)

# Design Palette from PROJECT_BRIEF.md Section 10
COLOR_GOOD = "#2E7D4F"  # Green
COLOR_SLIGHT = "#C98A00"  # Amber
COLOR_OVERFIT = "#B3261E"  # Red
COLOR_PRIMARY = "#1F4E46"  # Forest
COLOR_MUTED = "#5B6168"  # Muted grey
COLOR_SURFACE = "#FFFFFF"
COLOR_LINE = "#DDD8CE"


def determine_overfit_status(gap: float) -> str:
    """Classify generalization gap into standard audit status tiers.

    Tiers per PROJECT_BRIEF.md Section 7:
    - gap <= 0.05: 'Good fit'
    - 0.05 < gap <= 0.10: 'Slight overfit'
    - gap > 0.10: 'Overfit'
    """
    if gap <= 0.05:
        return "Good fit"
    elif gap <= 0.10:
        return "Slight overfit"
    else:
        return "Overfit"


def run_overfitting_audit(run_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Execute comprehensive overfitting audit across Track A and Track B.

    Reads leaderboards, per-model details, seed stability files, and deep learning
    training histories to compile a unified audit record.

    Args:
        run_dir: Directory of the active run. If None, resolves from active_run.txt.

    Returns:
        Structured audit dictionary containing Track A, Track B, and summary audits.
    """
    if run_dir is None:
        active_txt = ARTIFACTS_DIR / "active_run.txt"
        if not active_txt.exists():
            raise FileNotFoundError(f"Active run file not found at {active_txt}")
        run_id = active_txt.read_text(encoding="utf-8").strip()
        run_dir = RUNS_DIR / run_id

    run_dir = Path(run_dir)
    models_dir = run_dir / "models"
    logger.info(f"Running overfitting audit for run: {run_dir.name}")

    # 1. Load seed stability file if available
    seed_stability_file = run_dir / "seed_stability_deep.json"
    seed_stability: Dict[str, Any] = {}
    if seed_stability_file.exists():
        try:
            with open(seed_stability_file, "r", encoding="utf-8") as f:
                seed_stability = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load seed_stability_deep.json: {e}")

    # =========================================================================
    # TRACK A: CLASSIFICATION AUDIT
    # =========================================================================
    track_a_models: List[Dict[str, Any]] = []
    lb_cls_file = run_dir / "leaderboard_classification.json"
    if not lb_cls_file.exists():
        raise FileNotFoundError(f"Classification leaderboard not found: {lb_cls_file}")

    with open(lb_cls_file, "r", encoding="utf-8") as f:
        cls_data = json.load(f)

    for item in cls_data:
        mid = item["model_id"]
        train_f1 = float(item.get("train_f1_mean", 0.0))
        val_f1 = float(item.get("val_f1_optimal", item.get("val_f1", 0.0)))
        gap = float(item.get("generalization_gap", train_f1 - val_f1))
        status = determine_overfit_status(gap)

        # Retrieve CV std from detail.json fold_metrics if available
        cv_std: Optional[float] = None
        detail_file = models_dir / mid / "detail.json"
        if detail_file.exists():
            try:
                with open(detail_file, "r", encoding="utf-8") as df_in:
                    det = json.load(df_in)
                    fold_f1s = [
                        f["val_f1"]
                        for f in det.get("fold_metrics", [])
                        if "val_f1" in f
                    ]
                    if len(fold_f1s) > 1:
                        cv_std = float(np.std(fold_f1s, ddof=1))
            except Exception:
                pass

        # Retrieve seed std if available
        seed_std: Optional[float] = None
        if mid in seed_stability and "f1_std" in seed_stability[mid]:
            seed_std = float(seed_stability[mid]["f1_std"])

        # Retrieve epochs_after_best for deep models from curves.json
        epochs_after_best: Optional[int] = None
        curves_file = models_dir / mid / "curves.json"
        if curves_file.exists():
            try:
                with open(curves_file, "r", encoding="utf-8") as cf_in:
                    c_data = json.load(cf_in)
                    rep = c_data.get("representative_fold", {})
                    if isinstance(rep, dict):
                        b_epoch = rep.get("best_epoch")
                        hist = rep.get("history", {})
                        val_loss = hist.get("val_loss", [])
                        if b_epoch is not None and val_loss:
                            epochs_after_best = max(0, len(val_loss) - int(b_epoch))
            except Exception:
                pass

        record = {
            "model_id": mid,
            "display_name": item.get("display_name", mid),
            "plain_name": item.get("plain_name", mid),
            "category": item.get("category", "unknown"),
            "feature_set": item.get("feature_set", "unknown"),
            "rank": int(item.get("rank", 999)),
            "train_metric_name": "F1 (Train)",
            "train_metric": round(train_f1, 6),
            "val_metric_name": "F1 (Validation Optimal)",
            "val_metric": round(val_f1, 6),
            "generalization_gap": round(gap, 6),
            "cv_std": round(cv_std, 6) if cv_std is not None else None,
            "seed_std": round(seed_std, 6) if seed_std is not None else None,
            "epochs_after_best": epochs_after_best,
            "overfit_penalty": float(item.get("overfit_penalty", 0.0)),
            "leaderboard_score": float(item.get("leaderboard_score", 0.0)),
            "overfit_status": status,
        }
        track_a_models.append(record)

    # Sort Track A models by generalization gap ascending
    track_a_models.sort(key=lambda x: x["generalization_gap"])

    # =========================================================================
    # TRACK B: REGRESSION AUDIT
    # =========================================================================
    track_b_models: List[Dict[str, Any]] = []
    lb_reg_file = run_dir / "leaderboard_regression.json"
    if not lb_reg_file.exists():
        raise FileNotFoundError(f"Regression leaderboard not found: {lb_reg_file}")

    with open(lb_reg_file, "r", encoding="utf-8") as f:
        reg_data = json.load(f)

    for item in reg_data:
        mid = item["model_id"]
        train_r2 = float(item.get("train_r2_mean", 0.0))
        val_r2 = float(item.get("val_r2", 0.0))
        gap = float(item.get("generalization_gap", train_r2 - val_r2))
        status = determine_overfit_status(gap)
        cv_std = float(item.get("val_r2_std", 0.0)) if "val_r2_std" in item else None

        record = {
            "model_id": mid,
            "display_name": item.get("display_name", mid),
            "plain_name": item.get("plain_name", mid),
            "category": item.get("category", "unknown"),
            "feature_set": item.get("feature_set", "unknown"),
            "rank": int(item.get("rank", 999)),
            "train_metric_name": "R^2 (Train)",
            "train_metric": round(train_r2, 6),
            "val_metric_name": "R^2 (Validation)",
            "val_metric": round(val_r2, 6),
            "generalization_gap": round(gap, 6),
            "cv_std": round(cv_std, 6) if cv_std is not None else None,
            "seed_std": None,
            "epochs_after_best": None,
            "overfit_penalty": float(item.get("overfit_penalty", 0.0)),
            "leaderboard_score": float(item.get("leaderboard_score", 0.0)),
            "overfit_status": status,
        }
        track_b_models.append(record)

    # Sort Track B models by generalization gap ascending
    track_b_models.sort(key=lambda x: x["generalization_gap"])

    # Summary Statistics
    def compute_summary(models: List[Dict[str, Any]]) -> Dict[str, Any]:
        gaps = [m["generalization_gap"] for m in models]
        statuses = [m["overfit_status"] for m in models]
        return {
            "total_models": len(models),
            "good_fit_count": statuses.count("Good fit"),
            "slight_overfit_count": statuses.count("Slight overfit"),
            "overfit_count": statuses.count("Overfit"),
            "min_gap": round(float(np.min(gaps)), 6) if gaps else 0.0,
            "max_gap": round(float(np.max(gaps)), 6) if gaps else 0.0,
            "mean_gap": round(float(np.mean(gaps)), 6) if gaps else 0.0,
            "median_gap": round(float(np.median(gaps)), 6) if gaps else 0.0,
        }

    audit_result = {
        "run_id": run_dir.name,
        "track_a_classification": {
            "metric": "F1",
            "summary": compute_summary(track_a_models),
            "models": track_a_models,
        },
        "track_b_regression": {
            "metric": "R^2",
            "summary": compute_summary(track_b_models),
            "models": track_b_models,
        },
    }

    # Save to run artifacts and top-level artifacts
    out_run = run_dir / "overfit_audit.json"
    with open(out_run, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)

    out_top = ARTIFACTS_DIR / "overfit_audit.json"
    with open(out_top, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)

    logger.info(
        f"Overfitting audit complete. Saved to {out_run}. "
        f"Track A Good Fit: {audit_result['track_a_classification']['summary']['good_fit_count']}/{len(track_a_models)}, "
        f"Track B Good Fit: {audit_result['track_b_regression']['summary']['good_fit_count']}/{len(track_b_models)}"
    )

    return audit_result


def plot_generalization_gap_audit(
    audit_data: Dict[str, Any],
    output_dirs: List[Path],
) -> None:
    """Render publication figure showing train vs validation gap per model sorted by gap.

    Generates high-resolution PNG (300 DPI) and vector SVG across specified output dirs.
    """
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 12), sharey=False)
    fig.patch.set_facecolor(COLOR_SURFACE)

    tracks = [
        (
            "Track A: Binary Attrition Classification (Gap on F1)",
            audit_data["track_a_classification"],
            axes[0],
            "F1 Gap (Train - Val)",
        ),
        (
            "Track B: Continuous Intent Score Regression (Gap on R²)",
            audit_data["track_b_regression"],
            axes[1],
            "R² Gap (Train - Val)",
        ),
    ]

    for title, track_info, ax, xlabel in tracks:
        models = track_info["models"]
        names = [m["display_name"] for m in models]
        gaps = [m["generalization_gap"] for m in models]
        statuses = [m["overfit_status"] for m in models]

        colors = []
        for s in statuses:
            if s == "Good fit":
                colors.append(COLOR_GOOD)
            elif s == "Slight overfit":
                colors.append(COLOR_SLIGHT)
            else:
                colors.append(COLOR_OVERFIT)

        y_pos = np.arange(len(names))
        bars = ax.barh(
            y_pos, gaps, color=colors, height=0.65, edgecolor="none", alpha=0.90
        )

        # Add vertical reference lines for audit thresholds (0.05 and 0.10)
        ax.axvline(0.0, color="#1A1D21", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.axvline(
            0.05,
            color=COLOR_SLIGHT,
            linestyle="--",
            linewidth=1.2,
            label="Slight Overfit Threshold (0.05)",
        )
        ax.axvline(
            0.10,
            color=COLOR_OVERFIT,
            linestyle=":",
            linewidth=1.4,
            label="Overfit Threshold (0.10)",
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8.5, color="#1A1D21")
        ax.set_xlabel(xlabel, fontsize=10, fontweight="bold", color="#1A1D21")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=12, color=COLOR_PRIMARY)
        ax.grid(axis="x", linestyle="--", alpha=0.35, color=COLOR_LINE)
        ax.set_axisbelow(True)

        # Highlight value on bar end
        for bar, gap in zip(bars, gaps):
            text_x = gap + (0.005 if gap >= 0 else -0.015)
            ha = "left" if gap >= 0 else "right"
            ax.text(
                text_x,
                bar.get_y() + bar.get_height() / 2,
                f"{gap:+.3f}",
                va="center",
                ha=ha,
                fontsize=7.5,
                color="#1A1D21",
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLOR_LINE)
        ax.spines["bottom"].set_color(COLOR_LINE)

        # Add legend on first axis
        if ax == axes[0]:
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor=COLOR_GOOD, label="Good Fit (Gap ≤ 0.05)"),
                Patch(
                    facecolor=COLOR_SLIGHT, label="Slight Overfit (0.05 < Gap ≤ 0.10)"
                ),
                Patch(facecolor=COLOR_OVERFIT, label="Overfit (Gap > 0.10)"),
            ]
            ax.legend(
                handles=legend_elements,
                loc="lower right",
                framealpha=0.95,
                fontsize=8.5,
            )

    plt.tight_layout()

    # Save to all target paths
    filenames = [
        "fig_audit_01_generalization_gap.png",
        "fig_audit_01_generalization_gap.svg",
    ]
    for out_dir in output_dirs:
        for fname in filenames:
            out_path = out_dir / fname
            dpi = 300 if fname.endswith(".png") else None
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved audit figure: {out_path}")

    plt.close(fig)
