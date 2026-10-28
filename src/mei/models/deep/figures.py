"""Publication-grade visualization renderers for Deep Learning architectures in brand palette."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Canonical Brand Palette (PROJECT_BRIEF.md §10)
PALETTE = {
    "paper": "#F5F2EC",
    "surface": "#FFFFFF",
    "ink": "#1A1D21",
    "muted": "#5B6168",
    "line": "#DDD8CE",
    "primary_forest": "#1F4E46",
    "info_blue": "#2B5C8A",
    "warning_amber": "#C98A00",
    "risk_low": "#2E7D4F",
    "risk_moderate": "#C98A00",
    "risk_high": "#B3261E",
    "extra": [
        "#8C4A3B",
        "#6E7F3F",
        "#7A6F63",
        "#4A6B82",
        "#A65B32",
        "#5E8C61",
        "#B36B00",
        "#7B3F61",
    ],
}


def _save_fig(fig: plt.Figure, base_name: str, out_dirs: List[Path]) -> None:
    """Save matplotlib figure to 300 DPI PNG and vector SVG across directories."""
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            d / f"{base_name}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        fig.savefig(
            d / f"{base_name}.svg",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


def plot_deep_twin_curves(
    model_id: str,
    display_name: str,
    rep_history: Dict[str, Any],
    fig_num: int,
    out_dirs: List[Path],
) -> None:
    """Render research-grade Twin Panel (Loss & Accuracy) training curves.

    Features:
      - Left: Loss (Train: forest green, Val: warning amber)
      - Right: Accuracy (Train: forest green, Val: warning amber)
      - Vertical dashed line at Stage 2 start ('Start Refine')
      - Vertical dotted line at Best Validation Epoch
      - Shaded Overfitting Zone where val_loss diverges
      - Research caption line: 'Best validation loss X at epoch N (stage), val acc Y'
    """
    hist = rep_history["history"]
    best_epoch = rep_history["best_epoch"]
    best_val_loss = rep_history["best_val_loss"]
    stage_2_start = rep_history.get("stage_2_start_epoch", len(hist["train_loss"]))
    stage_2_used = rep_history.get("stage_2_used", False)
    overfit_zone = rep_history.get("overfitting_zone", [])

    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    val_acc_at_best = (
        hist["val_acc"][best_epoch - 1]
        if best_epoch <= len(hist["val_acc"])
        else hist["val_acc"][-1]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=PALETTE["paper"])

    # 1. Panel 1: Loss
    ax1.set_facecolor(PALETTE["surface"])
    ax1.plot(
        epochs,
        hist["train_loss"],
        color=PALETTE["primary_forest"],
        linewidth=2.2,
        label="Train Loss",
    )
    ax1.plot(
        epochs,
        hist["val_loss"],
        color=PALETTE["warning_amber"],
        linewidth=2.2,
        label="Validation Loss",
    )

    # Markers
    if stage_2_start < len(epochs):
        ax1.axvline(
            stage_2_start,
            color=PALETTE["info_blue"],
            linestyle="--",
            linewidth=1.8,
            label=f"Stage 2 Refine (Ep {stage_2_start})",
        )
    ax1.axvline(
        best_epoch,
        color=PALETTE["risk_high"],
        linestyle=":",
        linewidth=2.0,
        label=f"Best Model (Ep {best_epoch})",
    )

    # Shaded Overfitting Zone
    if overfit_zone:
        min_oz = min(overfit_zone)
        max_oz = max(overfit_zone)
        ax1.axvspan(
            min_oz,
            max_oz,
            color=PALETTE["risk_high"],
            alpha=0.12,
            label="Overfitting Zone (>3% Divergence)",
        )

    ax1.set_title(
        "Cross-Entropy Loss Progression",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax1.set_xlabel("Epoch", fontsize=11, color=PALETTE["ink"])
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=11, color=PALETTE["ink"])
    ax1.grid(True, linestyle="--", alpha=0.4, color=PALETTE["line"])
    ax1.legend(
        loc="upper right",
        frameon=True,
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["line"],
        fontsize=9,
    )
    for spine in ax1.spines.values():
        spine.set_color(PALETTE["line"])

    # 2. Panel 2: Accuracy
    ax2.set_facecolor(PALETTE["surface"])
    ax2.plot(
        epochs,
        hist["train_acc"],
        color=PALETTE["primary_forest"],
        linewidth=2.2,
        label="Train Accuracy",
    )
    ax2.plot(
        epochs,
        hist["val_acc"],
        color=PALETTE["warning_amber"],
        linewidth=2.2,
        label="Validation Accuracy",
    )

    if stage_2_start < len(epochs):
        ax2.axvline(
            stage_2_start,
            color=PALETTE["info_blue"],
            linestyle="--",
            linewidth=1.8,
            label=f"Stage 2 Refine (Ep {stage_2_start})",
        )
    ax2.axvline(
        best_epoch,
        color=PALETTE["risk_high"],
        linestyle=":",
        linewidth=2.0,
        label=f"Best Model (Ep {best_epoch})",
    )

    if overfit_zone:
        min_oz = min(overfit_zone)
        max_oz = max(overfit_zone)
        ax2.axvspan(
            min_oz,
            max_oz,
            color=PALETTE["risk_high"],
            alpha=0.12,
            label="Overfitting Zone",
        )

    ax2.set_title(
        "Classification Accuracy Progression",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax2.set_xlabel("Epoch", fontsize=11, color=PALETTE["ink"])
    ax2.set_ylabel("Accuracy", fontsize=11, color=PALETTE["ink"])
    ax2.grid(True, linestyle="--", alpha=0.4, color=PALETTE["line"])
    ax2.legend(
        loc="lower right",
        frameon=True,
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["line"],
        fontsize=9,
    )
    for spine in ax2.spines.values():
        spine.set_color(PALETTE["line"])

    # Figure Header & Academic Subtitle
    stage_str = "Stage 2 Refined" if stage_2_used else "Stage 1 Peak"
    caption_text = (
        f"Model: {display_name} | Best validation loss: {best_val_loss:.4f} at epoch {best_epoch} ({stage_str}), "
        f"val acc: {val_acc_at_best:.1%}"
    )
    fig.suptitle(
        f"FIG-DEEP-0{fig_num}: Learning Lifecycle — {display_name}",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["ink"],
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        caption_text,
        ha="center",
        fontsize=10,
        style="italic",
        color=PALETTE["muted"],
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    base_name = f"deep_0{fig_num}_{model_id}_curves"
    _save_fig(fig, base_name, out_dirs)


def plot_all_deep_validation_loss_comparison(
    models_data: List[Dict[str, Any]],
    out_dirs: List[Path],
) -> None:
    """Render comparison overlay of all deep models' validation loss curves in one chart."""
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=PALETTE["paper"])
    ax.set_facecolor(PALETTE["surface"])

    colors = PALETTE["extra"]

    for i, m in enumerate(models_data):
        d_name = m["spec"]["display_name"]
        hist = m["representative_fold"]["history"]
        val_loss = hist["val_loss"]
        epochs = np.arange(1, len(val_loss) + 1)
        c = colors[i % len(colors)]

        # Highlight best epoch
        best_ep = m["representative_fold"]["best_epoch"]
        best_loss = m["representative_fold"]["best_val_loss"]

        ax.plot(
            epochs,
            val_loss,
            color=c,
            linewidth=2.0,
            label=f"{d_name} (min={best_loss:.3f})",
        )
        if best_ep <= len(val_loss):
            ax.scatter(
                [best_ep],
                [val_loss[best_ep - 1]],
                color=c,
                s=50,
                edgecolors=PALETTE["ink"],
                zorder=5,
            )

    ax.set_title(
        "FIG-DEEP-09: Cross-Architecture Validation Loss Convergence",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.set_xlabel("Epoch", fontsize=11, color=PALETTE["ink"])
    ax.set_ylabel(
        "Validation Loss (Binary Cross-Entropy)", fontsize=11, color=PALETTE["ink"]
    )
    ax.grid(True, linestyle="--", alpha=0.4, color=PALETTE["line"])
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["line"],
        fontsize=9,
    )
    for spine in ax.spines.values():
        spine.set_color(PALETTE["line"])

    fig.text(
        0.5,
        0.01,
        "Dots indicate optimal early-stopping restoration checkpoints for each architecture.",
        ha="center",
        fontsize=10,
        style="italic",
        color=PALETTE["muted"],
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    _save_fig(fig, "deep_09_all_models_loss_comparison", out_dirs)


def plot_overfit_demonstration(
    reg_history: Dict[str, Any],
    overfit_history: Dict[str, Any],
    champion_name: str,
    out_dirs: List[Path],
) -> None:
    """Render the centerpiece viva figure: 'Before vs After Regularization'.

    Left: Deliberately over-parameterized model (512-256-128, no L2, no dropout, 300 epochs)
          showing catastrophic U-shaped validation divergence.
    Right: Regularized two-stage deep model (dropout, L2 decay, Gaussian noise, early stopping).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=PALETTE["paper"])

    # Panel 1: Unregularized Over-Parameterized Model
    ax1.set_facecolor(PALETTE["surface"])
    o_hist = overfit_history["history"]
    o_epochs = np.arange(1, len(o_hist["train_loss"]) + 1)

    ax1.plot(
        o_epochs,
        o_hist["train_loss"],
        color=PALETTE["primary_forest"],
        linewidth=2.0,
        label="Train Loss (→ 0)",
    )
    ax1.plot(
        o_epochs,
        o_hist["val_loss"],
        color=PALETTE["risk_high"],
        linewidth=2.2,
        label="Val Loss (Severe Divergence)",
    )

    best_o_ep = overfit_history["best_epoch"]
    ax1.axvline(
        best_o_ep,
        color=PALETTE["ink"],
        linestyle=":",
        linewidth=1.8,
        label=f"Optimal Epoch ({best_o_ep})",
    )
    ax1.axvspan(
        best_o_ep,
        len(o_epochs),
        color=PALETTE["risk_high"],
        alpha=0.15,
        label="Catastrophic Memorization",
    )

    ax1.set_title(
        "BEFORE Regularization (Unconstrained MLP 512-256-128)",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["risk_high"],
    )
    ax1.set_xlabel("Epoch (Full 300 Epochs)", fontsize=11, color=PALETTE["ink"])
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=11, color=PALETTE["ink"])
    ax1.set_ylim(0, max(2.5, np.percentile(o_hist["val_loss"], 98)))
    ax1.grid(True, linestyle="--", alpha=0.4, color=PALETTE["line"])
    ax1.legend(
        loc="upper right",
        frameon=True,
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["line"],
        fontsize=8.5,
    )
    for spine in ax1.spines.values():
        spine.set_color(PALETTE["line"])

    # Panel 2: Regularized Two-Stage Architecture
    ax2.set_facecolor(PALETTE["surface"])
    r_hist = reg_history["history"]
    r_epochs = np.arange(1, len(r_hist["train_loss"]) + 1)
    best_r_ep = reg_history["best_epoch"]

    ax2.plot(
        r_epochs,
        r_hist["train_loss"],
        color=PALETTE["primary_forest"],
        linewidth=2.0,
        label="Train Loss",
    )
    ax2.plot(
        r_epochs,
        r_hist["val_loss"],
        color=PALETTE["warning_amber"],
        linewidth=2.2,
        label="Val Loss (Controlled)",
    )
    ax2.axvline(
        best_r_ep,
        color=PALETTE["risk_high"],
        linestyle=":",
        linewidth=2.0,
        label=f"Restored Weights (Ep {best_r_ep})",
    )

    stage_2_start = reg_history.get("stage_2_start_epoch", len(r_epochs))
    if stage_2_start < len(r_epochs):
        ax2.axvline(
            stage_2_start,
            color=PALETTE["info_blue"],
            linestyle="--",
            linewidth=1.8,
            label=f"Stage 2 (Ep {stage_2_start})",
        )

    ax2.set_title(
        f"AFTER Regularization ({champion_name})",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["primary_forest"],
    )
    ax2.set_xlabel("Epoch", fontsize=11, color=PALETTE["ink"])
    ax2.set_ylabel("Binary Cross-Entropy Loss", fontsize=11, color=PALETTE["ink"])
    ax2.grid(True, linestyle="--", alpha=0.4, color=PALETTE["line"])
    ax2.legend(
        loc="upper right",
        frameon=True,
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["line"],
        fontsize=8.5,
    )
    for spine in ax2.spines.values():
        spine.set_color(PALETTE["line"])

    fig.suptitle(
        "FIG-DEEP-10: Overfitting Demonstration — Before vs After Regularization",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["ink"],
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Left: Zero L2 / Zero Dropout memorizes survey samples. Right: Two-stage regularization prevents generalization collapse.",
        ha="center",
        fontsize=10,
        style="italic",
        color=PALETTE["muted"],
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    _save_fig(fig, "deep_10_overfit_demonstration", out_dirs)


def plot_deep_calibration_curves(
    models_data: List[Dict[str, Any]],
    out_dirs: List[Path],
) -> None:
    """Render reliability diagram and calibration curves for deep models."""
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=PALETTE["paper"])
    ax.set_facecolor(PALETTE["surface"])

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=PALETTE["muted"],
        label="Perfect Calibration (Identity)",
    )

    colors = PALETTE["extra"]
    for i, m in enumerate(models_data):
        d_name = m["spec"]["display_name"]
        calib = m["curves"]["calibration"]
        p_pred = calib["prob_pred"]
        p_true = calib["prob_true"]
        ece = m["leaderboard_metrics"]["val_ece"]
        c = colors[i % len(colors)]

        ax.plot(
            p_pred,
            p_true,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=c,
            label=f"{d_name} (ECE={ece:.3f})",
        )

    ax.set_title(
        "FIG-DEEP-11: Deep Learning Reliability Calibration Diagrams",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.set_xlabel("Mean Predicted Probability", fontsize=11, color=PALETTE["ink"])
    ax.set_ylabel("Empirical At-Risk Fraction", fontsize=11, color=PALETTE["ink"])
    ax.grid(True, linestyle="--", alpha=0.4, color=PALETTE["line"])
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["line"],
        fontsize=8.5,
    )
    for spine in ax.spines.values():
        spine.set_color(PALETTE["line"])

    plt.tight_layout()
    _save_fig(fig, "deep_11_deep_calibration_curves", out_dirs)


def plot_cross_fold_variance(
    champion_model_data: Dict[str, Any],
    seed_summary: Dict[str, Any],
    out_dirs: List[Path],
) -> None:
    """Render cross-fold mean ± 1 std learning curves and seed stability."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=PALETTE["paper"])

    cf = champion_model_data["cross_fold_curves"]
    epochs = np.array(cf["epochs"])
    m_loss_tr = np.array(cf["mean"]["train_loss"])
    s_loss_tr = np.array(cf["std"]["train_loss"])
    m_loss_v = np.array(cf["mean"]["val_loss"])
    s_loss_v = np.array(cf["std"]["val_loss"])

    # Panel 1: Cross-Fold Loss (Mean ± 1 SD)
    ax1.set_facecolor(PALETTE["surface"])
    ax1.plot(
        epochs,
        m_loss_tr,
        color=PALETTE["primary_forest"],
        linewidth=2.0,
        label="Train Loss (Mean)",
    )
    ax1.fill_between(
        epochs,
        m_loss_tr - s_loss_tr,
        m_loss_tr + s_loss_tr,
        color=PALETTE["primary_forest"],
        alpha=0.15,
    )

    ax1.plot(
        epochs,
        m_loss_v,
        color=PALETTE["warning_amber"],
        linewidth=2.0,
        label="Val Loss (Mean)",
    )
    ax1.fill_between(
        epochs,
        m_loss_v - s_loss_v,
        m_loss_v + s_loss_v,
        color=PALETTE["warning_amber"],
        alpha=0.15,
    )

    ax1.set_title(
        f"15-Fold Cross-Validation Stability ({champion_model_data['spec']['display_name']})",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax1.set_xlabel("Epoch", fontsize=11, color=PALETTE["ink"])
    ax1.set_ylabel("Loss (Mean ± 1 SD across folds)", fontsize=11, color=PALETTE["ink"])
    ax1.grid(True, linestyle="--", alpha=0.4, color=PALETTE["line"])
    ax1.legend(
        loc="upper right",
        frameon=True,
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["line"],
    )
    for spine in ax1.spines.values():
        spine.set_color(PALETTE["line"])

    # Panel 2: Seed Variance Bar Chart
    ax2.set_facecolor(PALETTE["surface"])
    seeds = seed_summary.get("seeds", [42, 100, 2024, 777, 999])
    f1s = seed_summary.get("seed_f1s", [0.65] * len(seeds))
    aucs = seed_summary.get("seed_aucs", [0.85] * len(seeds))

    x = np.arange(len(seeds))
    width = 0.35

    ax2.bar(
        x - width / 2,
        f1s,
        width,
        label=f"Val F1 (Mean={seed_summary.get('f1_mean', 0.0):.3f} ± {seed_summary.get('f1_std', 0.0):.3f})",
        color=PALETTE["info_blue"],
    )
    ax2.bar(
        x + width / 2,
        aucs,
        width,
        label=f"Val AUC (Mean={seed_summary.get('auc_mean', 0.0):.3f} ± {seed_summary.get('auc_std', 0.0):.3f})",
        color=PALETTE["primary_forest"],
    )

    ax2.set_title(
        "5-Seed Initialization Sensitivity Audit",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Seed {s}" for s in seeds])
    ax2.set_ylabel("Validation Metric Value", fontsize=11, color=PALETTE["ink"])
    ax2.set_ylim(0.4, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.4, color=PALETTE["line"])
    ax2.legend(
        loc="lower right",
        frameon=True,
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["line"],
        fontsize=9,
    )
    for spine in ax2.spines.values():
        spine.set_color(PALETTE["line"])

    fig.suptitle(
        "FIG-DEEP-12: Variance and Random Seed Sensitivity Analysis",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["ink"],
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    _save_fig(fig, "deep_12_cross_fold_variance", out_dirs)
