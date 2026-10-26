"""Global Feature Attribution and Aggregation Module.

Computes:
- Mean Absolute SHAP values across cohorts ($\frac{1}{N} \sum |\phi_j|$).
- Factor-level attribution aggregation (mapping 15 Likert items or 5 factor scores to domain constructs).
- Diagnostic visualizations: Global importance bar chart, Beeswarm distribution, and Feature Dependence curves.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.features.build_features import FACTOR_MAPPING
from src.utils.logger import get_logger

logger = get_logger("explain.global_explain")

SHAP_DIR = Path("artifacts/shap")
PLOTS_DIR = Path("artifacts/plots")


def compute_global_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Compute mean absolute SHAP feature importance.

    Args:
        shap_values: 2D array of attributions (N, p).
        feature_names: List of column names corresponding to p features.

    Returns:
        DataFrame containing 'feature', 'mean_abs_shap', 'std_shap', and 'relative_importance_pct'.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    std_shap = np.std(shap_values, axis=0)
    total = np.sum(mean_abs)
    rel_pct = (mean_abs / total * 100.0) if total > 0 else np.zeros_like(mean_abs)

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.round(mean_abs, 4),
        "std_shap": np.round(std_shap, 4),
        "relative_importance_pct": np.round(rel_pct, 2),
    })

    return df.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)


def aggregate_factor_importance(global_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item-level or factor-level attributions into the 5 psychosocial latent factors.

    Args:
        global_df: DataFrame from compute_global_importance.

    Returns:
        DataFrame with aggregated 'factor', 'mean_abs_shap', and 'percentage_contribution'.
    """
    factor_sums: Dict[str, float] = {
        "stress": 0.0,
        "motivation": 0.0,
        "belongingness": 0.0,
        "self_efficacy": 0.0,
        "peer_support": 0.0,
    }

    # Invert mapping: e.g. 'S1' -> 'stress', 'factor_stress' -> 'stress'
    for _, row in global_df.iterrows():
        feat = str(row["feature"]).lower()
        val = float(row["mean_abs_shap"])

        matched = False
        # Direct factor score match
        for factor_key in factor_sums.keys():
            if factor_key in feat:
                factor_sums[factor_key] += val
                matched = True
                break

        # Likert item match (S1, M2, etc.)
        if not matched:
            for factor_key, items in FACTOR_MAPPING.items():
                if feat.upper() in [it.upper() for it in items]:
                    factor_sums[factor_key] += val
                    matched = True
                    break

        if not matched:
            # Unmapped feature
            factor_sums[feat] = factor_sums.get(feat, 0.0) + val

    total = sum(factor_sums.values())
    records = []
    for f_name, f_val in factor_sums.items():
        pct = (f_val / total * 100.0) if total > 0 else 0.0
        records.append({
            "factor": f_name,
            "mean_abs_shap": round(f_val, 4),
            "percentage_contribution": round(pct, 2),
        })

    agg_df = pd.DataFrame(records).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    return agg_df


def plot_global_summary_bar(
    global_df: pd.DataFrame,
    track: str,
    model_name: str,
    output_dir: Union[Path, str] = SHAP_DIR,
) -> Path:
    """Generate and save global mean |SHAP| horizontal bar chart."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"global_bar_{track}_{model_name}.png"

    plt.figure(figsize=(9, 5))
    df_plot = global_df.sort_values(by="mean_abs_shap", ascending=True)

    colors = sns.color_palette("mako", len(df_plot))
    bars = plt.barh(df_plot["feature"], df_plot["mean_abs_shap"], color=colors, edgecolor="black", linewidth=0.6)

    # Annotate bars with values
    for bar, pct in zip(bars, df_plot["relative_importance_pct"]):
        w = bar.get_width()
        plt.text(w + 0.01 * max(df_plot["mean_abs_shap"]), bar.get_y() + bar.get_height()/2,
                 f"{w:.3f} ({pct:.1f}%)", va="center", ha="left", fontsize=9, fontweight="bold")

    plt.title(f"Global Feature Importance — Mean |SHAP| ({model_name.upper()} - {track.capitalize()})",
              fontsize=12, pad=12)
    plt.xlabel("Mean Absolute SHAP Value (Average Impact on Model Output Magnitude)", fontsize=10)
    plt.ylabel("Factor / Item", fontsize=10)
    plt.xlim(0, max(df_plot["mean_abs_shap"]) * 1.25)
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Also mirror into artifacts/plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.barh(df_plot["feature"], df_plot["mean_abs_shap"], color=colors, edgecolor="black", linewidth=0.6)
    for bar, pct in zip(bars, df_plot["relative_importance_pct"]):
        w = bar.get_width()
        plt.text(w + 0.01 * max(df_plot["mean_abs_shap"]), bar.get_y() + bar.get_height()/2,
                 f"{w:.3f} ({pct:.1f}%)", va="center", ha="left", fontsize=9, fontweight="bold")
    plt.title(f"Global Feature Importance ({model_name.upper()})", fontsize=12)
    plt.xlabel("Mean |SHAP|", fontsize=10)
    plt.xlim(0, max(df_plot["mean_abs_shap"]) * 1.25)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"global_shap_bar_{track}.png", dpi=200, bbox_inches="tight")
    plt.close()

    logger.info("Saved global summary bar chart to %s", save_path)
    return save_path


def plot_beeswarm(
    shap_values: np.ndarray,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str],
    track: str,
    model_name: str,
    output_dir: Union[Path, str] = SHAP_DIR,
) -> Path:
    """Generate and save SHAP Beeswarm distribution plot."""
    import shap
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"beeswarm_{track}_{model_name}.png"

    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_names)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        features=X_df,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
    )
    plt.title(f"SHAP Beeswarm Distribution — Impact on PRI Output ({model_name.upper()} - {track.capitalize()})",
              fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Mirror to plots dir
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        features=X_df,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
    )
    plt.title(f"SHAP Beeswarm Plot ({track.capitalize()})", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"beeswarm_{track}.png", dpi=200, bbox_inches="tight")
    plt.close()

    logger.info("Saved beeswarm plot to %s", save_path)
    return save_path


def plot_dependence(
    shap_values: np.ndarray,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str],
    target_feature: str,
    track: str,
    model_name: str,
    output_dir: Union[Path, str] = SHAP_DIR,
) -> Optional[Path]:
    """Generate and save feature dependence plot."""
    import shap
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if target_feature not in feature_names:
        return None

    save_path = out_dir / f"dependence_{target_feature}_{track}_{model_name}.png"
    feat_idx = feature_names.index(target_feature)
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_names)

    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        feat_idx,
        shap_values,
        X_df,
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"SHAP Dependence: {target_feature} ({model_name.upper()})", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info("Saved dependence plot to %s", save_path)
    return save_path
