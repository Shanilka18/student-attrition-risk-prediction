"""Feature Diagnostics, Multicollinearity Audits, Leakage Verification, and Figures."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from mei.features.pipeline import build_feature_pipeline, get_canonical_28_columns
from mei.io.schema import load_labels, load_schema

logger = logging.getLogger(__name__)

# MEI Brand Palette (§10 of PROJECT_BRIEF.md)
PALETTE = {
    "paper": "#F5F2EC",
    "surface": "#FFFFFF",
    "ink": "#1A1D21",
    "muted": "#5B6168",
    "line": "#DDD8CE",
    "forest": "#1F4E46",
    "blue": "#2B5C8A",
    "amber": "#C98A00",
    "risk_low": "#2E7D4F",
    "risk_mod": "#C98A00",
    "risk_high": "#B3261E",
    "cat1": "#8C4A3B",
    "cat2": "#6E7F3F",
    "cat3": "#7A6F63",
}


def apply_figure_style() -> None:
    """Apply consistent styling across all project figures."""
    plt.rcParams.update(
        {
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
            "font.family": "sans-serif",
            "figure.facecolor": PALETTE["surface"],
            "axes.facecolor": PALETTE["surface"],
            "axes.edgecolor": PALETTE["line"],
            "axes.labelcolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "grid.color": PALETTE["line"],
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
        }
    )


def audit_leakage(
    df: pd.DataFrame,
    feature_names: List[str],
    label_col: str = "at_risk",
    threshold: float = 0.95,
) -> Dict[str, Any]:
    """Strictly audit feature set for target and metadata leakage (Rule G3).

    Checks:
      1. Outcome items (A1-A3, attrition_intent_score, at_risk) are strictly absent.
      2. Metadata items (ID, timestamp, consent, trap) are strictly absent.
      3. No single feature exhibits correlation |r| > threshold with the label.
    """
    schema = load_schema()
    labels_cfg = load_labels()

    meta_col_names = {
        col.name
        for col in schema.columns
        if col.role in ["id", "meta", "consent", "attention_check"]
    }
    forbidden_cols = (
        set(labels_cfg.outcome_items.keys())
        | {"attrition_intent_score", "at_risk"}
        | meta_col_names
        | {"consent", "attention_check", "student_id", "timestamp"}
    )

    found_forbidden = [c for c in feature_names if c in forbidden_cols]

    # Correlation check with label
    corrs: Dict[str, float] = {}
    high_corr_features: List[Tuple[str, float]] = []

    if label_col in df.columns:
        for col in feature_names:
            if col in df.columns:
                series = df[col].dropna()
                common_idx = series.index.intersection(df[label_col].dropna().index)
                if len(common_idx) > 10:
                    r = np.corrcoef(
                        df.loc[common_idx, col], df.loc[common_idx, label_col]
                    )[0, 1]
                    r_val = 0.0 if np.isnan(r) else float(r)
                    corrs[col] = r_val
                    if abs(r_val) > threshold:
                        high_corr_features.append((col, r_val))

    passed = (len(found_forbidden) == 0) and (len(high_corr_features) == 0)

    max_corr_feature = None
    max_corr_val = 0.0
    if corrs:
        max_corr_feature = max(corrs.keys(), key=lambda k: abs(corrs[k]))
        max_corr_val = corrs[max_corr_feature]

    return {
        "status": "PASS" if passed else "FAIL",
        "forbidden_columns_found": found_forbidden,
        "high_correlation_features": high_corr_features,
        "max_correlation_feature": max_corr_feature,
        "max_correlation_value": round(max_corr_val, 4),
        "total_features_audited": len(feature_names),
        "threshold": threshold,
    }


def compute_feature_diagnostics(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "at_risk",
    seed: int = 42,
) -> pd.DataFrame:
    """Compute per-feature class distributions, Cohen's d, and Mutual Information."""
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' missing from dataframe.")

    y = df[label_col].values
    mask_0 = y == 0
    mask_1 = y == 1

    records: List[Dict[str, Any]] = []

    # Mutual information
    valid_features = [c for c in feature_cols if c in df.columns]
    X_mat = df[valid_features].fillna(df[valid_features].median()).values
    mi_scores = mutual_info_classif(X_mat, y, random_state=seed)
    mi_dict = {col: score for col, score in zip(valid_features, mi_scores)}

    for col in valid_features:
        vals_0 = df.loc[mask_0, col].dropna().values
        vals_1 = df.loc[mask_1, col].dropna().values

        mean_0 = float(np.mean(vals_0)) if len(vals_0) > 0 else 0.0
        std_0 = float(np.std(vals_0, ddof=1)) if len(vals_0) > 1 else 0.0
        mean_1 = float(np.mean(vals_1)) if len(vals_1) > 0 else 0.0
        std_1 = float(np.std(vals_1, ddof=1)) if len(vals_1) > 1 else 0.0

        # Cohen's d: (mean_1 - mean_0) / s_pooled
        n0 = len(vals_0)
        n1 = len(vals_1)
        if n0 > 1 and n1 > 1:
            s_pooled = np.sqrt(
                ((n0 - 1) * (std_0**2) + (n1 - 1) * (std_1**2)) / (n0 + n1 - 2)
            )
            d = (mean_1 - mean_0) / s_pooled if s_pooled > 1e-9 else 0.0
        else:
            d = 0.0

        # Spearman correlation
        spearman_r = df[col].corr(df[label_col], method="spearman")

        records.append(
            {
                "feature": col,
                "mean_not_at_risk": round(mean_0, 3),
                "std_not_at_risk": round(std_0, 3),
                "mean_at_risk": round(mean_1, 3),
                "std_at_risk": round(std_1, 3),
                "cohens_d": round(float(d), 3),
                "abs_cohens_d": round(float(abs(d)), 3),
                "mutual_info": round(float(mi_dict.get(col, 0.0)), 4),
                "spearman_corr": round(float(spearman_r), 3)
                if not np.isnan(spearman_r)
                else 0.0,
            }
        )

    diag_df = pd.DataFrame(records)
    return diag_df.sort_values(by="mutual_info", ascending=False).reset_index(drop=True)


def compute_vif_report(
    feature_df: pd.DataFrame,
    features_to_check: Optional[List[str]] = None,
    max_cols: int = 35,
) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) report for linear models."""
    cols = features_to_check or list(feature_df.columns)
    valid_cols = [c for c in cols if c in feature_df.columns][:max_cols]

    # Impute and clean
    sub_df = feature_df[valid_cols].fillna(feature_df[valid_cols].median()).copy()

    # Drop zero variance columns
    non_const = sub_df.columns[sub_df.std() > 1e-6].tolist()
    sub_clean = sub_df[non_const].values

    # Standardize to avoid numerical overflow in matrix inversion
    scaler = StandardScaler()
    scaled_mat = scaler.fit_transform(sub_clean)

    vif_records = []
    for i, col_name in enumerate(non_const):
        try:
            v = variance_inflation_factor(scaled_mat, i)
            vif_val = round(float(v), 2)
        except Exception:
            vif_val = np.nan
        vif_records.append({"feature": col_name, "VIF": vif_val})

    vif_df = pd.DataFrame(vif_records)
    return vif_df.sort_values(by="VIF", ascending=False).reset_index(drop=True)


def run_cv_benchmark(
    train_val_df: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    label_col: str = "at_risk",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run 15-fold CV benchmark on raw28 vs agg features using LR and LightGBM."""
    y = train_val_df[label_col].values.astype(int)
    raw_28_cols = get_canonical_28_columns()

    results: Dict[str, Dict[str, List[float]]] = {
        "lr_raw28": {"auc": [], "f1": []},
        "lr_agg": {"auc": [], "f1": []},
        "lgbm_raw28": {"auc": [], "f1": []},
        "lgbm_agg": {"auc": [], "f1": []},
        "lr_agg_interp": {
            "auc": [],
            "f1": [],
        },  # Test student interpolation alternative
    }

    logger.info(f"Running CV benchmark across {len(folds)} folds...")

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        tr_df = train_val_df.iloc[tr_idx]
        va_df = train_val_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # 1. Pipeline: raw28 (median)
        pipe_raw = build_feature_pipeline(
            feature_set="raw28", scale=True, imputer_method="median"
        )
        X_tr_raw = pipe_raw.fit_transform(tr_df[raw_28_cols])
        X_va_raw = pipe_raw.transform(va_df[raw_28_cols])

        # 2. Pipeline: agg (median)
        pipe_agg = build_feature_pipeline(
            feature_set="agg", scale=True, imputer_method="median"
        )
        X_tr_agg = pipe_agg.fit_transform(tr_df[raw_28_cols])
        X_va_agg = pipe_agg.transform(va_df[raw_28_cols])

        # 3. Pipeline: agg (student interpolation)
        pipe_interp = build_feature_pipeline(
            feature_set="agg", scale=True, imputer_method="student_interpolation"
        )
        X_tr_interp = pipe_interp.fit_transform(tr_df[raw_28_cols])
        X_va_interp = pipe_interp.transform(va_df[raw_28_cols])

        # Unscaled versions for LightGBM
        pipe_raw_unscaled = build_feature_pipeline(
            feature_set="raw28", scale=False, imputer_method="median"
        )
        X_tr_raw_u = pipe_raw_unscaled.fit_transform(tr_df[raw_28_cols])
        X_va_raw_u = pipe_raw_unscaled.transform(va_df[raw_28_cols])

        pipe_agg_unscaled = build_feature_pipeline(
            feature_set="agg", scale=False, imputer_method="median"
        )
        X_tr_agg_u = pipe_agg_unscaled.fit_transform(tr_df[raw_28_cols])
        X_va_agg_u = pipe_agg_unscaled.transform(va_df[raw_28_cols])

        # Model A: LR on raw28
        lr_raw = LogisticRegression(
            class_weight="balanced", random_state=seed, max_iter=500
        )
        lr_raw.fit(X_tr_raw, y_tr)
        p_raw = lr_raw.predict_proba(X_va_raw)[:, 1]
        results["lr_raw28"]["auc"].append(roc_auc_score(y_va, p_raw))
        results["lr_raw28"]["f1"].append(f1_score(y_va, (p_raw >= 0.5).astype(int)))

        # Model B: LR on agg
        lr_agg = LogisticRegression(
            class_weight="balanced", random_state=seed, max_iter=500
        )
        lr_agg.fit(X_tr_agg, y_tr)
        p_agg = lr_agg.predict_proba(X_va_agg)[:, 1]
        results["lr_agg"]["auc"].append(roc_auc_score(y_va, p_agg))
        results["lr_agg"]["f1"].append(f1_score(y_va, (p_agg >= 0.5).astype(int)))

        # Model C: LR on agg with student interpolation
        lr_interp = LogisticRegression(
            class_weight="balanced", random_state=seed, max_iter=500
        )
        lr_interp.fit(X_tr_interp, y_tr)
        p_interp = lr_interp.predict_proba(X_va_interp)[:, 1]
        results["lr_agg_interp"]["auc"].append(roc_auc_score(y_va, p_interp))
        results["lr_agg_interp"]["f1"].append(
            f1_score(y_va, (p_interp >= 0.5).astype(int))
        )

        # Model D: LightGBM on raw28
        lgb_raw = LGBMClassifier(
            class_weight="balanced",
            random_state=seed,
            n_estimators=100,
            learning_rate=0.05,
            verbose=-1,
        )
        lgb_raw.fit(X_tr_raw_u, y_tr)
        p_lgb_raw = lgb_raw.predict_proba(X_va_raw_u)[:, 1]
        results["lgbm_raw28"]["auc"].append(roc_auc_score(y_va, p_lgb_raw))
        results["lgbm_raw28"]["f1"].append(
            f1_score(y_va, (p_lgb_raw >= 0.5).astype(int))
        )

        # Model E: LightGBM on agg
        lgb_agg = LGBMClassifier(
            class_weight="balanced",
            random_state=seed,
            n_estimators=100,
            learning_rate=0.05,
            verbose=-1,
        )
        lgb_agg.fit(X_tr_agg_u, y_tr)
        p_lgb_agg = lgb_agg.predict_proba(X_va_agg_u)[:, 1]
        results["lgbm_agg"]["auc"].append(roc_auc_score(y_va, p_lgb_agg))
        results["lgbm_agg"]["f1"].append(f1_score(y_va, (p_lgb_agg >= 0.5).astype(int)))

    # Compute summary statistics
    summary: Dict[str, Any] = {}
    n_folds = len(folds)
    for model_name, metrics in results.items():
        auc_arr = np.array(metrics["auc"])
        f1_arr = np.array(metrics["f1"])

        auc_mean = float(np.mean(auc_arr))
        auc_std = float(np.std(auc_arr, ddof=1))
        auc_ci = 1.96 * auc_std / np.sqrt(n_folds)

        f1_mean = float(np.mean(f1_arr))
        f1_std = float(np.std(f1_arr, ddof=1))
        f1_ci = 1.96 * f1_std / np.sqrt(n_folds)

        summary[model_name] = {
            "auc_mean": round(auc_mean, 4),
            "auc_std": round(auc_std, 4),
            "auc_ci95": round(auc_ci, 4),
            "f1_mean": round(f1_mean, 4),
            "f1_std": round(f1_std, 4),
            "f1_ci95": round(f1_ci, 4),
            "auc_folds": [round(float(v), 4) for v in auc_arr],
            "f1_folds": [round(float(v), 4) for v in f1_arr],
        }

    return summary


def render_feature_diagnostic_figures(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    benchmark_summary: Dict[str, Any],
    figures_dir: Path,
) -> List[str]:
    """Generate 4 publication-quality figures in the MEI brand palette (PNG + SVG)."""
    apply_figure_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[str] = []

    # -------------------------------------------------------------
    # Figure 1: Distributions of Top Engineered Features by At-Risk
    # -------------------------------------------------------------
    top_dyn_features = [
        "composite_slope",
        "composite_max_drop",
        "composite_last_value",
        "composite_mean",
        "resource_access_slope",
        "study_consistency_slope",
    ]
    present_top = [c for c in top_dyn_features if c in feature_df.columns]
    if present_top and "at_risk" in df.columns:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
        axes_flat = axes.flatten()

        for idx, feat in enumerate(present_top[:6]):
            ax = axes_flat[idx]
            plot_df = pd.DataFrame(
                {feat: feature_df[feat].values, "at_risk": df["at_risk"].values}
            )
            sns.boxplot(
                x="at_risk",
                y=feat,
                data=plot_df,
                ax=ax,
                hue="at_risk",
                palette=[PALETTE["risk_low"], PALETTE["risk_high"]],
                legend=False,
                width=0.45,
                fliersize=2,
            )
            ax.set_title(feat.replace("_", " ").title(), fontsize=11, fontweight="bold")
            ax.set_xlabel("At-Risk Status (0 = Not At-Risk, 1 = At-Risk)", fontsize=9)
            ax.set_ylabel("Feature Value", fontsize=9)
            ax.grid(True, axis="y")

        fig.suptitle(
            "Longitudinal Dynamics by At-Risk Status (Training Cohort)",
            fontsize=14,
            fontweight="bold",
            color=PALETTE["forest"],
            y=1.02,
        )
        plt.tight_layout()

        p1_png = figures_dir / "feat_01_feature_distributions_by_risk.png"
        p1_svg = figures_dir / "feat_01_feature_distributions_by_risk.svg"
        fig.savefig(p1_png, bbox_inches="tight")
        fig.savefig(p1_svg, bbox_inches="tight")
        plt.close(fig)
        generated_files.extend([str(p1_png), str(p1_svg)])

    # -------------------------------------------------------------
    # Figure 2: Top 15 Features by Cohen's d and Mutual Information
    # -------------------------------------------------------------
    top15 = diagnostics_df.head(15).copy().sort_values(by="mutual_info", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=300)

    # Subplot A: Mutual Information
    y_pos = np.arange(len(top15))
    ax1.barh(y_pos, top15["mutual_info"], color=PALETTE["forest"], height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top15["feature"], fontsize=9)
    ax1.set_xlabel("Mutual Information Score", fontsize=10, fontweight="bold")
    ax1.set_title(
        "Feature Relevance (Mutual Information)", fontsize=11, fontweight="bold"
    )
    ax1.grid(True, axis="x")

    # Subplot B: Cohen's d effect size
    colors_d = [
        PALETTE["risk_high"] if d < 0 else PALETTE["risk_low"]
        for d in top15["cohens_d"]
    ]
    ax2.barh(y_pos, top15["cohens_d"], color=colors_d, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  # Share left labels
    ax2.set_xlabel("Cohen's d Effect Size", fontsize=10, fontweight="bold")
    ax2.set_title(
        "Standardized Effect Size (At-Risk vs Not At-Risk)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.axvline(0, color=PALETTE["ink"], linestyle="-", linewidth=0.8)
    ax2.grid(True, axis="x")

    fig.suptitle(
        "Top 15 Engineered Features: Relevance and Separation",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["forest"],
        y=1.02,
    )
    plt.tight_layout()

    p2_png = figures_dir / "feat_02_effect_sizes_and_mi.png"
    p2_svg = figures_dir / "feat_02_effect_sizes_and_mi.svg"
    fig.savefig(p2_png, bbox_inches="tight")
    fig.savefig(p2_svg, bbox_inches="tight")
    plt.close(fig)
    generated_files.extend([str(p2_png), str(p2_svg)])

    # -------------------------------------------------------------
    # Figure 3: Spearman Correlation Heatmap of Composite & Dynamics
    # -------------------------------------------------------------
    heat_cols = [
        "composite_p1",
        "composite_p2",
        "composite_p3",
        "composite_p4",
        "composite_slope",
        "composite_max_drop",
        "composite_last_value",
        "composite_auc",
        "resource_access_slope",
        "study_time_slope",
        "quiz_practice_slope",
        "study_consistency_slope",
    ]
    present_heat = [c for c in heat_cols if c in feature_df.columns]
    if len(present_heat) >= 6:
        corr_mat = feature_df[present_heat].corr(method="spearman")

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        cmap = sns.diverging_palette(145, 10, s=85, l=45, n=7, as_cmap=True)
        sns.heatmap(
            corr_mat,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            vmin=-1.0,
            vmax=1.0,
            cbar_kws={"label": "Spearman Correlation (rho)"},
            linewidths=0.5,
            linecolor=PALETTE["line"],
        )
        ax.set_title(
            "Spearman Correlation Matrix: Longitudinal Dynamic Features",
            fontsize=12,
            fontweight="bold",
            color=PALETTE["forest"],
            pad=12,
        )
        plt.tight_layout()

        p3_png = figures_dir / "feat_03_spearman_correlation_heatmap.png"
        p3_svg = figures_dir / "feat_03_spearman_correlation_heatmap.svg"
        fig.savefig(p3_png, bbox_inches="tight")
        fig.savefig(p3_svg, bbox_inches="tight")
        plt.close(fig)
        generated_files.extend([str(p3_png), str(p3_svg)])

    # -------------------------------------------------------------
    # Figure 4: CV Benchmark (Raw28 vs Agg) with 95% CI Error Bars
    # -------------------------------------------------------------
    models_to_plot = [
        ("lr_raw28", "Logistic Reg (Raw 28)"),
        ("lr_agg", "Logistic Reg (Aggregated)"),
        ("lgbm_raw28", "LightGBM (Raw 28)"),
        ("lgbm_agg", "LightGBM (Aggregated)"),
    ]

    labels = [m[1] for m in models_to_plot]
    auc_means = [benchmark_summary[m[0]]["auc_mean"] for m in models_to_plot]
    auc_errors = [benchmark_summary[m[0]]["auc_ci95"] for m in models_to_plot]
    f1_means = [benchmark_summary[m[0]]["f1_mean"] for m in models_to_plot]
    f1_errors = [benchmark_summary[m[0]]["f1_ci95"] for m in models_to_plot]

    fig, (ax_auc, ax_f1) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    x = np.arange(len(labels))
    width = 0.45

    ax_auc.bar(
        x,
        auc_means,
        width,
        yerr=auc_errors,
        capsize=5,
        color=[PALETTE["muted"], PALETTE["forest"], PALETTE["blue"], PALETTE["amber"]],
        edgecolor=PALETTE["line"],
    )
    ax_auc.set_ylabel("AUC-ROC (Mean ± 95% CI)", fontsize=10, fontweight="bold")
    ax_auc.set_title("15-Fold Repeated CV: AUC-ROC", fontsize=11, fontweight="bold")
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax_auc.set_ylim(0.80, 0.95)
    ax_auc.grid(True, axis="y")

    ax_f1.bar(
        x,
        f1_means,
        width,
        yerr=f1_errors,
        capsize=5,
        color=[PALETTE["muted"], PALETTE["forest"], PALETTE["blue"], PALETTE["amber"]],
        edgecolor=PALETTE["line"],
    )
    ax_f1.set_ylabel("F1-Score At-Risk (Mean ± 95% CI)", fontsize=10, fontweight="bold")
    ax_f1.set_title("15-Fold Repeated CV: F1-Score", fontsize=11, fontweight="bold")
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax_f1.set_ylim(0.60, 0.85)
    ax_f1.grid(True, axis="y")

    fig.suptitle(
        "Cross-Validation Benchmark: Value of Longitudinal Feature Engineering",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["forest"],
        y=1.03,
    )
    plt.tight_layout()

    p4_png = figures_dir / "feat_04_cv_benchmark_raw_vs_agg.png"
    p4_svg = figures_dir / "feat_04_cv_benchmark_raw_vs_agg.svg"
    fig.savefig(p4_png, bbox_inches="tight")
    fig.savefig(p4_svg, bbox_inches="tight")
    plt.close(fig)
    generated_files.extend([str(p4_png), str(p4_svg)])

    return generated_files
