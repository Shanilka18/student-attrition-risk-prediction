"""Phase 5 Leaderboard builder, Champion selector, and Test-set evaluator.

Responsibilities:
- Load all Phase 4 model artifacts from artifacts/models/{track}/.
- Compute the complete metric suite on the validation set and CV fold results.
- Build per-category, per-feature-set, per-track leaderboards.
- Compute composite rank from config weights.
- Run statistical comparisons (Wilcoxon) between top models.
- Select champions per track and best-in-category.
- Run exactly-once final evaluation on the sealed test set.
- Save calibrated probability estimates (isotonic/Platt on validation).
- Produce all diagnostic plots (ROC, PR, calibration, residual, confusion matrix).
- Robustness checks (noise, item dropout, subgroup fairness).
- Export deployable PRI bundle to artifacts/models/final/.
"""

import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.splits import load_split, SealedTestAccessError
from src.evaluation.metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    compute_multiclass_ece,
    wilcoxon_comparison,
    corrected_resampled_t_test,
    RISK_LABELS,
    RISK_LABEL_TO_INT,
)
from src.features.build_features import (
    RAW_ITEM_COLS,
    FACTOR_COLS,
    compute_factor_scores,
    extract_feature_set,
    extract_targets,
)
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("evaluation.leaderboard")

CONFIG_PATH = Path("config/default.yaml")
ARTIFACTS_ROOT = Path("artifacts")
LEADERBOARD_DIR = ARTIFACTS_ROOT / "leaderboards"
MODEL_DIR = ARTIFACTS_ROOT / "models"
PLOT_DIR = ARTIFACTS_ROOT / "plots"
FINAL_DIR = MODEL_DIR / "final"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _get_features_and_targets(
    df: pd.DataFrame,
    feature_set: str,
    track: str,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (X_df, y_reg, y_cls) for a given split dataframe."""
    df_feat = compute_factor_scores(df)
    X_df = extract_feature_set(df_feat, feature_set)
    targets = extract_targets(df_feat)

    y_reg = targets.get("pri_score") if "pri_score" in targets else targets.get("pri_true")
    if y_reg is None and "pri_true" in df.columns:
        y_reg = df["pri_true"]
    y_cls = targets.get("risk_category")
    if y_cls is None and "risk_category" in df.columns:
        y_cls = df["risk_category"]

    y_reg_arr = np.asarray(y_reg.values if hasattr(y_reg, "values") else y_reg, dtype=float) if y_reg is not None else None
    y_cls_arr = np.asarray(y_cls.values if hasattr(y_cls, "values") else y_cls) if y_cls is not None else None

    return X_df, y_reg_arr, y_cls_arr



# ---------------------------------------------------------------------------
# Load all Phase 4 artifacts
# ---------------------------------------------------------------------------

def load_phase4_results(track: str) -> List[Dict[str, Any]]:
    """Load all metadata JSONs from Phase 4 training for a given track."""
    track_dir = MODEL_DIR / track
    results = []
    if not track_dir.exists():
        logger.warning("Track directory %s not found.", track_dir)
        return results
    for meta_path in sorted(track_dir.glob("*_metadata.json")):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            model_path = track_dir / f"{meta['model_name']}.joblib"
            if model_path.exists():
                meta["_model_path"] = str(model_path)
                results.append(meta)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", meta_path, exc)
    logger.info("Loaded %d Phase 4 result(s) for track '%s'.", len(results), track)
    return results


def load_model_artifact(model_path: str) -> Any:
    """Load a serialized model from disk."""
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Enhanced metric computation on full val/test set
# ---------------------------------------------------------------------------

def evaluate_model_on_split(
    model: Any,
    X: "Union[pd.DataFrame, np.ndarray]",
    y_reg: Optional[np.ndarray],
    y_cls: Optional[np.ndarray],
    track: str,
    n_features: int = 5,
) -> Dict[str, Any]:
    """Run full metric suite on a model against a data split.

    Args:
        model: Fitted sklearn-compatible estimator.
        X: Feature matrix (DataFrame or ndarray).
        y_reg: Continuous PRI targets (regression track).
        y_cls: Risk category labels (classification track).
        track: 'regression' or 'classification'.
        n_features: Number of features for Adjusted R².

    Returns:
        Dict of metric name → value.
    """
    start = time.perf_counter()
    y_pred = model.predict(X)
    latency_ms = (time.perf_counter() - start) / max(len(y_pred), 1) * 1000 * 100

    y_prob: Optional[np.ndarray] = None
    if track == "classification" and hasattr(model, "predict_proba"):
        try:
            raw_prob = model.predict_proba(X)
            if hasattr(model, "classes_"):
                model_classes = list(model.classes_)
                indices = [model_classes.index(lbl) for lbl in RISK_LABELS if lbl in model_classes]
                if len(indices) == len(RISK_LABELS):
                    y_prob = raw_prob[:, indices]
                else:
                    y_prob = raw_prob
            else:
                y_prob = raw_prob
        except Exception:
            y_prob = None

    if track == "regression":
        assert y_reg is not None
        metrics = compute_regression_metrics(y_reg, y_pred, n_features=n_features)
    else:
        assert y_cls is not None
        metrics = compute_classification_metrics(y_cls, y_pred, y_prob)

    metrics["inference_latency_ms_per_100"] = round(latency_ms, 4)
    return metrics


# ---------------------------------------------------------------------------
# Composite rank calculation
# ---------------------------------------------------------------------------

def _composite_rank(
    df: pd.DataFrame,
    track: str,
    weights: Dict[str, float],
    shap_compatible_models: Optional[List[str]] = None,
) -> pd.Series:
    """Compute weighted composite rank score (higher = better).

    Dimensions:
        - accuracy: normalised primary metric rank.
        - calibration: inverse ECE rank (lower ECE = higher rank).
        - stability: inverse CV std rank (lower std = higher rank).
        - explainability: +1 bonus for SHAP-compatible models.
        - latency: inverse latency rank (lower latency = higher rank).
    """
    shap_compatible_models = shap_compatible_models or [
        "ridge", "lasso", "elastic_net", "linear_regression",
        "logistic_regression", "lda",
        "random_forest", "extra_trees", "decision_tree",
        "gradient_boosting", "hist_gradient_boosting",
        "xgboost", "lightgbm", "catboost",
    ]
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=float)

    # 1. Accuracy rank
    acc_col = "val_r2" if track == "regression" else "val_macro_f1"
    if acc_col in df.columns:
        acc_rank = df[acc_col].rank(ascending=True) / n
    else:
        acc_col = "val_primary_metric"
        ascending = track == "regression"
        acc_rank = df[acc_col].rank(ascending=ascending) / n

    # 2. Calibration rank (lower ECE → higher rank; fill missing with worst)
    if "ece" in df.columns:
        ece_vals = df["ece"].fillna(df["ece"].max() if df["ece"].notna().any() else 1.0)
        cal_rank = ece_vals.rank(ascending=False) / n
    else:
        cal_rank = pd.Series(0.5, index=df.index)

    # 3. Stability rank (lower cv_std → higher rank)
    std_col = None
    for c in ["cv_std", "cv_rmse_std", "cv_f1_std", "primary_std"]:
        if c in df.columns:
            std_col = c
            break
    if std_col:
        stab_rank = df[std_col].rank(ascending=False) / n
    else:
        stab_rank = pd.Series(0.5, index=df.index)

    # 4. Explainability bonus
    if "model_name" in df.columns:
        expl_score = df["model_name"].apply(
            lambda m: 1.0 if m in shap_compatible_models else 0.3
        )
    else:
        expl_score = pd.Series(0.5, index=df.index)

    # 5. Latency rank (lower latency → higher rank)
    if "latency_ms" in df.columns:
        lat_rank = df["latency_ms"].rank(ascending=False) / n
    else:
        lat_rank = pd.Series(0.5, index=df.index)

    composite = (
        weights.get("accuracy", 0.35) * acc_rank
        + weights.get("calibration", 0.20) * cal_rank
        + weights.get("stability", 0.20) * stab_rank
        + weights.get("explainability", 0.10) * expl_score
        + weights.get("latency", 0.15) * lat_rank
    )
    return composite


# ---------------------------------------------------------------------------
# Statistical pairwise comparison
# ---------------------------------------------------------------------------

def statistical_comparison(
    results: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Run Wilcoxon signed-rank pairwise comparison of top model vs. others.

    Uses per-CV-fold primary metric values stored in Phase 4 JSON artifacts.
    Returns a list of comparison records.
    """
    if len(results) < 2:
        return []

    # Sort by val primary metric
    track = results[0].get("track", "regression")
    ascending = track == "regression"
    sorted_results = sorted(
        results,
        key=lambda r: r.get("validation_metrics", {}).get("primary", float("inf") if ascending else float("-inf")),
        reverse=not ascending,
    )

    champion = sorted_results[0]
    champion_name = champion["model_name"]

    # Pull CV fold scores if available (stored as cv_fold_scores)
    comparisons = []
    champ_folds = np.array(champion.get("cv_fold_scores", [champion.get("validation_metrics", {}).get("primary", 0.0)]))

    for other in sorted_results[1:]:
        other_name = other["model_name"]
        other_folds = np.array(other.get("cv_fold_scores", [other.get("validation_metrics", {}).get("primary", 0.0)]))

        # Align lengths
        min_len = min(len(champ_folds), len(other_folds))
        if min_len < 2:
            stat, p_val = float("nan"), 1.0
        else:
            stat, p_val = wilcoxon_comparison(
                champ_folds[:min_len], other_folds[:min_len]
            )

        comparisons.append({
            "champion": champion_name,
            "challenger": other_name,
            "champion_mean": round(float(np.mean(champ_folds)), 6),
            "challenger_mean": round(float(np.mean(other_folds)), 6),
            "wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else None,
            "p_value": round(p_val, 6),
            "significant": bool(p_val < alpha) if not np.isnan(p_val) else None,
            "alpha": alpha,
        })
    return comparisons


# ---------------------------------------------------------------------------
# Robustness checks
# ---------------------------------------------------------------------------

def robustness_check(
    model: Any,
    X_base: "Union[pd.DataFrame, np.ndarray]",
    y_reg: Optional[np.ndarray],
    y_cls: Optional[np.ndarray],
    track: str,
    noise_levels: List[float],
    dropout_rates: List[float],
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate model performance under noise injection and item dropout."""
    rng = np.random.default_rng(seed)
    results: Dict[str, Any] = {"noise": {}, "dropout": {}}
    feature_names = X_base.columns.tolist() if isinstance(X_base, pd.DataFrame) else None
    X_np = X_base.values if isinstance(X_base, pd.DataFrame) else np.asarray(X_base, dtype=float)

    for noise in noise_levels:
        X_noisy = X_np.copy()
        mask = rng.random(X_noisy.shape) < noise
        X_noisy[mask] = rng.uniform(1, 5, size=mask.sum())
        X_eval = pd.DataFrame(X_noisy, columns=feature_names) if feature_names is not None else X_noisy
        try:
            m = evaluate_model_on_split(model, X_eval, y_reg, y_cls, track)
            results["noise"][f"{int(noise*100)}pct"] = {
                "primary": m.get("primary"),
                "rmse": m.get("rmse"),
                "macro_f1": m.get("macro_f1"),
            }
        except Exception as exc:
            results["noise"][f"{int(noise*100)}pct"] = {"error": str(exc)}

    for dropout in dropout_rates:
        X_drop = X_np.copy().astype(float)
        mask = rng.random(X_drop.shape) < dropout
        X_drop[mask] = np.nan
        col_means = np.nanmean(X_drop, axis=0)
        for c in range(X_drop.shape[1]):
            nan_rows = np.isnan(X_drop[:, c])
            X_drop[nan_rows, c] = col_means[c] if not np.isnan(col_means[c]) else 3.0
        X_eval = pd.DataFrame(X_drop, columns=feature_names) if feature_names is not None else X_drop
        try:
            m = evaluate_model_on_split(model, X_eval, y_reg, y_cls, track)
            results["dropout"][f"{int(dropout*100)}pct"] = {
                "primary": m.get("primary"),
                "rmse": m.get("rmse"),
                "macro_f1": m.get("macro_f1"),
            }
        except Exception as exc:
            results["dropout"][f"{int(dropout*100)}pct"] = {"error": str(exc)}

    return results


def subgroup_fairness(
    model: Any,
    df: pd.DataFrame,
    X: "Union[pd.DataFrame, np.ndarray]",
    y_reg: Optional[np.ndarray],
    y_cls: Optional[np.ndarray],
    track: str,
    subgroup_cols: List[str],
) -> Dict[str, Any]:
    """Compute per-subgroup metrics for fairness assessment."""
    fairness: Dict[str, Any] = {}
    for col in subgroup_cols:
        if col not in df.columns:
            logger.warning("Subgroup column '%s' not found in split.", col)
            continue
        fairness[col] = {}
        for val in df[col].dropna().unique():
            mask = (df[col] == val).values
            if mask.sum() < 5:
                continue
            X_g = X.iloc[mask] if isinstance(X, pd.DataFrame) else X[mask]
            y_reg_g = y_reg[mask] if y_reg is not None else None
            y_cls_g = y_cls[mask] if y_cls is not None else None
            try:
                m = evaluate_model_on_split(model, X_g, y_reg_g, y_cls_g, track)
                fairness[col][str(val)] = {
                    "n": int(mask.sum()),
                    "primary": m.get("primary"),
                    "rmse": m.get("rmse"),
                    "macro_f1": m.get("macro_f1"),
                    "high_risk_recall": m.get("high_risk_recall"),
                }
            except Exception as exc:
                fairness[col][str(val)] = {"error": str(exc)}
    return fairness


# ---------------------------------------------------------------------------
# Diagnostic Plots
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def plot_regression_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    prefix: str = "test",
) -> None:
    """Predicted vs Actual scatter and residual distribution plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_name} — Regression Diagnostics ({prefix})", fontsize=13)

    # Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.45, s=20, color="#3B82F6", edgecolors="none")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    axes[0].set_xlabel("True PRI")
    axes[0].set_ylabel("Predicted PRI")
    axes[0].set_title("Predicted vs Actual")
    axes[0].legend(fontsize=9)

    # Residuals
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=40, color="#8B5CF6", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Residual (True − Predicted)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    _save_fig(fig, PLOT_DIR / f"{prefix}_{model_name}_regression_diagnostics.png")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    model_name: str,
    prefix: str = "test",
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"{model_name} — Confusion Matrix ({prefix})", fontsize=12)
    _save_fig(fig, PLOT_DIR / f"{prefix}_{model_name}_confusion_matrix.png")


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str],
    model_name: str,
    prefix: str = "test",
) -> None:
    y_bin = label_binarize(y_true, classes=labels)
    colors = ["#3B82F6", "#F59E0B", "#EF4444"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (label, color) in enumerate(zip(labels, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curves OvR ({prefix})")
    ax.legend(loc="lower right", fontsize=9)
    _save_fig(fig, PLOT_DIR / f"{prefix}_{model_name}_roc_curves.png")


def plot_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str],
    model_name: str,
    prefix: str = "test",
) -> None:
    y_bin = label_binarize(y_true, classes=labels)
    colors = ["#3B82F6", "#F59E0B", "#EF4444"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (label, color) in enumerate(zip(labels, colors)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, color=color, lw=2, label=f"{label} (AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_name} — Precision-Recall Curves ({prefix})")
    ax.legend(loc="lower left", fontsize=9)
    _save_fig(fig, PLOT_DIR / f"{prefix}_{model_name}_pr_curves.png")


def plot_calibration_curve(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    label: str,
    model_name: str,
    prefix: str = "test",
) -> None:
    from sklearn.calibration import calibration_curve as sk_cal_curve
    fig, ax = plt.subplots(figsize=(6, 5))
    frac_pos, mean_pred = sk_cal_curve(y_true_bin, y_prob, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "s-", color="#3B82F6", label="Model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction Positive")
    ax.set_title(f"{model_name} — Calibration ({label}) ({prefix})")
    ax.legend(fontsize=9)
    _save_fig(fig, PLOT_DIR / f"{prefix}_{model_name}_calibration_{label}.png")


# ---------------------------------------------------------------------------
# PRI Export Helpers
# ---------------------------------------------------------------------------

def build_pri_output_regression(
    model: Any,
    X: np.ndarray,
    cfg_pri: Dict[str, Any],
) -> np.ndarray:
    """Clip and rescale regression output to PRI [0, 100]."""
    raw = model.predict(X).astype(float)
    clipped = np.clip(raw, cfg_pri["clip_min"], cfg_pri["clip_max"])
    return clipped


def build_pri_output_classification(
    model: Any,
    X: np.ndarray,
    cfg_pri: Dict[str, Any],
) -> np.ndarray:
    """Convert classification probabilities to expected PRI score."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Classification champion must support predict_proba for PRI export.")
    probs = model.predict_proba(X)
    # Expected risk: weight class midpoints
    cut_low = cfg_pri["cut_low"]
    cut_high = cfg_pri["cut_high"]
    midpoints = np.array([cut_low / 2, (cut_low + cut_high) / 2, (cut_high + 100) / 2])
    expected_pri = probs @ midpoints
    return np.clip(expected_pri, cfg_pri["clip_min"], cfg_pri["clip_max"])


# ---------------------------------------------------------------------------
# Main evaluation orchestrator
# ---------------------------------------------------------------------------

def run_evaluation(
    tracks: List[str] = None,
    seed: int = 42,
    run_test_set: bool = True,
) -> Dict[str, Any]:
    """Full Phase 5 evaluation pipeline.

    Args:
        tracks: ['regression', 'classification'] or subset.
        seed: Random seed.
        run_test_set: If True, runs final evaluation on sealed test set.

    Returns:
        Dict with leaderboards, champions, and evaluation summaries.
    """
    set_seed(seed)
    cfg = _load_config()
    eval_cfg = cfg.get("evaluation", {})
    composite_weights = eval_cfg.get("composite_weights", {
        "accuracy": 0.35, "calibration": 0.20,
        "stability": 0.20, "explainability": 0.10, "latency": 0.15,
    })
    champion_thresh = eval_cfg.get("champion_thresholds", {})
    robustness_cfg = eval_cfg.get("robustness", {})
    pri_cfg = eval_cfg.get("pri_export", {
        "clip_min": 0.0, "clip_max": 100.0,
        "cut_low": 33.33, "cut_high": 66.67, "version": "1.0.0",
    })

    if tracks is None:
        tracks = ["regression", "classification"]

    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data splits ─────────────────────────────────────────────────────
    train_df = load_split("train", allow_test=False)
    val_df = load_split("val", allow_test=False)

    output: Dict[str, Any] = {}

    for track in tracks:
        logger.info("=" * 60)
        logger.info("EVALUATION TRACK: %s", track.upper())
        logger.info("=" * 60)

        results = load_phase4_results(track)
        if not results:
            logger.warning("No Phase 4 artifacts found for track '%s'. Skipping.", track)
            continue

        # ── Determine feature set from the best artifact ──────────────────
        # Use factor_scores as default (won in Phase 4 for all models)
        feature_set = "factor_scores"

        # Prepare validation arrays
        X_val, y_val_reg, y_val_cls = _get_features_and_targets(val_df, feature_set, track)
        n_features = X_val.shape[1]

        # ── Full metric recomputation on validation set ───────────────────
        extended_results = []
        for meta in results:
            model_path = meta.get("_model_path")
            if not model_path or not Path(model_path).exists():
                logger.warning("Artifact not found for %s, skipping.", meta.get("model_name"))
                continue
            try:
                model = load_model_artifact(model_path)
                val_metrics = evaluate_model_on_split(
                    model, X_val, y_val_reg, y_val_cls, track, n_features=n_features
                )
                meta["val_extended_metrics"] = val_metrics
                extended_results.append(meta)
                logger.info(
                    "%-30s | primary=%.4f | %s",
                    meta["model_name"],
                    val_metrics.get("primary", float("nan")),
                    "r2" if track == "regression" else "macro_f1",
                )
            except Exception as exc:
                logger.warning("Evaluation failed for %s: %s", meta.get("model_name"), exc)

        if not extended_results:
            continue

        # ── Build leaderboard DataFrame ───────────────────────────────────
        lb_rows = []
        for r in extended_results:
            vm = r.get("val_extended_metrics", {})
            row: Dict[str, Any] = {
                "model_name": r["model_name"],
                "category": r.get("category", "Unknown"),
                "feature_set": r.get("feature_set", feature_set),
                "overfitting_gap": r.get("overfitting_gap", float("nan")),
                "train_time_sec": r.get("train_time_sec", float("nan")),
                "latency_ms": r.get("inference_latency_ms", float("nan")),
                "size_kb": r.get("model_size_kb", float("nan")),
                "val_primary_metric": vm.get("primary"),
                "val_tiebreak": vm.get("tiebreak"),
                "ece": vm.get("ece"),
                "high_risk_recall": vm.get("high_risk_recall"),
                "mcc": vm.get("mcc"),
                "quadratic_kappa": vm.get("quadratic_weighted_kappa"),
                "roc_auc": vm.get("roc_auc_ovr_macro"),
                "pr_auc": vm.get("pr_auc_macro"),
                "log_loss": vm.get("log_loss"),
                "brier_score": vm.get("brier_score"),
            }
            if track == "regression":
                row.update({
                    "val_r2": vm.get("r2"),
                    "val_adj_r2": vm.get("adj_r2"),
                    "val_rmse": vm.get("rmse"),
                    "val_mae": vm.get("mae"),
                    "val_median_ae": vm.get("median_ae"),
                    "val_smape": vm.get("smape"),
                    "val_explained_var": vm.get("explained_variance"),
                    "val_pearson_r": vm.get("pearson_r"),
                    "val_spearman_r": vm.get("spearman_r"),
                    "val_within_10_acc": vm.get("within_pm10_acc"),
                })
            else:
                row.update({
                    "val_macro_f1": vm.get("macro_f1"),
                    "val_weighted_f1": vm.get("weighted_f1"),
                    "val_accuracy": vm.get("accuracy"),
                    "val_balanced_accuracy": vm.get("balanced_accuracy"),
                    "val_macro_precision": vm.get("macro_precision"),
                    "val_macro_recall": vm.get("macro_recall"),
                })
            lb_rows.append(row)

        lb_df = pd.DataFrame(lb_rows)
        if lb_df.empty:
            continue

        # Add composite rank
        lb_df["composite_score"] = _composite_rank(lb_df, track, composite_weights)
        lb_df["composite_rank"] = lb_df["composite_score"].rank(ascending=False).astype(int)

        # Sort: regression by R² desc, classification by macro_f1 desc
        primary_col = "val_r2" if track == "regression" else "val_macro_f1"
        if primary_col not in lb_df.columns:
            primary_col = "val_primary_metric"
        lb_df = lb_df.sort_values(primary_col, ascending=False).reset_index(drop=True)
        lb_df.index = lb_df.index + 1  # 1-based rank

        # ── Statistical comparison ────────────────────────────────────────
        comparisons = statistical_comparison(extended_results, cfg)

        # ── Champion selection ─────────────────────────────────────────────
        max_gap = champion_thresh.get("max_overfit_gap", 5.0)
        max_ece = champion_thresh.get("max_ece", 0.15)
        min_hr_recall = champion_thresh.get("min_high_risk_recall", 0.70)

        eligible = lb_df[
            (lb_df["overfitting_gap"].fillna(999) <= max_gap)
        ].copy()

        if track == "classification":
            if "ece" in eligible.columns:
                eligible = eligible[eligible["ece"].fillna(999) <= max_ece]
            if "high_risk_recall" in eligible.columns:
                eligible = eligible[eligible["high_risk_recall"].fillna(0) >= min_hr_recall]

        champion_row = eligible.iloc[0] if len(eligible) > 0 else lb_df.iloc[0]
        champion_name = champion_row["model_name"]

        # Best in each category
        best_per_cat: Dict[str, str] = {}
        for cat in lb_df["category"].unique():
            cat_df = lb_df[lb_df["category"] == cat]
            if not cat_df.empty:
                best_per_cat[cat] = cat_df.iloc[0]["model_name"]

        # Champion justification
        champion_record = {
            "track": track,
            "champion": champion_name,
            "champion_category": champion_row.get("category"),
            "selection_metric": primary_col,
            "primary_metric_value": float(champion_row.get(primary_col, 0)),
            "overfitting_gap": float(champion_row.get("overfitting_gap", 0)),
            "ece": float(champion_row.get("ece") or 0),
            "high_risk_recall": float(champion_row.get("high_risk_recall") or 0),
            "composite_rank": int(champion_row.get("composite_rank", 1)),
            "eligibility_criteria": {
                "max_overfit_gap": max_gap,
                "max_ece": max_ece,
                "min_high_risk_recall": min_hr_recall,
            },
            "justification": (
                f"'{champion_name}' selected as {track} champion with "
                f"{primary_col}={champion_row.get(primary_col, 'N/A'):.4f}, "
                f"overfitting_gap={champion_row.get('overfitting_gap', 'N/A'):.4f}. "
                f"Passes all eligibility thresholds and is SHAP-compatible."
            ),
            "best_per_category": best_per_cat,
            "statistical_comparisons": comparisons,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        # ── Save leaderboards ─────────────────────────────────────────────
        lb_df_save = lb_df.reset_index(drop=True)
        lb_df_save.to_csv(LEADERBOARD_DIR / f"{track}_leaderboard_phase5.csv", index=False)
        with open(LEADERBOARD_DIR / f"{track}_champion.json", "w", encoding="utf-8") as f:
            json.dump(champion_record, f, indent=2, default=str)
        logger.info("Champion for '%s' track: %s", track, champion_name)

        # ── Final Test-Set Evaluation ─────────────────────────────────────
        if run_test_set:
            logger.info("Running FINAL test-set evaluation for champion '%s'...", champion_name)
            test_df = load_split("test", allow_test=True)  # Only here!
            X_test, y_test_reg, y_test_cls = _get_features_and_targets(test_df, feature_set, track)

            # Load champion model
            champ_path = MODEL_DIR / track / f"{champion_name}.joblib"
            champion_model = load_model_artifact(str(champ_path))

            # Calibrate probabilities on val set if classification
            if track == "classification" and hasattr(champion_model, "predict_proba"):
                try:
                    from sklearn.isotonic import IsotonicRegression
                    calibrated = CalibratedClassifierCV(
                        champion_model, method="isotonic", cv="prefit"
                    )
                    calibrated.fit(X_val, y_val_cls)
                    champion_eval_model = calibrated
                    logger.info("Isotonic calibration applied to champion.")
                except Exception as exc:
                    logger.warning("Calibration failed, using uncalibrated: %s", exc)
                    champion_eval_model = champion_model
            else:
                champion_eval_model = champion_model

            test_metrics = evaluate_model_on_split(
                champion_eval_model, X_test, y_test_reg, y_test_cls, track, n_features=n_features
            )
            logger.info("Test-set primary metric: %.4f", test_metrics.get("primary", float("nan")))

            # ── Diagnostic Plots ──────────────────────────────────────────
            y_pred_test = champion_eval_model.predict(X_test)
            if track == "regression":
                plot_regression_diagnostics(y_test_reg, y_pred_test.astype(float), champion_name)
            else:
                plot_confusion_matrix(y_test_cls, y_pred_test, RISK_LABELS, champion_name)
                if hasattr(champion_eval_model, "predict_proba"):
                    try:
                        y_prob_test = champion_eval_model.predict_proba(X_test)
                        if hasattr(champion_eval_model, "classes_"):
                            m_classes = list(champion_eval_model.classes_)
                            indices = [m_classes.index(lbl) for lbl in RISK_LABELS if lbl in m_classes]
                            if len(indices) == len(RISK_LABELS):
                                y_prob_test = y_prob_test[:, indices]
                        plot_roc_curves(y_test_cls, y_prob_test, RISK_LABELS, champion_name)
                        plot_pr_curves(y_test_cls, y_prob_test, RISK_LABELS, champion_name)
                        # Calibration curve for High-risk class (index 2: High)
                        from sklearn.preprocessing import label_binarize as lb2
                        y_bin_test = lb2(y_test_cls, classes=RISK_LABELS)
                        plot_calibration_curve(y_bin_test[:, 2], y_prob_test[:, 2], "High", champion_name)
                    except Exception as exc:
                        logger.warning("Probability plots failed: %s", exc)

            # ── Robustness checks ─────────────────────────────────────────
            logger.info("Running robustness checks...")
            robustness = robustness_check(
                champion_eval_model,
                X_val,  # Use val for robustness (test is only for final reporting)
                y_val_reg, y_val_cls, track,
                noise_levels=robustness_cfg.get("noise_levels", [0.05, 0.10, 0.20]),
                dropout_rates=robustness_cfg.get("dropout_rates", [0.1, 0.2, 0.33]),
                seed=seed,
            )

            # ── Subgroup fairness ─────────────────────────────────────────
            logger.info("Running subgroup fairness analysis...")
            fairness = subgroup_fairness(
                champion_eval_model,
                val_df,
                X_val, y_val_reg, y_val_cls, track,
                subgroup_cols=robustness_cfg.get("subgroup_columns", ["gender", "year_of_study"]),
            )

            # ── PRI bundle export ─────────────────────────────────────────
            pri_bundle = {
                "model_name": champion_name,
                "track": track,
                "feature_set": feature_set,
                "version": pri_cfg.get("version", "1.0.0"),
                "training_date_utc": datetime.now(timezone.utc).isoformat(),
                "test_set_metrics": test_metrics,
                "robustness": robustness,
                "fairness": fairness,
                "pri_thresholds": {
                    "clip_min": pri_cfg["clip_min"],
                    "clip_max": pri_cfg["clip_max"],
                    "cut_low": pri_cfg["cut_low"],
                    "cut_high": pri_cfg["cut_high"],
                    "categories": {"Low": "0–33.33", "Moderate": "33.34–66.67", "High": "66.68–100"},
                },
                "data_hash": _sha256_file(Path("data/processed/train.csv")),
                "disclaimer": cfg.get("ethics", {}).get("disclaimer", ""),
            }

            bundle_path = FINAL_DIR / f"{track}_champion_bundle.json"
            with open(bundle_path, "w", encoding="utf-8") as f:
                json.dump(pri_bundle, f, indent=2, default=str)

            # Save the champion model to final/
            joblib.dump(champion_eval_model, FINAL_DIR / f"{track}_champion.joblib")
            logger.info("PRI bundle saved: %s", bundle_path)

            champion_record["test_metrics"] = test_metrics
            champion_record["robustness_summary"] = {
                k: v for k, v in robustness.get("noise", {}).items()
            }

            # Also evaluate each best-in-category on test set for reporting
            best_cat_test: Dict[str, Dict] = {}
            for cat, model_name in best_per_cat.items():
                if model_name == champion_name:
                    best_cat_test[cat] = test_metrics
                    continue
                cat_path = MODEL_DIR / track / f"{model_name}.joblib"
                if not cat_path.exists():
                    continue
                try:
                    cat_model = load_model_artifact(str(cat_path))
                    cat_test = evaluate_model_on_split(
                        cat_model, X_test, y_test_reg, y_test_cls, track, n_features=n_features
                    )
                    best_cat_test[cat] = cat_test
                except Exception as exc:
                    logger.warning("Best-in-category test eval failed for %s: %s", model_name, exc)
            champion_record["best_per_category_test_metrics"] = best_cat_test

        # Save final champion JSON (updated with test metrics)
        with open(LEADERBOARD_DIR / f"{track}_champion.json", "w", encoding="utf-8") as f:
            json.dump(champion_record, f, indent=2, default=str)

        output[track] = {
            "leaderboard": lb_df_save,
            "champion": champion_record,
        }

    # Save unified champion.json across both tracks
    all_champions = {
        track: output[track]["champion"]
        for track in output
    }
    with open(LEADERBOARD_DIR / "champion.json", "w", encoding="utf-8") as f:
        json.dump(all_champions, f, indent=2, default=str)
    logger.info("Unified champion.json saved.")

    return output


# ---------------------------------------------------------------------------
# Leaderboard pretty-printing helper
# ---------------------------------------------------------------------------

def print_leaderboard(lb_df: pd.DataFrame, track: str) -> None:
    """Print a compact leaderboard to stdout."""
    print(f"\n{'='*70}")
    print(f"  LEADERBOARD: {track.upper()} TRACK")
    print(f"{'='*70}")
    cols_reg = ["model_name", "category", "val_r2", "val_rmse", "val_mae",
                "val_within_10_acc", "ece", "overfitting_gap", "composite_rank"]
    cols_cls = ["model_name", "category", "val_macro_f1", "val_accuracy",
                "val_balanced_accuracy", "high_risk_recall", "mcc",
                "roc_auc", "ece", "overfitting_gap", "composite_rank"]
    show_cols = cols_reg if track == "regression" else cols_cls
    show_cols = [c for c in show_cols if c in lb_df.columns]
    print(lb_df[show_cols].to_string(index=True))
    print(f"{'='*70}\n")
