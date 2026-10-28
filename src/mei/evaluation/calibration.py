"""Calibration, Threshold Tuning, and MEI Scoring Bundle Generation.

Calibrates champion probabilities via Isotonic Regression on out-of-fold predictions
from Train+Val (N=1,856 >= 1,000). Reports ECE and Brier score before vs after calibration,
tunes optimal and sensitive decision thresholds, validates risk band monotonicity on the
holdout test set, and packages the complete champion bundle.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score

from mei.evaluation.metrics import compute_expected_calibration_error

logger = logging.getLogger(__name__)

# Project Palette
COLOR_SURFACE = "#FFFFFF"
COLOR_PRIMARY = "#1F4E46"
COLOR_LINE = "#DDD8CE"
COLOR_UNRELIABLE = "#B3261E"
COLOR_CALIBRATED = "#2E7D4F"
COLOR_MUTED = "#5B6168"


def tune_decision_thresholds(
    y_true: np.ndarray,
    calibrated_probs: np.ndarray,
) -> Dict[str, Any]:
    """Search thresholds to find optimal F1 threshold and sensitive (Recall >= 0.80) threshold.

    Searches tau in [0.05, 0.95] in steps of 0.01.
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_f1 = -1.0
    best_tau = 0.5
    best_prec = 0.0
    best_rec = 0.0

    sensitive_tau = 0.5
    sensitive_rec = 0.0
    sensitive_prec = 0.0
    sensitive_f1 = 0.0

    # For sensitive threshold, we want Recall >= 0.80 with highest F1/precision
    valid_sensitive_candidates = []

    for tau in thresholds:
        pred = (calibrated_probs >= tau).astype(int)
        f1 = float(f1_score(y_true, pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true, pred, pos_label=1, zero_division=0))
        prec = float(precision_score(y_true, pred, pos_label=1, zero_division=0))

        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
            best_prec = prec
            best_rec = rec

        if rec >= 0.80:
            valid_sensitive_candidates.append((tau, f1, rec, prec))

    if valid_sensitive_candidates:
        # Pick candidate with highest F1 among those with Recall >= 0.80
        valid_sensitive_candidates.sort(key=lambda x: (x[1], x[3]), reverse=True)
        s_tau, s_f1, s_rec, s_prec = valid_sensitive_candidates[0]
        sensitive_tau = float(s_tau)
        sensitive_f1 = float(s_f1)
        sensitive_rec = float(s_rec)
        sensitive_prec = float(s_prec)
    else:
        sensitive_tau = float(best_tau)
        sensitive_f1 = float(best_f1)
        sensitive_rec = float(best_rec)
        sensitive_prec = float(best_prec)

    return {
        "optimal_threshold": {
            "threshold": round(best_tau, 4),
            "f1": round(best_f1, 4),
            "recall": round(best_rec, 4),
            "precision": round(best_prec, 4),
            "description": "Threshold maximizing F1 on at-risk student classification",
        },
        "sensitive_threshold": {
            "threshold": round(sensitive_tau, 4),
            "f1": round(sensitive_f1, 4),
            "recall": round(sensitive_rec, 4),
            "precision": round(sensitive_prec, 4),
            "description": "High-sensitivity triage threshold ensuring at least 80% recall of at-risk students",
        },
    }


def fit_isotonic_calibrator(
    y_train_val: np.ndarray,
    oof_probs: np.ndarray,
) -> Tuple[IsotonicRegression, Dict[str, Any]]:
    """Fit Isotonic Regression on out-of-fold probabilities and evaluate metrics before vs after."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof_probs, y_train_val)
    cal_oof_probs = iso.predict(oof_probs)

    ece_before = float(compute_expected_calibration_error(y_train_val, oof_probs))
    ece_after = float(compute_expected_calibration_error(y_train_val, cal_oof_probs))

    brier_before = float(brier_score_loss(y_train_val, oof_probs))
    brier_after = float(brier_score_loss(y_train_val, cal_oof_probs))

    metrics = {
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
        "ece_relative_reduction_pct": round(
            float((ece_before - ece_after) / ece_before * 100), 2
        )
        if ece_before > 0
        else 0.0,
        "brier_before": round(brier_before, 6),
        "brier_after": round(brier_after, 6),
        "brier_relative_reduction_pct": round(
            float((brier_before - brier_after) / brier_before * 100), 2
        )
        if brier_before > 0
        else 0.0,
    }

    return iso, metrics


def plot_reliability_diagram(
    y_true: np.ndarray,
    probs_uncal: np.ndarray,
    probs_cal: np.ndarray,
    metrics: Dict[str, Any],
    output_dirs: List[Path],
) -> None:
    """Render publication figure showing reliability diagram before vs after calibration."""
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(COLOR_SURFACE)

    # Reference perfectly calibrated diagonal
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=COLOR_MUTED,
        linewidth=1.2,
        label="Perfect Calibration",
    )

    # Uncalibrated curve
    prob_true_uncal, prob_pred_uncal = calibration_curve(
        y_true, probs_uncal, n_bins=10, strategy="uniform"
    )
    ax.plot(
        prob_pred_uncal,
        prob_true_uncal,
        marker="o",
        linewidth=2.0,
        color=COLOR_UNRELIABLE,
        label=f"Uncalibrated (ECE = {metrics['ece_before']:.4f}, Brier = {metrics['brier_before']:.4f})",
    )

    # Calibrated curve
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_true, probs_cal, n_bins=10, strategy="uniform"
    )
    ax.plot(
        prob_pred_cal,
        prob_true_cal,
        marker="s",
        linewidth=2.2,
        color=COLOR_CALIBRATED,
        label=f"Isotonic Calibrated (ECE = {metrics['ece_after']:.4f}, Brier = {metrics['brier_after']:.4f})",
    )

    ax.set_xlabel(
        "Mean Predicted Probability", fontsize=10, fontweight="bold", color="#1A1D21"
    )
    ax.set_ylabel(
        "Empirical Fraction of Positives (At-Risk)",
        fontsize=10,
        fontweight="bold",
        color="#1A1D21",
    )
    ax.set_title(
        "Champion Probability Calibration: Reliability Diagram\nIsotonic Regression on Out-of-Fold Predictions (N = 1,856)",
        fontsize=11,
        fontweight="bold",
        pad=12,
        color=COLOR_PRIMARY,
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle="--", alpha=0.35, color=COLOR_LINE)
    ax.legend(framealpha=0.95, loc="upper left", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    filenames = ["fig_calibration_reliability.png", "fig_calibration_reliability.svg"]
    for out_dir in output_dirs:
        for fname in filenames:
            out_path = out_dir / fname
            dpi = 300 if fname.endswith(".png") else None
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved calibration reliability diagram: {out_path}")

    plt.close(fig)


def verify_band_monotonicity_on_test(
    y_test: np.ndarray,
    calibrated_test_probs: np.ndarray,
    bands_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify that empirical at-risk percentage rises monotonically across Low, Moderate, and High bands."""
    if bands_config is None:
        bands_config = {
            "low": {"min": 0, "max": 34, "label": "Low Risk"},
            "moderate": {"min": 35, "max": 64, "label": "Moderate Risk"},
            "high": {"min": 65, "max": 100, "label": "High Risk"},
        }

    mei_scores = np.round(100.0 * calibrated_test_probs).astype(int)

    band_records = []
    for band_key in ["low", "moderate", "high"]:
        b_info = bands_config[band_key]
        b_min, b_max = b_info["min"], b_info["max"]
        mask = (mei_scores >= b_min) & (mei_scores <= b_max)
        n_band = int(np.sum(mask))

        if n_band > 0:
            obs_at_risk_count = int(np.sum(y_test[mask]))
            obs_at_risk_pct = round(float(obs_at_risk_count / n_band * 100.0), 2)
            mean_mei = round(float(np.mean(mei_scores[mask])), 2)
        else:
            obs_at_risk_count = 0
            obs_at_risk_pct = 0.0
            mean_mei = 0.0

        band_records.append(
            {
                "band_key": band_key,
                "band_label": b_info.get("label", band_key.capitalize()),
                "score_range": f"{b_min}–{b_max}",
                "n_students": n_band,
                "observed_at_risk_count": obs_at_risk_count,
                "observed_at_risk_pct": obs_at_risk_pct,
                "mean_mei": mean_mei,
            }
        )

    # Check strict monotonicity: Low < Moderate < High
    rates = [r["observed_at_risk_pct"] for r in band_records]
    is_strictly_monotonic = bool(rates[0] < rates[1] < rates[2])

    return {
        "bands": band_records,
        "is_strictly_monotonic": is_strictly_monotonic,
        "observed_rates": rates,
        "monotonicity_status": "PASS" if is_strictly_monotonic else "FAIL",
    }


def save_champion_bundle(
    run_dir: Path,
    pipeline: Any,
    calibrator: IsotonicRegression,
    feature_names: List[str],
    thresholds_dict: Dict[str, Any],
    calibration_metrics: Dict[str, Any],
    band_analysis: Dict[str, Any],
    model_card_data: Dict[str, Any],
    shap_background_df: pd.DataFrame,
) -> Path:
    """Save the final frozen scoring bundle to artifacts/runs/<run_id>/champion/."""
    champion_dir = run_dir / "champion"
    champion_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pipeline and Calibrator
    joblib.dump(pipeline, champion_dir / "pipeline.joblib")
    joblib.dump(calibrator, champion_dir / "calibrator.joblib")

    # 2. Features list
    with open(champion_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # 3. Schema Version & Labels metadata
    schema_info = {
        "version": "1.0.0",
        "n_indicators": 7,
        "n_periods": 4,
        "total_survey_columns": 35,
        "feature_count": len(feature_names),
        "target": "at_risk",
        "target_formula": "attrition_intent_score >= 60.0",
        "mei_definition": "round(100 * calibrated_probability_at_risk)",
    }
    with open(champion_dir / "schema_version.json", "w", encoding="utf-8") as f:
        json.dump(schema_info, f, indent=2)

    # 4. Thresholds
    with open(champion_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=2)

    # 5. Bands configuration and observed test distribution
    bands_export = {
        "definitions": {
            "low": {
                "min": 0,
                "max": 34,
                "label": "Low Risk",
                "hex_color": "#2E7D4F",
                "action": "Routine monitoring",
            },
            "moderate": {
                "min": 35,
                "max": 64,
                "label": "Moderate Risk",
                "hex_color": "#C98A00",
                "action": "Advisor outreach & study groups",
            },
            "high": {
                "min": 65,
                "max": 100,
                "label": "High Risk",
                "hex_color": "#B3261E",
                "action": "Urgent student affairs intervention",
            },
        },
        "test_holdout_verification": band_analysis,
    }
    with open(champion_dir / "bands.json", "w", encoding="utf-8") as f:
        json.dump(bands_export, f, indent=2)

    # 6. SHAP background sample (200 rows, anonymized)
    anonymized_bg = shap_background_df.copy()
    if "student_id" in anonymized_bg.columns:
        anonymized_bg["student_id"] = [
            f"ANON_{i:04d}" for i in range(len(anonymized_bg))
        ]
    anonymized_bg.to_parquet(champion_dir / "shap_background_200.parquet", index=False)

    # 7. Model Card JSON
    with open(champion_dir / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card_data, f, indent=2)

    logger.info(f"Champion bundle frozen successfully at {champion_dir}")
    return champion_dir
