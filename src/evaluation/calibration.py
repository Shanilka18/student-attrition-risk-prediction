"""Probability calibration engine and Environmental Risk Index (ERI) calculator.

Fits Platt Scaling (logistic sigmoid) and Isotonic Regression on validation predictions,
selects the best calibrator via Brier score and ECE, converts calibrated probabilities into ERI
[0-100], checks calibration thresholds, tunes decision thresholds, and assigns risk bands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from src.evaluation.metrics import compute_ece
from src.utils.logging_config import setup_logger
from src.utils.paths import get_artifacts_dir, get_configs_dir

logger = setup_logger("calibration")


class PlattCalibrator:
    """Platt scaling (univariate logistic regression on uncalibrated logits / probabilities)."""

    def __init__(self):
        self.lr = LogisticRegression(C=1.0, solver="lbfgs")

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        # LogisticRegression expects 2D feature matrix
        X = y_prob.reshape(-1, 1)
        self.lr.fit(X, y_true)
        return self

    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        X = y_prob.reshape(-1, 1)
        return self.lr.predict_proba(X)[:, 1]

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        return self.predict_proba(y_prob)


def fit_calibrators(
    val_probs: np.ndarray,
    val_y: np.ndarray,
) -> Tuple[Any, str, Dict[str, float]]:
    """Compare Platt scaling vs Isotonic regression on validation data.

    Returns:
        (best_calibrator_obj, best_calibrator_name, comparison_metrics)
    """
    raw_brier = float(np.mean((val_probs - val_y) ** 2))
    raw_ece = float(compute_ece(val_y, val_probs))

    # 1. Platt Sigmoid
    platt = PlattCalibrator().fit(val_probs, val_y)
    platt_probs = platt.predict_proba(val_probs)
    platt_brier = float(np.mean((platt_probs - val_y) ** 2))
    platt_ece = float(compute_ece(val_y, platt_probs))

    # 2. Isotonic
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(val_probs, val_y)
    iso_probs = isotonic.predict(val_probs)
    iso_brier = float(np.mean((iso_probs - val_y) ** 2))
    iso_ece = float(compute_ece(val_y, iso_probs))

    metrics = {
        "raw_brier": round(raw_brier, 4),
        "raw_ece": round(raw_ece, 4),
        "platt_brier": round(platt_brier, 4),
        "platt_ece": round(platt_ece, 4),
        "isotonic_brier": round(iso_brier, 4),
        "isotonic_ece": round(iso_ece, 4),
    }

    # Selection rule: Minimize Brier score, with ECE as tie-breaker
    if platt_brier <= iso_brier:
        best_name = "platt_sigmoid"
        best_calibrator = platt
    else:
        best_name = "isotonic"
        best_calibrator = isotonic

    logger.info(f"Calibration fit complete. Chosen method: {best_name} (Platt Brier={platt_brier:.4f}, Iso Brier={iso_brier:.4f})")
    return best_calibrator, best_name, metrics


def calibrate_classifier(
    val_probs: np.ndarray,
    val_y: np.ndarray,
    prevalence: Optional[float] = None,
) -> Dict[str, Any]:
    """Calibrate a classifier's validation probabilities and compute Brier Skill Score."""
    calibrator, cal_name, metrics = fit_calibrators(val_probs, val_y)
    cal_probs = calibrator.predict(val_probs)
    brier = float(np.mean((cal_probs - val_y) ** 2))
    ece = float(compute_ece(val_y, cal_probs))

    p = float(np.mean(val_y)) if prevalence is None else float(prevalence)
    brier_baseline = float(p * (1.0 - p))
    # Brier Skill Score: 1 - (Brier / Brier_baseline)
    bss = float(1.0 - (brier / (brier_baseline + 1e-12)))

    return {
        "calibrator": calibrator,
        "method": cal_name,
        "calibrated_probs": cal_probs,
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "brier_baseline": round(brier_baseline, 4),
        "brier_skill_score": round(bss, 4),
        "comparison_metrics": metrics,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = "balanced_accuracy",
    tie_breaker: str = "f1",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    num_steps: int = 91,
) -> Tuple[float, Dict[str, float]]:
    """Grid-search the decision threshold strictly on validation data.

    Maximizes objective (e.g. balanced_accuracy) with tie-breaker (e.g. f1).
    """
    thresholds = np.linspace(min_threshold, max_threshold, num_steps)
    best_th = 0.5
    best_score = -1.0
    best_tie = -1.0
    best_metrics = {}

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        bal_acc = float(balanced_accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))

        score = bal_acc if objective == "balanced_accuracy" else f1
        tie = f1 if objective == "balanced_accuracy" else bal_acc

        if score > best_score or (np.isclose(score, best_score, atol=1e-5) and tie > best_tie):
            best_score = score
            best_tie = tie
            best_th = float(th)
            best_metrics = {
                "threshold": round(float(th), 4),
                "balanced_accuracy": round(bal_acc, 4),
                "f1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            }

    return best_th, best_metrics


def calculate_eri(
    calibrated_probs: np.ndarray,
    ece: float,
    ece_threshold: float = 0.15,
) -> Tuple[np.ndarray, str, bool]:
    """Compute Environmental Risk Index = 100 * P_calibrated.

    If ECE stays above ece_threshold, relabel as 'normalised model-risk score'
    with is_calibrated=False.
    """
    eri_scores = np.clip(calibrated_probs * 100.0, 0.0, 100.0)

    if ece <= ece_threshold:
        label = "Environmental Risk Index (ERI)"
        is_calibrated = True
    else:
        label = "normalised model-risk score"
        is_calibrated = False
        logger.warning(
            f"ECE ({ece:.4f}) exceeds acceptable threshold ({ece_threshold}). "
            f"Relabeling score output to '{label}'."
        )

    return eri_scores, label, is_calibrated


def assign_risk_band(eri_score: float) -> Dict[str, str]:
    """Map ERI score (0-100) to human-interpretable institutional risk band."""
    if eri_score < 25.0:
        return {
            "band": "Low",
            "action": "Standard student support services",
            "sla": "Routine semester review",
        }
    elif eri_score < 50.0:
        return {
            "band": "Moderate",
            "action": "Advisory check-in & financial aid information",
            "sla": "Within 4 weeks",
        }
    elif eri_score < 75.0:
        return {
            "band": "High",
            "action": "Proactive bursary outreach & commute/housing counseling",
            "sla": "Within 2 weeks",
        }
    else:
        return {
            "band": "Critical",
            "action": "Immediate emergency assistance & flexible study accommodations",
            "sla": "Immediate / within 48 hours",
        }
