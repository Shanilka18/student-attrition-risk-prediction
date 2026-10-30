"""
Probability Calibration Engine (Platt Scaling & Isotonic Regression).
Fitted strictly on validation predictions to preserve unbiased probability estimates.

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger("AcademicIntelligence.Calibration")


class ProbabilityCalibrator:
    """
    Calibrates raw model risk probabilities to reflect true empirical attrition likelihoods.
    Fitted STRICTLY on validation partition outputs.
    """

    def __init__(self, method: str = "platt"):
        self.method = method.lower()
        if self.method not in ("platt", "isotonic"):
            raise ValueError(f"Unsupported calibration method: {method}. Choose 'platt' or 'isotonic'.")
        self.calibrator: Any = None
        self.is_fitted: bool = False
        self.brier_before: float = 1.0
        self.brier_after: float = 1.0

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "ProbabilityCalibrator":
        """
        Fits post-hoc calibrator on validation ground truth and raw predicted probabilities.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)

        self.brier_before = float(brier_score_loss(y_true, y_prob))

        if self.method == "platt":
            # Platt scaling: univariate logistic regression on log-odds
            # logit = log(p / (1 - p))
            logits = np.log(y_prob / (1.0 - y_prob)).reshape(-1, 1)
            clf = LogisticRegression(solver="lbfgs", C=1.0)
            clf.fit(logits, y_true)
            self.calibrator = clf
        elif self.method == "isotonic":
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(y_prob, y_true)
            self.calibrator = iso

        self.is_fitted = True
        calibrated_probs = self.transform(y_prob)
        self.brier_after = float(brier_score_loss(y_true, calibrated_probs))
        logger.info(
            f"ProbabilityCalibrator ({self.method}) fitted: "
            f"Brier {self.brier_before:.4f} -> {self.brier_after:.4f}"
        )
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Transforms raw probabilities to calibrated probabilities."""
        if not self.is_fitted:
            raise RuntimeError("ProbabilityCalibrator is not fitted yet.")

        y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)

        if self.method == "platt":
            logits = np.log(y_prob / (1.0 - y_prob)).reshape(-1, 1)
            calibrated = self.calibrator.predict_proba(logits)[:, 1]
        elif self.method == "isotonic":
            calibrated = self.calibrator.predict(y_prob)
        else:
            calibrated = y_prob

        return np.clip(calibrated.astype(np.float32), 0.0, 1.0)

    def summary(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "is_fitted": self.is_fitted,
            "brier_score_before": round(self.brier_before, 4),
            "brier_score_after": round(self.brier_after, 4),
            "brier_improvement_pct": round(
                max(0.0, (self.brier_before - self.brier_after) / max(self.brier_before, 1e-6)) * 100.0, 2
            ),
        }

    def save(self, filepath: Path) -> Path:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "ProbabilityCalibrator":
        return joblib.load(filepath)
