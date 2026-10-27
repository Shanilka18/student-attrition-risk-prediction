"""
Zero-Leakage Academic Data Preprocessor.
Fits imputation statistics and feature scaling strictly on training student records.
Serializes pipeline state via Joblib.

Component: Academic Intelligence Framework (IT23164680)
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger("AcademicIntelligence.Preprocessor")

NUMERIC_FEATURES = [
    "gpa",
    "exam_marks",
    "attendance_pct",
    "assignment_delay_days",
    "late_assignments_count"
]

CLIPPING_BOUNDS = {
    "gpa": (0.0, 4.0),
    "exam_marks": (0.0, 100.0),
    "attendance_pct": (0.0, 100.0),
    "assignment_delay_days": (0.0, 45.0),
    "late_assignments_count": (0, 15)
}


class AcademicPreprocessor:
    """
    Fits imputation medians and scaling parameters strictly on training partitions.
    Performs longitudinal forward-fill within student, followed by training-median back-off.
    """

    def __init__(self):
        self.is_fitted: bool = False
        self.train_medians: Dict[str, float] = {}
        self.train_means: Dict[str, float] = {}
        self.train_stds: Dict[str, float] = {}

    def fit(self, df_train_long: pd.DataFrame) -> "AcademicPreprocessor":
        """
        Learns imputation medians, means, and standard deviations strictly from training observations.
        """
        logger.info(f"Fitting preprocessor on {len(df_train_long)} training records...")

        for col in NUMERIC_FEATURES:
            valid_vals = df_train_long[col].dropna()
            if len(valid_vals) == 0:
                raise ValueError(f"Feature column '{col}' has no valid values in training set!")
            med = float(valid_vals.median())
            mean = float(valid_vals.mean())
            std = float(valid_vals.std()) if valid_vals.std() > 1e-6 else 1.0

            self.train_medians[col] = med
            self.train_means[col] = mean
            self.train_stds[col] = std

        self.is_fitted = True
        logger.info(f"Preprocessor fit complete. Learned medians: {self.train_medians}")
        return self

    def transform(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms student observations:
        1. Longitudinal forward-fill within each student sequence.
        2. Fills remaining leading NaNs using learned training medians.
        3. Clips outliers to valid academic bounds.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fit on training data before calling transform()!")

        df_out = df_long.copy()

        # Sort by student and semester to guarantee temporal order
        df_out = df_out.sort_values(["student_id", "semester"]).reset_index(drop=True)

        # 1. Forward-fill within each student
        for col in NUMERIC_FEATURES:
            df_out[col] = df_out.groupby("student_id")[col].ffill()

        # 2. Fill any remaining leading NaNs with training medians
        for col in NUMERIC_FEATURES:
            df_out[col] = df_out[col].fillna(self.train_medians[col])

        # 3. Clip outliers to valid physical bounds
        for col, (low, high) in CLIPPING_BOUNDS.items():
            df_out[col] = df_out[col].clip(lower=low, upper=high)

        return df_out

    def fit_transform(self, df_train_long: pd.DataFrame) -> pd.DataFrame:
        """Fits on training data and transforms it."""
        return self.fit(df_train_long).transform(df_train_long)

    def normalize_feature(self, series: pd.Series, feature_name: str) -> pd.Series:
        """Standardizes a feature using training mean and std."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fit before normalizing!")
        mean = self.train_means[feature_name]
        std = self.train_stds[feature_name]
        return (series - mean) / std

    def save(self, filepath: Path) -> None:
        """Saves fitted preprocessor to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved preprocessor to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "AcademicPreprocessor":
        """Loads fitted preprocessor from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor file not found at: {filepath}")
        obj = joblib.load(filepath)
        logger.info(f"Loaded preprocessor from: {filepath}")
        return obj
