"""Scikit-Learn Compatible Transformers for Multimodal Engagement Intelligence."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CANONICAL_INDICATORS = [
    "resource_access",
    "study_time",
    "review_time",
    "quiz_practice",
    "assignment_engagement",
    "study_consistency",
    "participation",
]

PERIODS = ["p1", "p2", "p3", "p4"]


def get_canonical_28_columns() -> List[str]:
    """Return ordered list of 28 canonical indicator feature names."""
    cols = []
    for ind in CANONICAL_INDICATORS:
        for p in PERIODS:
            cols.append(f"{ind}_{p}")
    return cols


class PeriodReshaper(BaseEstimator, TransformerMixin):
    """Reshapes 28 canonical tabular columns into (n_samples, 4, 7) sequence tensor.

    Dimension 0: Samples (n)
    Dimension 1: Periods (4: P1, P2, P3, P4)
    Dimension 2: Indicators (7: canonical survey indicators)
    """

    def __init__(
        self,
        indicators: Optional[List[str]] = None,
        periods: Optional[List[str]] = None,
    ):
        self.indicators = indicators or CANONICAL_INDICATORS
        self.periods = periods or PERIODS
        self.expected_columns_ = [
            f"{ind}_{p}" for ind in self.indicators for p in self.periods
        ]

    def fit(self, X: Any, y: Any = None) -> "PeriodReshaper":
        """Validate input feature dimensions."""
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.expected_columns_ if c not in X.columns]
            if missing:
                raise ValueError(f"PeriodReshaper missing required columns: {missing}")
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Transform 2D (n, 28) array or DataFrame into 3D (n, 4, 7) tensor."""
        if isinstance(X, pd.DataFrame):
            df_vals = X[self.expected_columns_].values
        else:
            df_vals = np.asarray(X)
            if df_vals.shape[1] != len(self.expected_columns_):
                raise ValueError(
                    f"Expected 28 columns for PeriodReshaper, got {df_vals.shape[1]}"
                )

        n_samples = len(df_vals)
        tensor = np.zeros(
            (n_samples, len(self.periods), len(self.indicators)), dtype=float
        )

        col_idx = 0
        for ind_idx in range(len(self.indicators)):
            for p_idx in range(len(self.periods)):
                tensor[:, p_idx, ind_idx] = df_vals[:, col_idx]
                col_idx += 1

        return tensor

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        """Invert 3D (n, 4, 7) tensor back to 2D (n, 28) DataFrame."""
        if X.ndim != 3 or X.shape[1] != 4 or X.shape[2] != 7:
            raise ValueError(f"Expected shape (n, 4, 7), got {X.shape}")

        n_samples = X.shape[0]
        data = np.zeros((n_samples, 28), dtype=float)

        col_idx = 0
        for ind_idx in range(len(self.indicators)):
            for p_idx in range(len(self.periods)):
                data[:, col_idx] = X[:, p_idx, ind_idx]
                col_idx += 1

        return pd.DataFrame(data, columns=self.expected_columns_)


class EngagementImputer(BaseEstimator, TransformerMixin):
    """Honest imputer fitted strictly on training folds (Rule G3).

    Methods:
      - 'median': Computes median per indicator-period for ordinal codes (1..5).
      - 'student_interpolation': Linear interpolation within student across P1..P4,
        with forward/back-fill, followed by median for any completely blank indicators.
    Also appends missing-indicator binary columns for any feature with >3% missing
    in the training fold.
    """

    def __init__(
        self,
        method: str = "median",
        missing_threshold: float = 0.03,
        add_indicators: bool = True,
    ):
        self.method = method
        self.missing_threshold = missing_threshold
        self.add_indicators = add_indicators
        self.medians_: Dict[str, float] = {}
        self.high_missing_cols_: List[str] = []
        self.feature_names_: List[str] = []

    def fit(self, X: Any, y: Any = None) -> "EngagementImputer":
        """Learn medians and high-missingness indicator columns from training."""
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        else:
            X_df = X.copy()

        self.feature_names_ = list(X_df.columns)
        self.medians_ = {}
        self.high_missing_cols_ = []

        n_rows = len(X_df)
        for col in self.feature_names_:
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                continue
            missing_rate = X_df[col].isna().sum() / max(1, n_rows)
            # Medians are rounded to nearest ordinal integer [1, 5]
            med = X_df[col].dropna().median()
            med_val = 3.0 if pd.isna(med) else float(round(med))
            self.medians_[col] = med_val

            if self.add_indicators and missing_rate > self.missing_threshold:
                self.high_missing_cols_.append(col)

        return self

    def transform(self, X: Any) -> pd.DataFrame:
        """Impute missing values and generate missing-indicator columns."""
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.feature_names_)
        else:
            X_df = X.copy()

        # 1. Missing indicators
        indicator_data: Dict[str, np.ndarray] = {}
        for col in self.high_missing_cols_:
            if col in X_df.columns:
                indicator_data[f"missing_{col}"] = X_df[col].isna().astype(float).values

        # 2. Imputation
        if self.method == "student_interpolation":
            # Within-student temporal interpolation for 28 indicators
            clean_df = X_df.copy()

            for ind in CANONICAL_INDICATORS:
                p_cols = [
                    f"{ind}_{p}" for p in PERIODS if f"{ind}_{p}" in clean_df.columns
                ]
                if len(p_cols) == 4:
                    # Interpolate row-wise across periods 1..4
                    ind_vals = clean_df[p_cols].values.astype(float)
                    # Row-by-row interpolation
                    for i in range(len(ind_vals)):
                        row = ind_vals[i]
                        if np.isnan(row).any():
                            s = pd.Series(row)
                            s_interp = s.interpolate(
                                method="linear", limit_direction="both"
                            )
                            ind_vals[i] = s_interp.values
                    clean_df[p_cols] = np.round(ind_vals)

            # Impute remaining NaNs with learned training medians
            for col in self.feature_names_:
                if col in clean_df.columns:
                    clean_df[col] = clean_df[col].fillna(self.medians_.get(col, 3.0))
            out_df = clean_df
        else:
            # Simple median imputation
            out_df = X_df.copy()
            for col in self.feature_names_:
                if col in out_df.columns:
                    out_df[col] = out_df[col].fillna(self.medians_.get(col, 3.0))

        # 3. Concatenate missing indicators
        if indicator_data:
            ind_df = pd.DataFrame(indicator_data, index=out_df.index)
            out_df = pd.concat([out_df, ind_df], axis=1)

        return out_df


class EngagementFeatures(BaseEstimator, TransformerMixin):
    """Engineers longitudinal dynamics and composite engagement features.

    For every indicator (7):
      - mean, std, min, max
      - ols_slope across periods 1..4 (formula: 0.3*(p4 - p1) + 0.1*(p3 - p2))
      - last_minus_first (p4 - p1)
      - last_minus_mean_first_two (p4 - (p1 + p2)/2)
      - reversals (count of sign flips in successive period differences)

    Composite per period E_t = mean of 7 indicators rescaled to 0..100:
      - composite_p1..p4
      - mean, std, slope
      - max_drop (largest single-period downward drop)
      - reversals (direction changes in E_t)
      - auc (trapezoidal area under the curve)
      - last_value (E_4)
      - drop_from_best_to_last (max(E_t) - E_4)
    """

    def __init__(self, include_raw: bool = True):
        self.include_raw = include_raw
        self.feature_names_: List[str] = []

    def fit(self, X: Any, y: Any = None) -> "EngagementFeatures":
        """Inspect columns and precompute output feature names."""
        if not isinstance(X, pd.DataFrame):
            cols = get_canonical_28_columns()
        else:
            cols = list(X.columns)

        engineered_names: List[str] = []

        # Indicator-specific feature names (7 * 8 = 56)
        for ind in CANONICAL_INDICATORS:
            engineered_names.extend(
                [
                    f"{ind}_mean",
                    f"{ind}_std",
                    f"{ind}_min",
                    f"{ind}_max",
                    f"{ind}_slope",
                    f"{ind}_last_minus_first",
                    f"{ind}_last_minus_mean_first_two",
                    f"{ind}_reversals",
                ]
            )

        # Composite feature names (4 period composites + 8 dynamics = 12)
        for p in PERIODS:
            engineered_names.append(f"composite_{p}")

        engineered_names.extend(
            [
                "composite_mean",
                "composite_std",
                "composite_slope",
                "composite_max_drop",
                "composite_reversals",
                "composite_auc",
                "composite_last_value",
                "composite_drop_from_best_to_last",
            ]
        )

        if self.include_raw:
            self.feature_names_ = cols + engineered_names
        else:
            self.feature_names_ = engineered_names

        return self

    def transform(self, X: Any) -> pd.DataFrame:
        """Transform 28 canonical indicators into rich dynamic feature matrix."""
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X, columns=get_canonical_28_columns())
        else:
            df = X.copy()

        n_samples = len(df)
        eng_features: Dict[str, np.ndarray] = {}

        # 1. Per-indicator dynamic features
        for ind in CANONICAL_INDICATORS:
            p_cols = [f"{ind}_{p}" for p in PERIODS]
            if not all(col in df.columns for col in p_cols):
                continue

            vals = df[p_cols].values.astype(float)  # shape (n, 4)
            p1, p2, p3, p4 = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]

            eng_features[f"{ind}_mean"] = np.mean(vals, axis=1)
            eng_features[f"{ind}_std"] = np.std(vals, axis=1, ddof=1)
            eng_features[f"{ind}_min"] = np.min(vals, axis=1)
            eng_features[f"{ind}_max"] = np.max(vals, axis=1)

            # OLS slope against t = [1, 2, 3, 4] -> sum((t-t_bar)^2) = 5.0
            # slope = sum((t-2.5)*y) / 5.0 = (-1.5*p1 - 0.5*p2 + 0.5*p3 + 1.5*p4) / 5.0
            eng_features[f"{ind}_slope"] = 0.3 * (p4 - p1) + 0.1 * (p3 - p2)

            eng_features[f"{ind}_last_minus_first"] = p4 - p1
            eng_features[f"{ind}_last_minus_mean_first_two"] = p4 - 0.5 * (p1 + p2)

            # Reversals in trajectory
            d1 = p2 - p1
            d2 = p3 - p2
            d3 = p4 - p3
            rev1 = ((d1 * d2) < 0).astype(float)
            rev2 = ((d2 * d3) < 0).astype(float)
            eng_features[f"{ind}_reversals"] = rev1 + rev2

        # 2. Composite Engagement per period E_t (rescaled 0..100)
        e_periods = np.zeros((n_samples, 4), dtype=float)
        for t_idx, p in enumerate(PERIODS):
            p_ind_cols = [
                f"{ind}_{p}"
                for ind in CANONICAL_INDICATORS
                if f"{ind}_{p}" in df.columns
            ]
            if p_ind_cols:
                # Coded 1..5 -> rescale ((mean - 1) / 4) * 100
                raw_mean = df[p_ind_cols].mean(axis=1).values
                e_t = ((raw_mean - 1.0) / 4.0) * 100.0
                e_periods[:, t_idx] = np.clip(e_t, 0.0, 100.0)
                eng_features[f"composite_{p}"] = e_periods[:, t_idx]

        e1 = e_periods[:, 0]
        e2 = e_periods[:, 1]
        e3 = e_periods[:, 2]
        e4 = e_periods[:, 3]

        eng_features["composite_mean"] = np.mean(e_periods, axis=1)
        eng_features["composite_std"] = np.std(e_periods, axis=1, ddof=1)
        eng_features["composite_slope"] = 0.3 * (e4 - e1) + 0.1 * (e3 - e2)

        # Max downward drop between any adjacent periods
        drop1 = np.maximum(0.0, e1 - e2)
        drop2 = np.maximum(0.0, e2 - e3)
        drop3 = np.maximum(0.0, e3 - e4)
        eng_features["composite_max_drop"] = np.maximum(drop1, np.maximum(drop2, drop3))

        # Reversals in composite
        de1 = e2 - e1
        de2 = e3 - e2
        de3 = e4 - e3
        rev_e1 = ((de1 * de2) < 0).astype(float)
        rev_e2 = ((de2 * de3) < 0).astype(float)
        eng_features["composite_reversals"] = rev_e1 + rev_e2

        # Area under curve (trapezoidal rule across periods 1..4 with dt=1)
        # AUC = (e1+e2)/2 + (e2+e3)/2 + (e3+e4)/2 = 0.5*e1 + e2 + e3 + 0.5*e4
        eng_features["composite_auc"] = 0.5 * e1 + e2 + e3 + 0.5 * e4

        # Last value and drop from best
        eng_features["composite_last_value"] = e4
        max_e = np.max(e_periods, axis=1)
        eng_features["composite_drop_from_best_to_last"] = np.maximum(0.0, max_e - e4)

        eng_df = pd.DataFrame(eng_features, index=df.index)

        if self.include_raw:
            return pd.concat([df, eng_df], axis=1)
        return eng_df


class ConstantNearConstantFilter(BaseEstimator, TransformerMixin):
    """Filters constant and near-constant columns on training folds."""

    def __init__(self, variance_threshold: float = 1e-6):
        self.variance_threshold = variance_threshold
        self.kept_columns_: List[str] = []
        self.dropped_columns_: List[str] = []

    def fit(self, X: Any, y: Any = None) -> "ConstantNearConstantFilter":
        """Detect zero or near-zero variance features."""
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X

        variances = X_df.var()
        self.kept_columns_ = list(variances[variances > self.variance_threshold].index)
        self.dropped_columns_ = list(
            variances[variances <= self.variance_threshold].index
        )
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        """Drop constant columns identified during training fit."""
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X

        valid_cols = [c for c in self.kept_columns_ if c in X_df.columns]
        return X_df[valid_cols]
