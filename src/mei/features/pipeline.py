"""Feature Pipeline Assembly, Named Feature Sets, and Pipeline Generators."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from mei.features.transformers import (
    PERIODS,
    ConstantNearConstantFilter,
    EngagementFeatures,
    EngagementImputer,
    PeriodReshaper,
    get_canonical_28_columns,
)

logger = logging.getLogger(__name__)

SINGLE_SIGNAL_COLUMNS = [f"resource_access_{p}" for p in PERIODS]


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects specific columns from a DataFrame."""

    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: Any, y: Any = None) -> "ColumnSelector":
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X[[c for c in self.columns if c in X.columns]].copy()
        X_df = pd.DataFrame(X)
        return X_df[self.columns].copy()


def compute_cohort_class_weights(y: Union[np.ndarray, pd.Series]) -> Dict[int, float]:
    """Compute balanced class weights for cost-sensitive learning.

    Formula: w_j = N / (2 * N_j)
    """
    y_arr = np.asarray(y).astype(int)
    classes = np.unique(y_arr)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_arr)
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    return weight_dict


def build_feature_pipeline(
    feature_set: str = "agg",
    scale: bool = False,
    imputer_method: str = "median",
    add_smote: bool = False,
    variance_threshold: Optional[float] = None,
    random_state: int = 42,
) -> Union[Pipeline, ImbPipeline]:
    """Build an honest, scikit-learn compatible feature pipeline.

    Fitted strictly on training folds to prevent leakage (Rule G3).

    Args:
        feature_set: One of 'raw28', 'agg', 'single_signal', 'sequence', 'agg_states'.
        scale: Whether to append StandardScaler (for linear models, SVM, KNN).
        imputer_method: 'median' or 'student_interpolation'.
        add_smote: If True, uses imblearn Pipeline and appends SMOTE inside fold.
        variance_threshold: Optional threshold to filter constant/near-constant columns.
        random_state: Seed for SMOTE oversampling.

    Returns:
        Configured Pipeline or ImbPipeline.
    """
    valid_sets = ["raw28", "agg", "single_signal", "sequence", "agg_states"]
    if feature_set not in valid_sets:
        raise ValueError(f"Unknown feature set '{feature_set}'. Allowed: {valid_sets}")

    steps: List[Tuple[str, Any]] = []

    if feature_set == "single_signal":
        steps.append(("selector", ColumnSelector(SINGLE_SIGNAL_COLUMNS)))
        steps.append(
            ("imputer", EngagementImputer(method=imputer_method, add_indicators=False))
        )
    elif feature_set == "raw28":
        steps.append(
            ("imputer", EngagementImputer(method=imputer_method, add_indicators=True))
        )
    elif feature_set == "agg":
        steps.append(
            ("imputer", EngagementImputer(method=imputer_method, add_indicators=True))
        )
        steps.append(("engagement_features", EngagementFeatures(include_raw=True)))
    elif feature_set == "agg_states":
        # Placeholder for Phase 5 HMM and clustering state features
        steps.append(
            ("imputer", EngagementImputer(method=imputer_method, add_indicators=True))
        )
        steps.append(("engagement_features", EngagementFeatures(include_raw=True)))
    elif feature_set == "sequence":
        steps.append(
            ("imputer", EngagementImputer(method=imputer_method, add_indicators=False))
        )
        steps.append(("reshaper", PeriodReshaper()))
        # Sequence reshaper returns 3D array; scaling/SMOTE not applied here
        return Pipeline(steps)

    # Optional variance threshold filtering
    if variance_threshold is not None:
        steps.append(
            (
                "variance_filter",
                ConstantNearConstantFilter(variance_threshold=variance_threshold),
            )
        )

    # Optional scaling for linear models
    if scale:
        steps.append(("scaler", StandardScaler()))

    # Optional SMOTE inside training fold (Rule G3 compliance)
    if add_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
        return ImbPipeline(steps)

    return Pipeline(steps)


def extract_features(
    df: pd.DataFrame,
    feature_set: str = "agg",
    imputer_method: str = "median",
    fitted_pipeline: Optional[Pipeline] = None,
) -> Tuple[Union[pd.DataFrame, np.ndarray], Pipeline]:
    """Extract and transform features using an honest pipeline (Rule G3).

    If fitted_pipeline is None, fits a new pipeline on df.
    Otherwise, applies fitted_pipeline.transform(df) WITHOUT refitting.
    """
    raw_28 = get_canonical_28_columns()
    X_input = df[raw_28].copy()

    if fitted_pipeline is None:
        pipeline = build_feature_pipeline(
            feature_set=feature_set,
            scale=False,
            imputer_method=imputer_method,
            add_smote=False,
        )
        X_trans = pipeline.fit_transform(X_input)
    else:
        pipeline = fitted_pipeline
        X_trans = pipeline.transform(X_input)

    return X_trans, pipeline
