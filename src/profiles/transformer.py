"""Fold-fitted profile selection and model-feature generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.profiles.selection import DEFAULT_INDEX_COLUMNS, select_profile_candidate
from src.utils.config_loader import load_yaml
from src.utils.paths import get_project_root

PROFILE_INPUT_COLUMNS = (*DEFAULT_INDEX_COLUMNS, "overall_pressure_index")


class ProfileFeatureTransformer(BaseEstimator, TransformerMixin):
    """Learn profiles on fit rows and emit aligned profile descriptors."""

    def __init__(
        self,
        selection_config_path: str | None = None,
        index_columns: tuple[str, ...] = DEFAULT_INDEX_COLUMNS,
        overall_column: str = "overall_pressure_index",
    ) -> None:
        self.selection_config_path = selection_config_path
        self.index_columns = index_columns
        self.overall_column = overall_column

    def _selection_settings(self) -> dict[str, Any]:
        path = (
            Path(self.selection_config_path)
            if self.selection_config_path is not None
            else get_project_root() / "config" / "profile_selection.yaml"
        )
        return load_yaml(path).get("profile_selection", {})

    def _profile_matrix(self, X: pd.DataFrame) -> np.ndarray:
        columns = (*self.index_columns, self.overall_column)
        missing = set(columns) - set(X.columns)
        if missing:
            raise ValueError(f"Profile inputs are missing columns: {sorted(missing)}")
        return X.loc[:, columns].to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y: Any = None) -> ProfileFeatureTransformer:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ProfileFeatureTransformer requires a pandas DataFrame")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        matrix = self._profile_matrix(X)
        self.scaler_ = StandardScaler().fit(matrix)
        standardized = self.scaler_.transform(matrix)
        self.clusterer_, self.selection_summary_ = select_profile_candidate(
            standardized,
            settings=self._selection_settings(),
            index_columns=self.index_columns,
        )
        self.n_profiles_ = self.clusterer_.n_clusters
        self.profile_names_ = tuple(self.clusterer_.profile_names_)
        self.training_labels_ = self.clusterer_.predict(standardized)
        self.feature_names_out_ = self._profile_feature_names()
        return self

    def _profile_feature_names(self) -> tuple[str, ...]:
        one_hot = tuple(f"profile_id_{cluster}" for cluster in range(self.n_profiles_))
        probabilities = (
            tuple(
                f"profile_probability_{cluster}" for cluster in range(self.n_profiles_)
            )
            if self.clusterer_.algorithm.startswith("gmm_")
            else ()
        )
        return (
            *one_hot,
            *probabilities,
            "profile_distance_to_center",
            "profile_high_pressure_index_count",
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "clusterer_"):
            raise RuntimeError(
                "ProfileFeatureTransformer must be fitted before transform"
            )
        standardized = self.scaler_.transform(self._profile_matrix(X))
        labels = self.clusterer_.predict(standardized)
        profiles = pd.DataFrame(
            np.eye(self.n_profiles_, dtype=np.int8)[labels],
            index=X.index,
            columns=[f"profile_id_{cluster}" for cluster in range(self.n_profiles_)],
        )
        probabilities = self.clusterer_.membership_probabilities(standardized)
        if probabilities is not None:
            probability_frame = pd.DataFrame(
                probabilities,
                index=X.index,
                columns=[
                    f"profile_probability_{cluster}"
                    for cluster in range(self.n_profiles_)
                ],
            )
            profiles = pd.concat([profiles, probability_frame], axis=1)

        centers = self.clusterer_.centers[labels]
        distances = np.linalg.norm(standardized - centers, axis=1)
        profiles["profile_distance_to_center"] = distances
        profiles["profile_high_pressure_index_count"] = (
            X.loc[:, self.index_columns].gt(0.66).sum(axis=1).astype("int8")
        )
        return pd.concat([X.copy(), profiles], axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError(
                "ProfileFeatureTransformer must be fitted before requesting feature names"
            )
        names = self.feature_names_in_ if input_features is None else input_features
        return np.asarray([*names, *self.feature_names_out_], dtype=object)
