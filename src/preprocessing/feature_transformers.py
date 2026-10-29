"""Fold-fitted imputation and questionnaire item encoding."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing.codec import EnvFinCodec
from src.utils.config_loader import load_yaml
from src.utils.paths import get_project_root

_NON_FEATURE_COLUMNS = {
    "student_id",
    "timestamp",
    "consent",
    "attention_check",
    "attrition_intent_score",
    "at_risk",
    "row_reasons",
}
_INTERMEDIATE_CODE_ITEMS = (
    "income_band",
    "aid_status",
    "part_time_job",
    "job_hours",
    "commute_time",
    "commute_spend",
    "home_internet",
    "main_device",
)


def _codec_from_paths(schema_path: str | None, mapping_path: str | None) -> EnvFinCodec:
    return EnvFinCodec(schema_path=schema_path, mapping_path=mapping_path)


class ItemImputer(BaseEstimator, TransformerMixin):
    """Impute item codes from the fitting partition and flag material missingness."""

    def __init__(
        self,
        codec: EnvFinCodec | None = None,
        missing_indicator_threshold: float = 0.03,
        schema_path: str | None = None,
        mapping_path: str | None = None,
    ) -> None:
        self.codec = codec
        self.missing_indicator_threshold = missing_indicator_threshold
        self.schema_path = schema_path
        self.mapping_path = mapping_path

    def _get_codec(self) -> EnvFinCodec:
        if self.codec is not None:
            return self.codec
        return _codec_from_paths(self.schema_path, self.mapping_path)

    def fit(self, X: pd.DataFrame, y: Any = None) -> ItemImputer:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ItemImputer requires a pandas DataFrame")
        if not 0 <= self.missing_indicator_threshold <= 1:
            raise ValueError("missing_indicator_threshold must be between 0 and 1")
        self.codec_ = self._get_codec()
        self.item_names_ = tuple(self.codec_.items)
        missing_items = set(self.item_names_) - set(X.columns)
        if missing_items:
            raise ValueError(f"Missing questionnaire items: {sorted(missing_items)}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.statistics_: dict[str, float] = {}
        self.strategies_: dict[str, str] = {}
        self.indicator_items_: tuple[str, ...] = ()
        indicator_items = []

        for item, metadata in self.codec_.items.items():
            values = pd.to_numeric(X[item], errors="coerce")
            observed = values.dropna()
            if observed.empty:
                raise ValueError(
                    f"Cannot fit an imputer for entirely missing item {item}"
                )
            strategy = "most_frequent" if metadata["type"] == "nominal" else "median"
            if strategy == "most_frequent":
                statistic = float(observed.mode(dropna=True).iloc[0])
            else:
                statistic = float(observed.median())
            self.statistics_[item] = statistic
            self.strategies_[item] = strategy
            if values.isna().mean() > self.missing_indicator_threshold:
                indicator_items.append(item)

        self.indicator_items_ = tuple(indicator_items)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "statistics_"):
            raise RuntimeError("ItemImputer must be fitted before transform")
        missing_items = set(self.item_names_) - set(X.columns)
        if missing_items:
            raise ValueError(f"Missing questionnaire items: {sorted(missing_items)}")
        result = X.copy()
        for item in self.item_names_:
            values = pd.to_numeric(result[item], errors="coerce")
            if item in self.indicator_items_:
                result[f"missing__{item}"] = values.isna().astype("int8")
            result[item] = values.fillna(self.statistics_[item]).astype(float)
        return result

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError(
                "ItemImputer must be fitted before requesting feature names"
            )
        names = list(
            self.feature_names_in_ if input_features is None else input_features
        )
        names.extend(f"missing__{item}" for item in self.indicator_items_)
        return np.asarray(names, dtype=object)


class ItemEncoder(BaseEstimator, TransformerMixin):
    """Rescale ordered responses, one-hot nominal responses, and retain audit flags."""

    def __init__(
        self,
        codec: EnvFinCodec | None = None,
        schema_path: str | None = None,
        mapping_path: str | None = None,
    ) -> None:
        self.codec = codec
        self.schema_path = schema_path
        self.mapping_path = mapping_path

    def _get_codec(self) -> EnvFinCodec:
        if self.codec is not None:
            return self.codec
        return _codec_from_paths(self.schema_path, self.mapping_path)

    def fit(self, X: pd.DataFrame, y: Any = None) -> ItemEncoder:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ItemEncoder requires a pandas DataFrame")
        self.codec_ = self._get_codec()
        self.item_names_ = tuple(self.codec_.items)
        missing_items = set(self.item_names_) - set(X.columns)
        if missing_items:
            raise ValueError(f"Missing questionnaire items: {sorted(missing_items)}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.nominal_items_ = tuple(
            item
            for item, metadata in self.codec_.items.items()
            if metadata["type"] == "nominal"
        )
        self.ordinal_ranges_: dict[str, tuple[float, float, bool, float | None]] = {}
        self.pressure_maps_: dict[str, dict[float, float]] = {}
        for item, metadata in self.codec_.items.items():
            codes = [float(level["code"]) for level in metadata["levels"]]
            if metadata["type"] != "nominal":
                self.ordinal_ranges_[item] = (
                    min(codes),
                    max(codes),
                    metadata.get("pressure_direction") == "reversed",
                    (
                        float(metadata["structural_na"]["code"])
                        if metadata.get("structural_na") is not None
                        else None
                    ),
                )
            else:
                self.pressure_maps_[item] = self._nominal_pressure_map(item, metadata)

        categories = []
        for item in self.nominal_items_:
            metadata = self.codec_.items[item]
            codes = [int(level["code"]) for level in metadata["levels"]]
            special = metadata.get("structural_na")
            if special is not None:
                codes.append(int(special["code"]))
            categories.append(np.asarray(sorted(set(codes)), dtype=float))
        self.one_hot_encoder_ = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
        if self.nominal_items_:
            self.one_hot_encoder_.fit(X.loc[:, self.nominal_items_])
        self.encoded_names_ = tuple(
            self.one_hot_encoder_.get_feature_names_out(self.nominal_items_)
        )
        self.passthrough_names_ = tuple(
            column
            for column in X.columns
            if column not in self.item_names_
            and column not in _NON_FEATURE_COLUMNS
            and column.startswith(("is_", "pns_", "flag_", "missing__"))
        )
        return self

    def _nominal_pressure_map(
        self, item: str, metadata: dict[str, Any]
    ) -> dict[float, float]:
        configured = {
            re.sub(r"\s+", " ", key.replace("_", " ").casefold()).strip(): float(value)
            for key, value in metadata.get("pressure_map", {}).items()
            if key != "_note"
        }
        code_scores: dict[float, float] = {}
        for level in metadata["levels"]:
            label = re.sub(
                r"[^\w\s]",
                " ",
                self.codec_.normalize_text(level["label"]).casefold(),
            )
            label = re.sub(r"\s+", " ", label).strip()
            label_tokens = set(label.split())
            matches = [
                (len(key), score)
                for key, score in configured.items()
                if key in label or set(key.split()).issubset(label_tokens)
            ]
            if not matches:
                raise ValueError(
                    f"No configured pressure-map value for {item}: {label}"
                )
            code_scores[float(level["code"])] = max(matches)[1]
        special = metadata.get("structural_na")
        if special is not None:
            special_key = "not applicable"
            special_scores = [
                score for key, score in configured.items() if special_key in key
            ]
            if special_scores:
                code_scores[float(special["code"])] = special_scores[0]
            else:
                code_scores[float(special["code"])] = 0.0
        return code_scores

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "ordinal_ranges_"):
            raise RuntimeError("ItemEncoder must be fitted before transform")
        missing_items = set(self.item_names_) - set(X.columns)
        if missing_items:
            raise ValueError(f"Missing questionnaire items: {sorted(missing_items)}")
        encoded = pd.DataFrame(index=X.index)

        for item, (
            minimum,
            maximum,
            reversed_pressure,
            structural_code,
        ) in self.ordinal_ranges_.items():
            values = pd.to_numeric(X[item], errors="coerce")
            scale = maximum - minimum
            if scale == 0:
                pressure = pd.Series(0.0, index=X.index)
            elif reversed_pressure:
                pressure = (maximum - values) / scale
            else:
                pressure = (values - minimum) / scale
            if structural_code is not None:
                pressure = pressure.mask(values.eq(structural_code), 0.0)
            encoded[f"{item}__pressure"] = pressure

        if self.nominal_items_:
            nominal_values = X.loc[:, self.nominal_items_]
            one_hot = self.one_hot_encoder_.transform(nominal_values)
            encoded = pd.concat(
                [
                    encoded,
                    pd.DataFrame(
                        one_hot,
                        index=X.index,
                        columns=self.encoded_names_,
                    ),
                ],
                axis=1,
            )
            for item, pressure_by_code in self.pressure_maps_.items():
                values = pd.to_numeric(X[item], errors="coerce")
                encoded[f"{item}__pressure"] = values.map(pressure_by_code)

        for column in self.passthrough_names_:
            encoded[column] = X[column]
        for item in _INTERMEDIATE_CODE_ITEMS:
            encoded[f"{item}__code"] = pd.to_numeric(X[item], errors="coerce")
        return encoded

    def fit_transform(
        self, X: pd.DataFrame, y: Any = None, **fit_params: Any
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "encoded_names_"):
            raise RuntimeError(
                "ItemEncoder must be fitted before requesting feature names"
            )
        ordinal = [
            f"{item}__pressure"
            for item, metadata in self.codec_.items.items()
            if metadata["type"] != "nominal"
        ]
        nominal_pressure = [f"{item}__pressure" for item in self.nominal_items_]
        code_names = [f"{item}__code" for item in _INTERMEDIATE_CODE_ITEMS]
        return np.asarray(
            [
                *ordinal,
                *self.encoded_names_,
                *nominal_pressure,
                *self.passthrough_names_,
                *code_names,
            ],
            dtype=object,
        )


class IndexBuilder(BaseEstimator, TransformerMixin):
    """Build schema-defined pressure indices and configured ratio/interactions."""

    def __init__(
        self,
        codec: EnvFinCodec | None = None,
        settings: dict[str, Any] | None = None,
        schema_path: str | None = None,
        project_config_path: str | None = None,
    ) -> None:
        self.codec = codec
        self.settings = settings
        self.schema_path = schema_path
        self.project_config_path = project_config_path

    def _get_codec(self) -> EnvFinCodec:
        if self.codec is not None:
            return self.codec
        return _codec_from_paths(self.schema_path, None)

    def _get_settings(self) -> dict[str, Any]:
        if self.settings is not None:
            return self.settings
        config_path = (
            Path(self.project_config_path)
            if self.project_config_path is not None
            else get_project_root() / "config" / "project.yaml"
        )
        return load_yaml(config_path).get("feature_engineering", {})

    def fit(self, X: pd.DataFrame, y: Any = None) -> IndexBuilder:
        self.codec_ = self._get_codec()
        self.settings_ = self._get_settings()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.index_columns_ = tuple(
            f"{name}_index" for name in self.codec_.derived_indices
        )
        return self

    @staticmethod
    def _interpolate_midpoints(
        codes: pd.Series, midpoint_config: dict[Any, Any]
    ) -> pd.Series:
        code_midpoints = sorted(
            (float(code), float(value)) for code, value in midpoint_config.items()
        )
        if not code_midpoints:
            raise ValueError("Monetary midpoint configuration cannot be empty")
        code_values, monetary_values = map(np.asarray, zip(*code_midpoints))
        numeric_codes = pd.to_numeric(codes, errors="coerce")
        interpolated = np.interp(
            numeric_codes.to_numpy(dtype=float),
            code_values,
            monetary_values,
            left=np.nan,
            right=np.nan,
        )
        return pd.Series(interpolated, index=codes.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "settings_"):
            raise RuntimeError("IndexBuilder must be fitted before transform")
        result = X.copy()
        for name, metadata in self.codec_.derived_indices.items():
            pressure_columns = [f"{item}__pressure" for item in metadata["items"]]
            missing = set(pressure_columns) - set(result.columns)
            if missing:
                raise ValueError(
                    f"Inputs are missing pressure values for {name}: {sorted(missing)}"
                )
            values = result[pressure_columns].mean(axis=1, skipna=True)
            if name == "commute_burden" and "is_online" in result.columns:
                values = values.mask(result["is_online"].astype(bool), 0.0)
            result[f"{name}_index"] = values.clip(0.0, 1.0)

        result["overall_pressure_index"] = (
            result[list(self.index_columns_)].mean(axis=1, skipna=True).clip(0.0, 1.0)
        )
        threshold = float(self.settings_["index_high_pressure_threshold"])
        result["indices_above_066_count"] = (
            result[list(self.index_columns_)].gt(threshold).sum(axis=1).astype("int8")
        )

        income_amount = self._interpolate_midpoints(
            result["income_band__code"],
            self.settings_["income_band_monthly_midpoints_lkr"],
        )
        commute_amount = self._interpolate_midpoints(
            result["commute_spend__code"],
            self.settings_["commute_spend_monthly_midpoints_lkr"],
        )
        ratio = commute_amount.div(income_amount.where(income_amount > 0))
        if "is_online" in result.columns:
            ratio = ratio.mask(result["is_online"].astype(bool), 0.0)
        result["travel_spend_to_income_ratio"] = ratio.clip(0.0, 1.0)

        result["low_income_no_aid"] = (
            result["income_band__code"].le(float(self.settings_["low_income_max_code"]))
            & result["aid_status__code"].eq(
                max(
                    float(level["code"])
                    for level in self.codec_.items["aid_status"]["levels"]
                )
            )
        ).astype("int8")
        result["working_over_20h_long_commute"] = (
            result["part_time_job__code"].gt(0)
            & result["job_hours__code"].ge(
                float(self.settings_["work_hours_over_20_min_code"])
            )
            & result["commute_time__code"].ge(
                float(self.settings_["long_commute_min_code"])
            )
        ).astype("int8")
        result["weak_internet_smartphone_only"] = (
            result["home_internet__code"].ge(
                float(self.settings_["weak_internet_min_code"])
            )
            & result["main_device__code"].eq(
                float(self.settings_["smartphone_only_code"])
            )
        ).astype("int8")

        result = result.drop(
            columns=[f"{item}__code" for item in _INTERMEDIATE_CODE_ITEMS],
            errors="ignore",
        )
        return result

    def fit_transform(
        self, X: pd.DataFrame, y: Any = None, **fit_params: Any
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "settings_"):
            raise RuntimeError(
                "IndexBuilder must be fitted before requesting feature names"
            )
        input_names = (
            self.feature_names_in_ if input_features is None else input_features
        )
        dropped = {f"{item}__code" for item in _INTERMEDIATE_CODE_ITEMS}
        added = [
            *self.index_columns_,
            "overall_pressure_index",
            "indices_above_066_count",
            "travel_spend_to_income_ratio",
            "low_income_no_aid",
            "working_over_20h_long_commute",
            "weak_internet_smartphone_only",
        ]
        return np.asarray(
            [column for column in input_names if column not in dropped] + added,
            dtype=object,
        )


class ProfileFeatureEncoder(BaseEstimator, TransformerMixin):
    """Fit profile memberships on the current training partition only."""

    def __init__(
        self,
        n_profiles: int = 4,
        random_state: int = 42,
        index_columns: tuple[str, ...] = (
            "financial_stress_index",
            "commute_burden_index",
            "digital_access_gap_index",
            "study_environment_pressure_index",
        ),
    ) -> None:
        self.n_profiles = n_profiles
        self.random_state = random_state
        self.index_columns = index_columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> ProfileFeatureEncoder:
        missing = set(self.index_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Profile inputs are missing indices: {sorted(missing)}")
        if len(X) < self.n_profiles:
            raise ValueError(
                f"At least {self.n_profiles} rows are required to fit profile groups"
            )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.clusterer_ = KMeans(
            n_clusters=self.n_profiles,
            random_state=self.random_state,
            n_init=10,
        ).fit(X.loc[:, self.index_columns])
        self.profile_feature_names_ = tuple(
            f"profile_cluster_{index}" for index in range(self.n_profiles)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "clusterer_"):
            raise RuntimeError("ProfileFeatureEncoder must be fitted before transform")
        labels = self.clusterer_.predict(X.loc[:, self.index_columns])
        membership = pd.DataFrame(
            np.eye(self.n_profiles, dtype=np.int8)[labels],
            index=X.index,
            columns=self.profile_feature_names_,
        )
        return pd.concat([X.copy(), membership], axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "profile_feature_names_"):
            raise RuntimeError(
                "ProfileFeatureEncoder must be fitted before requesting feature names"
            )
        input_names = (
            self.feature_names_in_ if input_features is None else input_features
        )
        return np.asarray(
            [*input_names, *self.profile_feature_names_],
            dtype=object,
        )


class FeatureSetSelector(BaseEstimator, TransformerMixin):
    """Select configured raw, aggregated, profile, or income-only columns."""

    def __init__(
        self,
        feature_set: str,
        model_zoo_path: str | None = None,
    ) -> None:
        self.feature_set = feature_set
        self.model_zoo_path = model_zoo_path

    def _load_model_zoo(self) -> dict[str, Any]:
        path = (
            Path(self.model_zoo_path)
            if self.model_zoo_path is not None
            else get_project_root() / "config" / "model_zoo.yaml"
        )
        return load_yaml(path)

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureSetSelector:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureSetSelector requires a pandas DataFrame")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.model_zoo_ = self._load_model_zoo()
        sets = self.model_zoo_.get("feature_sets", {})
        if self.feature_set not in sets:
            raise ValueError(f"Unknown feature set: {self.feature_set}")
        specification = sets[self.feature_set]
        if self.feature_set == "single_signal":
            selected = specification["features"]
        else:
            raw = sets["raw"]
            prefixes = tuple(raw["nominal_prefixes"] + raw["flag_prefixes"])
            selected = [
                column
                for column in X.columns
                if column.endswith(raw["pressure_suffix"])
                or column.startswith(prefixes)
            ]
            if self.feature_set in {"agg", "agg_profiles"}:
                selected.extend(sets["agg"]["engineered_features"])
            if self.feature_set == "agg_profiles":
                profile_prefix = specification["profile_prefix"]
                selected.extend(
                    column for column in X.columns if column.startswith(profile_prefix)
                )
        missing = set(selected) - set(X.columns)
        if missing:
            raise ValueError(
                f"Feature set {self.feature_set} requests missing columns: "
                f"{sorted(missing)}"
            )
        self.selected_features_ = tuple(dict.fromkeys(selected))
        if not self.selected_features_:
            raise ValueError(f"Feature set {self.feature_set} selected no columns")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("FeatureSetSelector must be fitted before transform")
        missing = set(self.selected_features_) - set(X.columns)
        if missing:
            raise ValueError(f"Transform input is missing columns: {sorted(missing)}")
        return X.loc[:, self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError(
                "FeatureSetSelector must be fitted before requesting feature names"
            )
        return np.asarray(self.selected_features_, dtype=object)


class NearConstantFeatureRemover(BaseEstimator, TransformerMixin):
    """Remove features with training-partition variance at or below a threshold."""

    def __init__(self, variance_threshold: float = 0.005) -> None:
        self.variance_threshold = variance_threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> NearConstantFeatureRemover:
        if self.variance_threshold < 0:
            raise ValueError("variance_threshold must be non-negative")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        variances = X.astype(float).var(axis=0, ddof=0)
        self.selected_features_ = tuple(
            variances[variances > self.variance_threshold].index
        )
        self.removed_features_ = tuple(
            variances[variances <= self.variance_threshold].index
        )
        if not self.selected_features_:
            raise ValueError(
                "Near-constant removal would eliminate every selected feature"
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError(
                "NearConstantFeatureRemover must be fitted before transform"
            )
        return X.loc[:, self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError(
                "NearConstantFeatureRemover must be fitted before requesting feature names"
            )
        return np.asarray(self.selected_features_, dtype=object)


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Apply a training-fitted standard scaler while retaining data-frame columns."""

    def __init__(self, method: str = "standard") -> None:
        self.method = method

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureScaler:
        if self.method != "standard":
            raise ValueError(f"Unsupported feature scaling method: {self.method}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X.astype(float))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "scaler_"):
            raise RuntimeError("FeatureScaler must be fitted before transform")
        transformed = self.scaler_.transform(
            X.loc[:, self.feature_names_in_].astype(float)
        )
        return pd.DataFrame(
            transformed,
            index=X.index,
            columns=self.feature_names_in_,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError(
                "FeatureScaler must be fitted before requesting feature names"
            )
        return np.asarray(self.feature_names_in_, dtype=object)


def build_feature_pipeline(
    feature_set: str,
    model_name: str,
    codec: EnvFinCodec | None = None,
    model_zoo_path: str | None = None,
) -> Any:
    """Build the complete fold-fitted sklearn pipeline for a registered model."""
    from sklearn.pipeline import Pipeline

    config_path = (
        Path(model_zoo_path)
        if model_zoo_path is not None
        else get_project_root() / "config" / "model_zoo.yaml"
    )
    model_zoo = load_yaml(config_path)
    registry = model_zoo.get("model_registry", {})
    if model_name not in registry:
        raise ValueError(f"Unknown model registry name: {model_name}")
    model_config = registry[model_name]
    preprocessing = model_zoo.get("preprocessing", {})
    steps = [
        ("imputer", ItemImputer(codec=codec)),
        ("encoder", ItemEncoder(codec=codec)),
        ("indices", IndexBuilder(codec=codec)),
    ]
    feature_set_config = model_zoo["feature_sets"].get(feature_set)
    if feature_set_config is None:
        raise ValueError(f"Unknown feature set: {feature_set}")
    if feature_set == "agg_profiles":
        steps.append(
            (
                "profiles",
                ProfileFeatureEncoder(
                    n_profiles=int(feature_set_config["profile_count"]),
                    random_state=42,
                ),
            )
        )
    steps.extend(
        [
            ("select", FeatureSetSelector(feature_set, str(config_path))),
            (
                "near_constant",
                NearConstantFeatureRemover(
                    float(preprocessing["near_constant_variance_threshold"])
                ),
            ),
        ]
    )
    if model_config["family"] in preprocessing["scaled_model_families"]:
        steps.append(
            (
                "scale",
                FeatureScaler(method=preprocessing["scaling_method"]),
            )
        )
    return Pipeline(steps)
