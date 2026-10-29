"""Fold-fitted imputation and questionnaire item encoding."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from src.preprocessing.codec import EnvFinCodec

_NON_FEATURE_COLUMNS = {
    "student_id",
    "timestamp",
    "consent",
    "attention_check",
    "attrition_intent_score",
    "at_risk",
    "row_reasons",
}


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
        return np.asarray(
            [
                *ordinal,
                *self.encoded_names_,
                *nominal_pressure,
                *self.passthrough_names_,
            ],
            dtype=object,
        )
