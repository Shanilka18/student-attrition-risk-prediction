"""Tests for fold-fitted item imputation and encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.preprocessing.codec import EnvFinCodec
from src.preprocessing.feature_transformers import (
    IndexBuilder,
    ItemEncoder,
    ItemImputer,
)


@pytest.fixture
def codec():
    return EnvFinCodec()


def _frame(codec, count=100):
    values = {}
    for item, metadata in codec.items.items():
        codes = [level["code"] for level in metadata["levels"]]
        values[item] = np.resize(codes, count)
    values["is_online"] = np.zeros(count, dtype=bool)
    values["pns_cost_difficulty"] = np.zeros(count, dtype=bool)
    values["flag_commute_partial_na"] = np.zeros(count, dtype=bool)
    values["student_id"] = [f"S{index:05d}" for index in range(count)]
    values["at_risk"] = np.arange(count) % 2
    return pd.DataFrame(values)


def test_imputer_learns_from_fit_rows_and_adds_thresholded_indicators(codec):
    train = _frame(codec)
    train.loc[:9, "cost_difficulty"] = np.nan
    train.loc[:2, "money_worry"] = np.nan
    imputer = ItemImputer(codec=codec, missing_indicator_threshold=0.03).fit(train)

    test = _frame(codec, 4)
    test.loc[0, "cost_difficulty"] = np.nan
    test.loc[0, "money_worry"] = np.nan
    transformed = imputer.transform(test)

    assert "missing__cost_difficulty" in transformed
    assert "missing__money_worry" not in transformed
    assert transformed.loc[0, "missing__cost_difficulty"] == 1
    assert (
        transformed.loc[0, "cost_difficulty"] == imputer.statistics_["cost_difficulty"]
    )
    assert pd.notna(transformed.loc[0, "money_worry"])
    assert imputer.strategies_["income_band"] == "median"
    assert imputer.strategies_["main_device"] == "most_frequent"


def test_item_encoder_rescales_direction_and_keeps_flags_only(codec):
    train = _frame(codec)
    fitted = clone(ItemEncoder(codec=codec)).fit(train)
    transformed = fitted.transform(train.iloc[:5])

    assert transformed.loc[0, "income_band__pressure"] == 1.0
    assert transformed.loc[0, "aid_status__pressure"] == 0.0
    assert transformed.loc[0, "main_device__pressure"] == pytest.approx(0.1)
    assert "main_device_1" in transformed.columns
    assert "is_online" in transformed.columns
    assert "pns_cost_difficulty" in transformed.columns
    assert "flag_commute_partial_na" in transformed.columns
    assert "at_risk" not in transformed.columns
    assert "student_id" not in transformed.columns


def test_structural_commute_state_is_encoded_as_zero_pressure(codec):
    frame = _frame(codec, 4)
    for item in ("commute_time", "commute_spend", "transport_mode"):
        frame[item] = codec.items[item]["structural_na"]["code"]
    transformed = ItemEncoder(codec=codec).fit_transform(frame)

    assert transformed["commute_time__pressure"].eq(0.0).all()
    assert transformed["commute_spend__pressure"].eq(0.0).all()
    assert transformed["transport_mode__pressure"].eq(0.0).all()


def test_index_builder_creates_indices_ratio_and_configured_interactions(codec):
    frame = _frame(codec, 4)
    frame.loc[0, "income_band"] = 0
    frame.loc[0, "aid_status"] = 2
    frame.loc[0, "part_time_job"] = 2
    frame.loc[0, "job_hours"] = 3
    frame.loc[0, "commute_time"] = 2
    frame.loc[0, "commute_spend"] = 3
    frame.loc[0, "home_internet"] = 2
    frame.loc[0, "main_device"] = 1
    frame.loc[1, "commute_time"] = -2
    frame.loc[1, "commute_spend"] = -2
    frame.loc[1, "transport_mode"] = -2
    frame.loc[1, "is_online"] = True

    encoded = ItemEncoder(codec=codec).fit_transform(
        ItemImputer(codec=codec).fit_transform(frame)
    )
    result = IndexBuilder(codec=codec).fit_transform(encoded)
    index_columns = [
        "financial_stress_index",
        "commute_burden_index",
        "digital_access_gap_index",
        "study_environment_pressure_index",
    ]

    assert result[index_columns].ge(0.0).all().all()
    assert result[index_columns].le(1.0).all().all()
    assert result.loc[0, "travel_spend_to_income_ratio"] == pytest.approx(1.0)
    assert result.loc[0, "low_income_no_aid"] == 1
    assert result.loc[0, "working_over_20h_long_commute"] == 1
    assert result.loc[0, "weak_internet_smartphone_only"] == 1
    assert result.loc[1, "commute_burden_index"] == 0.0
    assert result.loc[1, "travel_spend_to_income_ratio"] == 0.0
    assert result.loc[1, "working_over_20h_long_commute"] == 0
    assert not any(column.endswith("__code") for column in result.columns)
