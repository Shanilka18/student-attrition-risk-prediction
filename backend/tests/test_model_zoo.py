"""
Phase 4 Model Zoo Tests — Academic Intelligence Framework (IT23164680).

Tests:
  - Registry completeness: all categories A-F present, ≥27 models
  - Proposal models correctly marked
  - All tabular models: smoke fit + predict_proba shape/range
  - All sequence models: smoke fit + predict_proba shape/range
  - Save / load roundtrip (key representative models)
  - GPAThreshold rule-based model (no sklearn fit needed)
  - Sequence input: dict payload AND raw tensor both accepted
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# ── Bootstrap registry ────────────────────────────────────────────────────────
import backend.src.models  # noqa: F401 — triggers all @register decorators
from backend.src.models.base import ModelRegistry, CATEGORIES


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES = 60
N_TAB_FEATURES = 22
N_STATIC_FEATURES = 2
N_SEQ_TIMESTEPS = 6
N_SEQ_FEATURES = 4
SEED = 42


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture(scope="module")
def tabular_data(rng):
    """Small balanced tabular dataset for Category A/B/C/D smoke tests."""
    X = rng.standard_normal((N_SAMPLES, N_TAB_FEATURES)).astype(np.float32)
    y = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2), dtype=np.float32)
    rng.shuffle(y)
    return X, y


@pytest.fixture(scope="module")
def static_data(rng):
    """2-feature static snapshot dataset for Category F smoke tests."""
    X = rng.standard_normal((N_SAMPLES, N_STATIC_FEATURES)).astype(np.float32)
    y = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2), dtype=np.float32)
    rng.shuffle(y)
    return X, y


@pytest.fixture(scope="module")
def sequence_dict_data(rng):
    """Dict-payload sequence data (mimics Phase 3 sequence_tensors.pt format)."""
    tensors = torch.tensor(
        rng.standard_normal((N_SAMPLES, N_SEQ_TIMESTEPS, N_SEQ_FEATURES)),
        dtype=torch.float32,
    )
    # Simulate variable lengths: 3 to 6 semesters
    lengths = torch.randint(3, N_SEQ_TIMESTEPS + 1, (N_SAMPLES,))
    mask = torch.zeros(N_SAMPLES, N_SEQ_TIMESTEPS, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    y = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2), dtype=np.float32)
    rng.shuffle(y)
    return {"tensors": tensors, "padding_mask": mask, "seq_lengths": lengths}, y


@pytest.fixture(scope="module")
def sequence_raw_data(rng):
    """Raw tensor sequence data (used for minimal smoke tests)."""
    tensors = torch.tensor(
        rng.standard_normal((N_SAMPLES, N_SEQ_TIMESTEPS, N_SEQ_FEATURES)),
        dtype=torch.float32,
    )
    y = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2), dtype=np.float32)
    rng.shuffle(y)
    return tensors, y


# ─────────────────────────────────────────────────────────────────────────────
# 1. Registry completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_all_categories_present(self):
        """Categories A through F must all contain at least one model."""
        by_cat = ModelRegistry.list_by_category()
        for cat in ("A", "B", "C", "D", "E", "F"):
            assert cat in by_cat, f"Category {cat} missing from registry"
            assert len(by_cat[cat]) >= 1, f"Category {cat} has no models"

    def test_minimum_model_count(self):
        """Registry must contain at least 27 models."""
        all_models = ModelRegistry.list_all()
        assert len(all_models) >= 27, (
            f"Expected ≥27 models, got {len(all_models)}: {[m['key'] for m in all_models]}"
        )

    def test_category_keys_match_categories_dict(self):
        """Every registered model's category must be in the CATEGORIES dict."""
        for row in ModelRegistry.list_all():
            assert row["category"] in CATEGORIES, (
                f"Model '{row['key']}' has unknown category '{row['category']}'"
            )

    def test_keys_method(self):
        """ModelRegistry.keys() returns a non-empty list of strings."""
        keys = ModelRegistry.keys()
        assert isinstance(keys, list)
        assert len(keys) >= 27
        assert all(isinstance(k, str) and k for k in keys)

    def test_build_returns_instance(self):
        """ModelRegistry.build() instantiates a model with merged params."""
        model = ModelRegistry.build("logistic_l2")
        from backend.src.models.base import BaseAcademicModel
        assert isinstance(model, BaseAcademicModel)

    def test_build_unknown_key_raises(self):
        """Building an unknown key must raise KeyError."""
        with pytest.raises(KeyError):
            ModelRegistry.build("definitely_not_a_real_model_key")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Proposal models correctly marked
# ─────────────────────────────────────────────────────────────────────────────

class TestProposalModels:
    EXPECTED_PROPOSAL_KEYS = {"xgboost", "lightgbm", "lstm", "gru"}

    def test_proposal_models_are_marked(self):
        """LSTM, GRU, XGBoost, LightGBM must have IS_PROPOSAL_MODEL=True."""
        for key in self.EXPECTED_PROPOSAL_KEYS:
            model_cls = ModelRegistry._registry[key]
            assert model_cls.IS_PROPOSAL_MODEL is True, (
                f"Expected {key}.IS_PROPOSAL_MODEL=True"
            )

    def test_non_proposal_model_not_marked(self):
        """A non-proposal model (e.g. knn) must NOT be marked as proposal."""
        knn_cls = ModelRegistry._registry["knn"]
        assert knn_cls.IS_PROPOSAL_MODEL is False

    def test_proposal_models_in_correct_categories(self):
        """XGBoost/LightGBM must be Category C; LSTM/GRU must be Category E."""
        assert ModelRegistry._registry["xgboost"].CATEGORY == "C"
        assert ModelRegistry._registry["lightgbm"].CATEGORY == "C"
        assert ModelRegistry._registry["lstm"].CATEGORY == "E"
        assert ModelRegistry._registry["gru"].CATEGORY == "E"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tabular models smoke test (Categories A, B, C, D)
# ─────────────────────────────────────────────────────────────────────────────

TABULAR_MODEL_KEYS = [
    # Category A
    "logistic_l2", "logistic_elasticnet", "svm_calibrated",
    "gaussian_nb", "knn",
    # Category B
    "decision_tree", "random_forest", "extra_trees",
    # Category C
    "gradient_boosting_sklearn", "xgboost", "lightgbm", "catboost", "adaboost",
    # Category D
    "mlp_tabular",
]


@pytest.mark.parametrize("key", TABULAR_MODEL_KEYS)
class TestTabularModelSmoke:
    def test_fit_predict_proba(self, key, tabular_data):
        """Fit for 1 iteration then predict_proba: shape=(N,) and values in [0,1]."""
        X, y = tabular_data
        # Use very fast params to keep test quick
        fast_params: dict = {"n_estimators": 2, "max_iter": 3, "epochs": 1, "n_neighbors": 3}
        model = ModelRegistry.build(key, params=fast_params)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (N_SAMPLES,), (
            f"{key}: expected shape ({N_SAMPLES},), got {probs.shape}"
        )
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0), (
            f"{key}: probabilities out of [0,1] range. min={probs.min():.4f}, max={probs.max():.4f}"
        )

    def test_is_fitted_flag(self, key, tabular_data):
        """After fit(), is_fitted must be True."""
        X, y = tabular_data
        model = ModelRegistry.build(key, params={"n_estimators": 2, "max_iter": 3, "epochs": 1})
        assert model.is_fitted is False
        model.fit(X, y)
        assert model.is_fitted is True

    def test_predict_binary(self, key, tabular_data):
        """predict() must return only 0/1 values."""
        X, y = tabular_data
        model = ModelRegistry.build(key, params={"n_estimators": 2, "max_iter": 3, "epochs": 1})
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1}), (
            f"{key}: predict() returned non-binary values: {np.unique(preds)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Category F (static snapshot) smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestStaticSnapshotModels:
    def test_static_logistic_smoke(self, static_data):
        X, y = static_data
        model = ModelRegistry.build("static_logistic", params={"max_iter": 3})
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (N_SAMPLES,)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_static_lightgbm_smoke(self, static_data):
        X, y = static_data
        model = ModelRegistry.build("static_lightgbm", params={"n_estimators": 2})
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (N_SAMPLES,)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_static_models_use_category_f(self):
        assert ModelRegistry._registry["static_logistic"].CATEGORY == "F"
        assert ModelRegistry._registry["static_lightgbm"].CATEGORY == "F"


# ─────────────────────────────────────────────────────────────────────────────
# 5. GPAThreshold rule-based model
# ─────────────────────────────────────────────────────────────────────────────

class TestGPAThreshold:
    def test_no_fit_required(self, tabular_data):
        """GPAThreshold must work without any training data (rule-based)."""
        X, y = tabular_data
        model = ModelRegistry.build("gpa_threshold")
        # NOT calling fit at all — must still produce predictions
        probs = model.predict_proba(X)
        assert probs.shape == (N_SAMPLES,)
        assert set(np.unique(probs)).issubset({0.0, 1.0}), (
            "GPAThreshold should return hard 0/1 probabilities"
        )

    def test_fit_does_not_raise(self, tabular_data):
        """Calling fit on GPAThreshold must not raise — it's a no-op."""
        X, y = tabular_data
        model = ModelRegistry.build("gpa_threshold")
        model.fit(X, y)  # should not raise

    def test_gpa_threshold_category(self):
        assert ModelRegistry._registry["gpa_threshold"].CATEGORY == "A"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Sequence models smoke test (Category E)
# ─────────────────────────────────────────────────────────────────────────────

SEQUENCE_MODEL_KEYS = [
    "vanilla_rnn", "lstm", "gru", "bilstm", "bigru",
    "lstm_attention", "gru_attention",
    "cnn_1d", "cnn_lstm", "tcn", "transformer_enc",
]

# Fast params: 1 epoch, small hidden, no early stopping
SEQ_FAST_PARAMS = {
    "epochs": 1,
    "batch_size": 32,
    "hidden_size": 16,
    "lstm_hidden": 16,
    "num_layers": 1,
    "d_model": 16,
    "nhead": 2,
    "dim_feedforward": 32,
    "num_filters": 16,
    "num_channels": [16, 16],
    "dropout": 0.0,
    "early_stopping_patience": 999,
}


@pytest.mark.parametrize("key", SEQUENCE_MODEL_KEYS)
class TestSequenceModelSmoke:
    def test_fit_predict_dict_payload(self, key, sequence_dict_data):
        """Fit 1 epoch with dict payload (Phase 3 format), predict, check shape/range."""
        X_dict, y = sequence_dict_data
        model = ModelRegistry.build(key, params=SEQ_FAST_PARAMS)
        model.fit(X_dict, y)
        probs = model.predict_proba(X_dict)
        assert probs.shape == (N_SAMPLES,), (
            f"{key}: expected ({N_SAMPLES},), got {probs.shape}"
        )
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0), (
            f"{key}: probs out of [0,1]. min={probs.min():.4f}, max={probs.max():.4f}"
        )

    def test_fit_predict_raw_tensor(self, key, sequence_raw_data):
        """Fit 1 epoch with raw tensor (unit test shortcut), predict, check shape/range."""
        X_raw, y = sequence_raw_data
        model = ModelRegistry.build(key, params=SEQ_FAST_PARAMS)
        model.fit(X_raw, y)
        probs = model.predict_proba(X_raw)
        assert probs.shape == (N_SAMPLES,)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_is_sequence_model_flag(self, key):
        """Category E models must have IS_SEQUENCE_MODEL=True."""
        assert ModelRegistry._registry[key].IS_SEQUENCE_MODEL is True

    def test_training_history_recorded(self, key, sequence_raw_data):
        """After fit(), training_history must have at least 1 entry."""
        X_raw, y = sequence_raw_data
        model = ModelRegistry.build(key, params=SEQ_FAST_PARAMS)
        model.fit(X_raw, y)
        assert len(model.training_history) >= 1
        entry = model.training_history[0]
        assert "epoch" in entry
        assert "train_loss" in entry
        assert isinstance(entry["train_loss"], float)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Save / Load roundtrip
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveLoadRoundtrip:
    def _assert_probs_close(self, p1, p2, key):
        np.testing.assert_allclose(
            p1, p2, atol=1e-5,
            err_msg=f"{key}: predict_proba differs after save/load"
        )

    def test_logistic_l2_roundtrip(self, tabular_data, tmp_path):
        X, y = tabular_data
        model = ModelRegistry.build("logistic_l2", params={"max_iter": 3})
        model.fit(X, y)
        p_before = model.predict_proba(X)
        model.save(tmp_path)
        from backend.src.models.tabular_models import LogisticL2Model
        loaded = LogisticL2Model.load(tmp_path)
        p_after = loaded.predict_proba(X)
        self._assert_probs_close(p_before, p_after, "logistic_l2")

    def test_xgboost_roundtrip(self, tabular_data, tmp_path):
        X, y = tabular_data
        model = ModelRegistry.build("xgboost", params={"n_estimators": 3})
        model.fit(X, y)
        p_before = model.predict_proba(X)
        model.save(tmp_path)
        from backend.src.models.tabular_models import XGBoostModel
        loaded = XGBoostModel.load(tmp_path)
        p_after = loaded.predict_proba(X)
        self._assert_probs_close(p_before, p_after, "xgboost")

    def test_lstm_roundtrip(self, sequence_raw_data, tmp_path):
        X_raw, y = sequence_raw_data
        model = ModelRegistry.build("lstm", params=SEQ_FAST_PARAMS)
        model.fit(X_raw, y)
        p_before = model.predict_proba(X_raw)
        model.save(tmp_path)
        from backend.src.models.sequence_models import LSTMModel
        loaded = LSTMModel.load(tmp_path)
        p_after = loaded.predict_proba(X_raw)
        self._assert_probs_close(p_before, p_after, "lstm")

    def test_save_writes_meta_json(self, tabular_data, tmp_path):
        X, y = tabular_data
        model = ModelRegistry.build("random_forest", params={"n_estimators": 2})
        model.fit(X, y)
        model.save(tmp_path)
        meta_path = tmp_path / "random_forest_meta.json"
        assert meta_path.exists(), "save() must write a _meta.json file"
        meta = json.loads(meta_path.read_text())
        assert meta["model_key"] == "random_forest"
        assert meta["category"] == "B"


# ─────────────────────────────────────────────────────────────────────────────
# 8. metadata() contract
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataContract:
    REQUIRED_KEYS = {
        "model_key", "category", "category_label", "label",
        "is_proposal_model", "is_sequence_model", "is_fitted",
        "fit_time_seconds", "params",
    }

    def test_metadata_has_required_keys(self, tabular_data):
        X, y = tabular_data
        model = ModelRegistry.build("lightgbm", params={"n_estimators": 2})
        model.fit(X, y)
        meta = model.metadata()
        missing = self.REQUIRED_KEYS - set(meta.keys())
        assert not missing, f"metadata() missing keys: {missing}"

    def test_metadata_is_json_serialisable(self, tabular_data):
        X, y = tabular_data
        model = ModelRegistry.build("decision_tree", params={"max_depth": 2})
        model.fit(X, y)
        meta = model.metadata()
        # Must not raise
        json.dumps(meta)

    def test_fit_time_recorded(self, tabular_data):
        X, y = tabular_data
        model = ModelRegistry.build("logistic_l2")
        model.fit(X, y)
        assert model.fit_time_seconds > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Category counts (structural sanity)
# ─────────────────────────────────────────────────────────────────────────────

class TestCategoryCounts:
    def test_category_a_has_at_least_5_models(self):
        assert len(ModelRegistry.list_by_category()["A"]) >= 5

    def test_category_b_has_at_least_3_models(self):
        assert len(ModelRegistry.list_by_category()["B"]) >= 3

    def test_category_c_has_at_least_4_models(self):
        assert len(ModelRegistry.list_by_category()["C"]) >= 4

    def test_category_d_has_at_least_1_model(self):
        assert len(ModelRegistry.list_by_category()["D"]) >= 1

    def test_category_e_has_at_least_11_models(self):
        assert len(ModelRegistry.list_by_category()["E"]) >= 11

    def test_category_f_has_at_least_2_models(self):
        assert len(ModelRegistry.list_by_category()["F"]) >= 2
