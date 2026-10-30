"""
Unified Model Registry and Base Interface for Academic Intelligence Framework.

Every model in the zoo — from a Logistic Regression to a Transformer — exposes
an identical interface so the training engine, hyperparameter tuner, leaderboard,
and frontend need zero knowledge of each model's internals.

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("AcademicIntelligence.Registry")

# ── Model Categories ─────────────────────────────────────────────────────────
CATEGORIES = {
    "A": "Linear & Simple Baselines (tabular)",
    "B": "Tree & Bagging Ensembles (tabular)",
    "C": "Boosting — Engineered Trend Features (PROPOSAL BASELINE)",
    "D": "Neural, Tabular (MLP)",
    "E": "Deep Sequence Models (PyTorch, masked)",
    "F": "Static-Snapshot Variants (value-of-trajectory study)",
}


# ── Base Interface ────────────────────────────────────────────────────────────
class BaseAcademicModel(ABC):
    """
    Abstract base class that every model in the zoo must implement.
    Guarantees a unified interface for the training engine, Optuna tuner,
    evaluation harness, and REST API.
    """

    #: set by each concrete class
    MODEL_KEY: str = ""
    CATEGORY: str = ""
    LABEL: str = ""
    IS_PROPOSAL_MODEL: bool = False
    IS_SEQUENCE_MODEL: bool = False  # True for Cat-E deep models

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = params or {}
        self.is_fitted: bool = False
        self.fit_time_seconds: float = 0.0
        self.training_history: List[Dict[str, Any]] = []  # epoch logs for deep models

    # ── Required interface ────────────────────────────────────────────────────

    @abstractmethod
    def fit(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_val: Optional[Any] = None,
        y_val: Optional[np.ndarray] = None,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> "BaseAcademicModel":
        """Train the model. Returns self for chaining."""

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Return probability of the positive class (attrition) for each sample.
        Shape: (n_samples,)  — a 1-D float array in [0, 1].
        """

    @abstractmethod
    def save(self, directory: Path) -> Path:
        """Persist model weights/state to *directory*. Returns path of saved artifact."""

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "BaseAcademicModel":
        """Load model from *directory*. Returns new instance."""

    # ── Convenience helpers (not abstract) ───────────────────────────────────

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions at *threshold*."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def metadata(self) -> Dict[str, Any]:
        """Serialisable metadata for leaderboard and API responses."""
        return {
            "model_key": self.MODEL_KEY,
            "category": self.CATEGORY,
            "category_label": CATEGORIES.get(self.CATEGORY, "Unknown"),
            "label": self.LABEL,
            "is_proposal_model": self.IS_PROPOSAL_MODEL,
            "is_sequence_model": self.IS_SEQUENCE_MODEL,
            "is_fitted": self.is_fitted,
            "fit_time_seconds": round(self.fit_time_seconds, 2),
            "params": self.params,
        }

    def _start_timer(self) -> float:
        return time.perf_counter()

    def _stop_timer(self, t0: float) -> None:
        self.fit_time_seconds = time.perf_counter() - t0


# ── Model Registry ────────────────────────────────────────────────────────────
class ModelRegistry:
    """
    Central registry: maps MODEL_KEY → concrete class.
    Loaded config from configs/models.yaml provides default params and search spaces.
    """

    _registry: Dict[str, Type[BaseAcademicModel]] = {}
    _config_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def register(cls, model_cls: Type[BaseAcademicModel]) -> Type[BaseAcademicModel]:
        """Decorator that registers a model class by its MODEL_KEY."""
        if not model_cls.MODEL_KEY:
            raise ValueError(f"Model class {model_cls.__name__} must define MODEL_KEY.")
        cls._registry[model_cls.MODEL_KEY] = model_cls
        return model_cls

    @classmethod
    def _load_config(cls) -> Dict[str, Any]:
        if cls._config_cache is None:
            cfg_path = Path(__file__).resolve().parents[3] / "configs" / "models.yaml"
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            cls._config_cache = raw.get("models", {})
        return cls._config_cache

    @classmethod
    def get_default_params(cls, key: str) -> Dict[str, Any]:
        cfg = cls._load_config()
        if key not in cfg:
            raise KeyError(f"Model key '{key}' not found in configs/models.yaml")
        return dict(cfg[key].get("default", {}) or {})

    @classmethod
    def get_search_space(cls, key: str) -> Dict[str, Any]:
        cfg = cls._load_config()
        return dict(cfg[key].get("search", {}) or {})

    @classmethod
    def get_meta(cls, key: str) -> Dict[str, Any]:
        cfg = cls._load_config()
        return dict(cfg[key].get("meta", {}) or {})

    @classmethod
    def build(cls, key: str, params: Optional[Dict[str, Any]] = None) -> BaseAcademicModel:
        """Instantiate a model by key, merging defaults with any overrides."""
        if key not in cls._registry:
            raise KeyError(
                f"Model '{key}' not registered. Available: {list(cls._registry.keys())}"
            )
        defaults = cls.get_default_params(key)
        merged = {**defaults, **(params or {})}
        return cls._registry[key](params=merged)

    @classmethod
    def load_model(cls, key: str, directory: Path) -> BaseAcademicModel:
        """Loads a persisted model from directory using the registered model class."""
        if key not in cls._registry:
            raise KeyError(
                f"Model '{key}' not registered. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key].load(directory)

    @classmethod
    def list_all(cls) -> List[Dict[str, Any]]:
        """Returns a sorted list of all registered models with their metadata."""
        rows = []
        cfg = cls._load_config()
        for key, model_cls in sorted(cls._registry.items()):
            meta = dict(cfg.get(key, {}).get("meta", {}) or {})
            rows.append({
                "key": key,
                "category": meta.get("category", "?"),
                "label": meta.get("label", key),
                "is_proposal_model": meta.get("proposal_model", False),
                "is_sequence_model": model_cls.IS_SEQUENCE_MODEL,
                "expected_runtime_mins": meta.get("expected_runtime_mins", "?"),
            })
        rows.sort(key=lambda r: (r["category"], r["label"]))
        return rows

    @classmethod
    def list_by_category(cls) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CATEGORIES}
        for row in cls.list_all():
            cat = row["category"]
            if cat in out:
                out[cat].append(row)
        return out

    @classmethod
    def keys(cls) -> List[str]:
        return list(cls._registry.keys())


def register(model_cls: Type[BaseAcademicModel]) -> Type[BaseAcademicModel]:
    """Module-level decorator alias for ModelRegistry.register."""
    return ModelRegistry.register(model_cls)
