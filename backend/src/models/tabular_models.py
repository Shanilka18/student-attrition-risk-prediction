"""
Category A — Linear & Simple Baselines, Category B — Tree/Bagging Ensembles,
Category C — Boosting (Proposal Baselines), Category D — Tabular MLP,
Category F — Static-Snapshot Variants.

All use scikit-learn API wrapped inside BaseAcademicModel.
Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

from backend.src.models.base import BaseAcademicModel, register

logger = logging.getLogger("AcademicIntelligence.TabularModels")


# ─────────────────────────────────────────────────────────────────────────────
# Shared sklearn wrapper mixin
# ─────────────────────────────────────────────────────────────────────────────

class _SklearnMixin:
    """Mixin adding sklearn-compatible fit/predict_proba/save/load."""

    _clf: Any = None
    IS_SEQUENCE_MODEL = False

    def _make_clf(self) -> Any:
        raise NotImplementedError

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weights=None):
        t0 = self._start_timer()
        clf = self._make_clf()
        # Pass sample_weight for estimators that accept it
        try:
            if class_weights is not None:
                sw = np.array([class_weights[int(y)] for y in y_train])
                clf.fit(X_train, y_train, sample_weight=sw)
            else:
                clf.fit(X_train, y_train)
        except TypeError:
            clf.fit(X_train, y_train)
        self._clf = clf
        self.is_fitted = True
        self._stop_timer(t0)
        logger.info(f"{self.LABEL} trained in {self.fit_time_seconds:.1f}s")
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.LABEL} not fitted yet.")
        proba = self._clf.predict_proba(X)
        return proba[:, 1].astype(np.float32)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.MODEL_KEY}.joblib"
        joblib.dump(self._clf, path)
        meta_path = directory / f"{self.MODEL_KEY}_meta.json"
        meta_path.write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path):
        directory = Path(directory)
        path = directory / f"{cls.MODEL_KEY}.joblib"
        obj = cls()
        obj._clf = joblib.load(path)
        obj.is_fitted = True
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY A — Linear & Simple
# ─────────────────────────────────────────────────────────────────────────────

@register
class LogisticL2Model(_SklearnMixin, BaseAcademicModel):
    MODEL_KEY = "logistic_l2"
    CATEGORY = "A"
    LABEL = "Logistic Regression (L2)"
    IS_PROPOSAL_MODEL = False

    def _make_clf(self):
        p = self.params
        return LogisticRegression(
            C=p.get("C", 1.0),
            penalty="l2",
            solver=p.get("solver", "lbfgs"),
            max_iter=p.get("max_iter", 500),
            class_weight=p.get("class_weight", "balanced"),
            random_state=p.get("random_state", 42),
        )


@register
class LogisticElasticNetModel(_SklearnMixin, BaseAcademicModel):
    MODEL_KEY = "logistic_elasticnet"
    CATEGORY = "A"
    LABEL = "Logistic Regression (ElasticNet)"
    IS_PROPOSAL_MODEL = False

    def _make_clf(self):
        p = self.params
        return LogisticRegression(
            C=p.get("C", 1.0),
            penalty="elasticnet",
            l1_ratio=p.get("l1_ratio", 0.5),
            solver="saga",
            max_iter=p.get("max_iter", 500),
            class_weight=p.get("class_weight", "balanced"),
            random_state=p.get("random_state", 42),
        )


@register
class SVMCalibratedModel(_SklearnMixin, BaseAcademicModel):
    MODEL_KEY = "svm_calibrated"
    CATEGORY = "A"
    LABEL = "Linear SVM (Calibrated)"
    IS_PROPOSAL_MODEL = False

    def _make_clf(self):
        p = self.params
        base = LinearSVC(
            C=p.get("C", 1.0),
            max_iter=p.get("max_iter", 1000),
            class_weight=p.get("class_weight", "balanced"),
            random_state=p.get("random_state", 42),
        )
        return CalibratedClassifierCV(base, cv=3)


@register
class GaussianNBModel(_SklearnMixin, BaseAcademicModel):
    MODEL_KEY = "gaussian_nb"
    CATEGORY = "A"
    LABEL = "Gaussian Naive Bayes"
    IS_PROPOSAL_MODEL = False

    def _make_clf(self):
        return GaussianNB(var_smoothing=self.params.get("var_smoothing", 1e-9))


@register
class KNNModel(_SklearnMixin, BaseAcademicModel):
    MODEL_KEY = "knn"
    CATEGORY = "A"
    LABEL = "k-Nearest Neighbours"
    IS_PROPOSAL_MODEL = False

    def _make_clf(self):
        p = self.params
        return KNeighborsClassifier(
            n_neighbors=p.get("n_neighbors", 7),
            weights=p.get("weights", "distance"),
            metric=p.get("metric", "euclidean"),
        )


@register
class GPAThresholdModel(BaseAcademicModel):
    """Rule-based reference — no learning, no sklearn."""
    MODEL_KEY = "gpa_threshold"
    CATEGORY = "A"
    LABEL = "GPA Threshold Rule (Reference)"
    IS_PROPOSAL_MODEL = False
    IS_SEQUENCE_MODEL = False

    def __init__(self, params=None):
        super().__init__(params)
        # Set threshold immediately so predict_proba works without fit()
        self.threshold: float = self.params.get("threshold", 2.0)

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weights=None):
        # Re-read in case params were updated before fit()
        self.threshold = self.params.get("threshold", 2.0)
        self.is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            col = "last_gpa" if "last_gpa" in X.columns else X.columns[0]
            vals = X[col].values
        else:
            vals = np.asarray(X)[:, 0]
        # Hard probability: 1.0 if below threshold, 0.0 otherwise
        return np.where(vals < self.threshold, 1.0, 0.0).astype(np.float32)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / f"{self.MODEL_KEY}_meta.json"
        p.write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, directory: Path):
        meta_path = directory / f"{cls.MODEL_KEY}_meta.json"
        obj = cls()
        if meta_path.exists():
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            obj.params = data.get("params", {})
        obj.is_fitted = True
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY B — Tree & Bagging
# ─────────────────────────────────────────────────────────────────────────────

@register
class DecisionTreeModel(_SklearnMixin, BaseAcademicModel):
    MODEL_KEY = "decision_tree"
    CATEGORY = "B"
    LABEL = "Decision Tree"
    IS_PROPOSAL_MODEL = False

    def _make_clf(self):
        p = self.params
        return DecisionTreeClassifier(
            max_depth=p.get("max_depth", 6),
            min_samples_leaf=p.get("min_samples_leaf", 10),
            class_weight=p.get("class_weight", "balanced"),
            random_state=p.get("random_state", 42),
        )


@register
class RandomForestModel(_SklearnMixin, BaseAcademicModel):
    MODEL_KEY = "random_forest"
    CATEGORY = "B"
    LABEL = "Random Forest"
    IS_PROPOSAL_MODEL = False

    def _make_clf(self):
        p = self.params
        return RandomForestClassifier(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", 10),
            min_samples_leaf=p.get("min_samples_leaf", 5),
            class_weight=p.get("class_weight", "balanced"),
            n_jobs=p.get("n_jobs", -1),
            random_state=p.get("random_state", 42),
        )


@register
class ExtraTreesModel(_SklearnMixin, BaseAcademicModel):
    MODEL_KEY = "extra_trees"
    CATEGORY = "B"
    LABEL = "Extra Trees"
    IS_PROPOSAL_MODEL = False

    def _make_clf(self):
        p = self.params
        return ExtraTreesClassifier(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", 10),
            min_samples_leaf=p.get("min_samples_leaf", 5),
            class_weight=p.get("class_weight", "balanced"),
            n_jobs=p.get("n_jobs", -1),
            random_state=p.get("random_state", 42),
        )


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY C — Boosting (Proposal Baselines)
# ─────────────────────────────────────────────────────────────────────────────

@register
class GradientBoostingSklearnModel(_SklearnMixin, BaseAcademicModel):
    MODEL_KEY = "gradient_boosting_sklearn"
    CATEGORY = "C"
    LABEL = "Gradient Boosting (sklearn)"
    IS_PROPOSAL_MODEL = False

    def _make_clf(self):
        p = self.params
        return GradientBoostingClassifier(
            n_estimators=p.get("n_estimators", 200),
            learning_rate=p.get("learning_rate", 0.05),
            max_depth=p.get("max_depth", 4),
            subsample=p.get("subsample", 0.8),
            min_samples_leaf=p.get("min_samples_leaf", 10),
            random_state=p.get("random_state", 42),
        )


@register
class XGBoostModel(BaseAcademicModel):
    MODEL_KEY = "xgboost"
    CATEGORY = "C"
    LABEL = "XGBoost (Proposal Baseline)"
    IS_PROPOSAL_MODEL = True
    IS_SEQUENCE_MODEL = False
    _clf: Any = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weights=None):
        t0 = self._start_timer()
        p = self.params
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        spw = float(n_neg / max(n_pos, 1))

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self._clf = xgb.XGBClassifier(
            n_estimators=p.get("n_estimators", 300),
            learning_rate=p.get("learning_rate", 0.05),
            max_depth=p.get("max_depth", 5),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            reg_alpha=p.get("reg_alpha", 0.1),
            reg_lambda=p.get("reg_lambda", 1.0),
            scale_pos_weight=spw,
            eval_metric="logloss",
            use_label_encoder=False,
            early_stopping_rounds=p.get("early_stopping_rounds", 30) if eval_set is not None else None,
            random_state=p.get("random_state", 42),
            verbosity=0,
        )
        fit_kwargs: Dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False
        self._clf.fit(X_train, y_train, **fit_kwargs)
        self.is_fitted = True
        self._stop_timer(t0)
        logger.info(f"XGBoost trained in {self.fit_time_seconds:.1f}s")
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._clf.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / f"{self.MODEL_KEY}.joblib"
        joblib.dump(self._clf, p)
        (directory / f"{self.MODEL_KEY}_meta.json").write_text(
            json.dumps(self.metadata(), indent=2), encoding="utf-8"
        )
        return p

    @classmethod
    def load(cls, directory: Path):
        obj = cls()
        obj._clf = joblib.load(directory / f"{cls.MODEL_KEY}.joblib")
        obj.is_fitted = True
        return obj


@register
class LightGBMModel(BaseAcademicModel):
    MODEL_KEY = "lightgbm"
    CATEGORY = "C"
    LABEL = "LightGBM (Proposal Baseline)"
    IS_PROPOSAL_MODEL = True
    IS_SEQUENCE_MODEL = False
    _clf: Any = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weights=None):
        t0 = self._start_timer()
        p = self.params
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self._clf = lgb.LGBMClassifier(
            n_estimators=p.get("n_estimators", 300),
            learning_rate=p.get("learning_rate", 0.05),
            num_leaves=p.get("num_leaves", 31),
            max_depth=p.get("max_depth", -1),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            reg_alpha=p.get("reg_alpha", 0.1),
            reg_lambda=p.get("reg_lambda", 1.0),
            class_weight=p.get("class_weight", "balanced"),
            random_state=p.get("random_state", 42),
            verbosity=-1,
        )
        fit_kwargs: Dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["callbacks"] = callbacks
        self._clf.fit(X_train, y_train, **fit_kwargs)
        self.is_fitted = True
        self._stop_timer(t0)
        logger.info(f"LightGBM trained in {self.fit_time_seconds:.1f}s")
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._clf.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / f"{self.MODEL_KEY}.joblib"
        joblib.dump(self._clf, p)
        (directory / f"{self.MODEL_KEY}_meta.json").write_text(
            json.dumps(self.metadata(), indent=2), encoding="utf-8"
        )
        return p

    @classmethod
    def load(cls, directory: Path):
        directory = Path(directory)
        obj = cls()
        obj._clf = joblib.load(directory / f"{cls.MODEL_KEY}.joblib")
        obj.is_fitted = True
        return obj


@register
class CatBoostModel(BaseAcademicModel):
    MODEL_KEY = "catboost"
    CATEGORY = "C"
    LABEL = "CatBoost"
    IS_PROPOSAL_MODEL = False
    IS_SEQUENCE_MODEL = False
    _clf: Any = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weights=None):
        t0 = self._start_timer()
        p = self.params
        eval_set = cb.Pool(X_val, y_val) if X_val is not None else None
        self._clf = cb.CatBoostClassifier(
            iterations=p.get("iterations", 300),
            learning_rate=p.get("learning_rate", 0.05),
            depth=p.get("depth", 6),
            auto_class_weights=p.get("auto_class_weights", "Balanced"),
            random_seed=p.get("random_seed", p.get("random_state", 42)),
            verbose=0,
        )
        fit_kwargs: Dict[str, Any] = {"early_stopping_rounds": 50} if eval_set else {}
        self._clf.fit(X_train, y_train, eval_set=eval_set, **fit_kwargs)
        self.is_fitted = True
        self._stop_timer(t0)
        logger.info(f"CatBoost trained in {self.fit_time_seconds:.1f}s")
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._clf.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / f"{self.MODEL_KEY}.cbm"
        self._clf.save_model(str(model_path))
        (directory / f"{self.MODEL_KEY}_meta.json").write_text(
            json.dumps(self.metadata(), indent=2), encoding="utf-8"
        )
        return model_path

    @classmethod
    def load(cls, directory: Path):
        directory = Path(directory)
        obj = cls()
        obj._clf = cb.CatBoostClassifier()
        obj._clf.load_model(str(directory / f"{cls.MODEL_KEY}.cbm"))
        obj.is_fitted = True
        return obj


@register
class AdaBoostModel(_SklearnMixin, BaseAcademicModel):
    MODEL_KEY = "adaboost"
    CATEGORY = "C"
    LABEL = "AdaBoost"
    IS_PROPOSAL_MODEL = False

    def _make_clf(self):
        p = self.params
        return AdaBoostClassifier(
            n_estimators=p.get("n_estimators", 200),
            learning_rate=p.get("learning_rate", 0.1),
            random_state=p.get("random_state", 42),
        )


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY F — Static-snapshot variants
# ─────────────────────────────────────────────────────────────────────────────

@register
class StaticLogisticModel(_SklearnMixin, BaseAcademicModel):
    MODEL_KEY = "static_logistic"
    CATEGORY = "F"
    LABEL = "Logistic Regression (Static Snapshot)"
    IS_PROPOSAL_MODEL = False

    def _make_clf(self):
        p = self.params
        return LogisticRegression(
            C=p.get("C", 1.0),
            solver=p.get("solver", "lbfgs"),
            max_iter=p.get("max_iter", 500),
            class_weight=p.get("class_weight", "balanced"),
            random_state=p.get("random_state", 42),
        )


@register
class StaticLightGBMModel(LightGBMModel):
    MODEL_KEY = "static_lightgbm"
    CATEGORY = "F"
    LABEL = "LightGBM (Static Snapshot)"
    IS_PROPOSAL_MODEL = False
