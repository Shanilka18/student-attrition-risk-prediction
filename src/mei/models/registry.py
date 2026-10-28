"""Model zoo registry, specification models, and estimator factory for MEI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from mei.config import CONFIG_DIR


class RuleLoginImitationClassifier(BaseEstimator, ClassifierMixin):
    """Simple deterministic heuristic rule imitating Moodle/LMS login frequency alerts.

    Rule: If the student's mean resource_access across periods is below threshold (default 2.0),
    flag as at-risk (1), otherwise 0.
    """

    def __init__(self, threshold: float = 2.0) -> None:
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def fit(self, X: Any, y: Optional[Any] = None) -> "RuleLoginImitationClassifier":
        """No parameters to fit."""
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict binary risk flag based on average activity."""
        X_arr = np.asarray(X, dtype=float)
        mean_val = np.mean(X_arr, axis=1)
        return (mean_val < self.threshold).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return calibrated pseudo-probabilities based on inverse linear distance to threshold."""
        X_arr = np.asarray(X, dtype=float)
        mean_val = np.mean(X_arr, axis=1)
        # Scaled between 0 and 1: 1 (min activity) -> p=1.0, 5 (max activity) -> p=0.0
        p_risk = np.clip(1.0 - ((mean_val - 1.0) / 4.0), 0.01, 0.99)
        p_safe = 1.0 - p_risk
        return np.column_stack([p_safe, p_risk])


@dataclass
class ModelSpec:
    """Standardized metadata and execution specification for a candidate model."""

    id: str
    display_name: str
    plain_name: str
    category: str
    task: str
    feature_set: str
    needs_scaling: bool = False
    supports_class_weight: bool = False
    twin_model_id: Optional[str] = None
    n_tuning_trials: int = 20
    time_budget_seconds: int = 60
    key_hyperparameter: Optional[str] = None
    key_hyperparameter_range: List[Any] = field(default_factory=list)
    search_space: Dict[str, Any] = field(default_factory=dict)
    fixed_params: Dict[str, Any] = field(default_factory=dict)


def load_model_zoo_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and parse config/model_zoo.yaml."""
    if config_path is None:
        config_path = CONFIG_DIR / "model_zoo.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_all_model_specs(config_path: Optional[Path] = None) -> Dict[str, ModelSpec]:
    """Load all registered model specifications from model_zoo.yaml."""
    zoo_cfg = load_model_zoo_config(config_path)
    models_dict = zoo_cfg.get("models", {})
    specs: Dict[str, ModelSpec] = {}

    for model_id, m_info in models_dict.items():
        spec = ModelSpec(
            id=model_id,
            display_name=m_info.get("display_name", model_id),
            plain_name=m_info.get("plain_name", model_id),
            category=m_info.get("category", "other"),
            task=m_info.get("task", "classification"),
            feature_set=m_info.get("feature_set", "agg"),
            needs_scaling=m_info.get("needs_scaling", False),
            supports_class_weight=m_info.get("supports_class_weight", False),
            twin_model_id=m_info.get("twin_model_id"),
            n_tuning_trials=m_info.get("n_tuning_trials", 20),
            time_budget_seconds=m_info.get("time_budget_seconds", 60),
            key_hyperparameter=m_info.get("key_hyperparameter"),
            key_hyperparameter_range=m_info.get("key_hyperparameter_range", []),
            search_space=m_info.get("search_space", {}),
            fixed_params=m_info.get("fixed_params", {}),
        )
        specs[model_id] = spec

    return specs


def create_base_estimator(
    spec: ModelSpec,
    hyperparams: Optional[Dict[str, Any]] = None,
    random_seed: int = 42,
    use_class_weights: bool = True,
) -> BaseEstimator:
    """Instantiate a concrete scikit-learn, XGBoost, LightGBM, or CatBoost classifier."""
    params = dict(spec.fixed_params)
    if hyperparams:
        params.update(hyperparams)

    mid = spec.id

    # 1. Baselines
    if mid == "single_signal_lr":
        cw = "balanced" if use_class_weights and spec.supports_class_weight else None
        return LogisticRegression(
            C=params.get("C", 1.0),
            solver="lbfgs",
            max_iter=1000,
            class_weight=cw,
            random_state=random_seed,
        )
    if mid == "majority_dummy":
        return DummyClassifier(
            strategy=params.get("strategy", "prior"), random_state=random_seed
        )
    if mid == "rule_login_imitation":
        return RuleLoginImitationClassifier(threshold=params.get("threshold", 2.0))

    # 2. Linear Models
    if mid in ["logistic_regression_l2", "logistic_regression_hybrid"]:
        cw = "balanced" if use_class_weights and spec.supports_class_weight else None
        return LogisticRegression(
            C=params.get("C", 1.0),
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight=cw,
            random_state=random_seed,
        )
    if mid == "logistic_regression_elasticnet":
        cw = "balanced" if use_class_weights and spec.supports_class_weight else None
        return LogisticRegression(
            C=params.get("C", 1.0),
            l1_ratio=params.get("l1_ratio", 0.5),
            penalty="elasticnet",
            solver="saga",
            max_iter=1500,
            class_weight=cw,
            random_state=random_seed,
        )
    if mid == "linear_svm_calibrated":
        cw = "balanced" if use_class_weights and spec.supports_class_weight else None
        base_svm = LinearSVC(
            C=params.get("C", 1.0),
            class_weight=cw,
            dual=False,
            max_iter=2000,
            random_state=random_seed,
        )
        return CalibratedClassifierCV(estimator=base_svm, method="sigmoid", cv=3)
    if mid == "ridge_calibrated":
        cw = "balanced" if use_class_weights and spec.supports_class_weight else None
        base_ridge = RidgeClassifier(
            alpha=params.get("alpha", 1.0),
            class_weight=cw,
            random_state=random_seed,
        )
        return CalibratedClassifierCV(estimator=base_ridge, method="sigmoid", cv=3)
    if mid == "lda":
        shrinkage = params.get("shrinkage")
        solver = "lsqr" if shrinkage is not None and shrinkage > 0.0 else "svd"
        shrink_val = shrinkage if solver == "lsqr" else None
        return LinearDiscriminantAnalysis(solver=solver, shrinkage=shrink_val)

    # 3. Classical / Shallow
    if mid == "knn":
        return KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "uniform"),
        )
    if mid == "gaussian_nb":
        return GaussianNB(var_smoothing=params.get("var_smoothing", 1e-9))
    if mid == "decision_tree":
        cw = "balanced" if use_class_weights and spec.supports_class_weight else None
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", 5),
            min_samples_split=params.get("min_samples_split", 10),
            class_weight=cw,
            random_state=random_seed,
        )
    if mid == "svm_rbf":
        cw = "balanced" if use_class_weights and spec.supports_class_weight else None
        return SVC(
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            kernel="rbf",
            probability=True,
            class_weight=cw,
            random_state=random_seed,
        )
    if mid == "mlp_shallow":
        h_units = params.get("hidden_units", 32)
        return MLPClassifier(
            hidden_layer_sizes=(h_units,),
            alpha=params.get("alpha", 1e-4),
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=random_seed,
        )
    if mid == "extra_trees":
        cw = "balanced" if use_class_weights and spec.supports_class_weight else None
        return ExtraTreesClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 8),
            class_weight=cw,
            n_jobs=-1,
            random_state=random_seed,
        )

    # 4. Boosting & Tree Ensembles
    if mid in ["random_forest", "random_forest_hybrid"]:
        cw = "balanced" if use_class_weights and spec.supports_class_weight else None
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 8),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            class_weight=cw,
            n_jobs=-1,
            random_state=random_seed,
        )
    if mid == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=random_seed,
        )
    if mid == "hist_gradient_boosting":
        cw = "balanced" if use_class_weights and spec.supports_class_weight else None
        return HistGradientBoostingClassifier(
            max_iter=params.get("max_iter", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            class_weight=cw,
            random_state=random_seed,
        )
    if mid in ["xgboost", "xgboost_hybrid"]:
        import xgboost as xgb

        # scale_pos_weight is approx (1 - prevalence) / prevalence ~= 3.0 for 25% prevalence
        spw = 2.95 if use_class_weights and spec.supports_class_weight else 1.0
        return xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 4),
            subsample=params.get("subsample", 0.8),
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=random_seed,
            n_jobs=-1,
        )
    if mid in ["lightgbm", "lightgbm_hybrid"]:
        import lightgbm as lgb

        cw = "balanced" if use_class_weights and spec.supports_class_weight else None
        return lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 31),
            subsample=params.get("subsample", 0.8),
            class_weight=cw,
            random_state=random_seed,
            verbose=-1,
            n_jobs=-1,
        )
    if mid in ["catboost", "catboost_hybrid"]:
        import catboost as cb

        auto_cw = (
            "Balanced" if use_class_weights and spec.supports_class_weight else None
        )
        return cb.CatBoostClassifier(
            iterations=params.get("iterations", 100),
            learning_rate=params.get("learning_rate", 0.05),
            depth=params.get("depth", 5),
            auto_class_weights=auto_cw,
            random_seed=random_seed,
            verbose=0,
            thread_count=-1,
        )
    if mid == "adaboost":
        return AdaBoostClassifier(
            n_estimators=params.get("n_estimators", 50),
            learning_rate=params.get("learning_rate", 0.1),
            algorithm="SAMME",
            random_state=random_seed,
        )

    raise ValueError(f"Unrecognized model ID: {mid}")
