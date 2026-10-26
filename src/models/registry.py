"""Model Registry and Factory Module.

Provides:
- Central registry of all candidate models categorized into:
  - Baseline (Dummy)
  - Linear/Shallow
  - Ensemble/Bagging
  - Boosting
  - Deep Learning (PyTorch)
  - Meta (Voting & Stacking)
- Dynamic instantiation for regression (PRI continuous 0-100) and classification (3-tier risk category).
- Optuna hyperparameter search spaces per model family.
- Graceful exception handling for any optional dependencies that fail to import.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, RidgeClassifier,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    VotingRegressor, VotingClassifier,
    StackingRegressor, StackingClassifier,
)

from src.utils.logger import get_logger

logger = get_logger("models.registry")

# Safe optional imports
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    xgb = None
    _XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    lgb = None
    _LGB_AVAILABLE = False

try:
    import catboost as cb
    _CB_AVAILABLE = True
except ImportError:
    cb = None
    _CB_AVAILABLE = False

try:
    from src.models.architectures import PyTorchTabularEstimator, _TORCH_AVAILABLE
except Exception:
    PyTorchTabularEstimator = None
    _TORCH_AVAILABLE = False


MODEL_CATEGORIES: Dict[str, str] = {
    # Baseline
    "dummy": "Baseline",
    # Linear / Shallow
    "linear_regression": "Linear/Shallow",
    "logistic_regression": "Linear/Shallow",
    "ridge": "Linear/Shallow",
    "lasso": "Linear/Shallow",
    "elasticnet": "Linear/Shallow",
    "lda": "Linear/Shallow",
    "gaussian_nb": "Linear/Shallow",
    "knn": "Linear/Shallow",
    "svm_linear": "Linear/Shallow",
    "svm_rbf": "Linear/Shallow",
    "decision_tree": "Linear/Shallow",
    # Ensemble / Bagging
    "random_forest": "Ensemble/Bagging",
    "extra_trees": "Ensemble/Bagging",
    # Boosting
    "gradient_boosting": "Boosting",
    "hist_gradient_boosting": "Boosting",
    "adaboost": "Boosting",
    "xgboost": "Boosting",
    "lightgbm": "Boosting",
    "catboost": "Boosting",
    # Deep Learning
    "tabular_mlp": "Deep Learning",
    "residual_mlp": "Deep Learning",
    # Meta
    "voting_ensemble": "Meta",
    "stacking_ensemble": "Meta",
}


def list_registered_models(track: str = "both") -> List[str]:
    """Return list of supported model identifiers for a given track."""
    track_clean = track.lower().strip()
    all_models = list(MODEL_CATEGORIES.keys())

    if track_clean == "regression":
        # Exclude purely classification models
        excluded = {"lda", "gaussian_nb"}
        return [m for m in all_models if m not in excluded]
    elif track_clean == "classification":
        # Exclude purely regression models
        excluded = {"linear_regression", "lasso"}
        return [m for m in all_models if m not in excluded]
    return all_models


def build_model(
    model_name: str,
    track: str = "regression",
    params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Any:
    """Instantiate a model estimator configured with parameters.

    Args:
        model_name: Identifier from MODEL_CATEGORIES.
        track: 'regression' or 'classification'.
        params: Hyperparameter dictionary override.
        seed: Random state seed.

    Returns:
        Estimator instance implementing fit and predict.
    """
    p = (params or {}).copy()
    m_name = model_name.lower().strip()
    is_reg = track.lower() == "regression"

    # 1. Baseline
    if m_name == "dummy":
        return DummyRegressor(strategy="mean") if is_reg else DummyClassifier(strategy="prior")

    # 2. Linear / Shallow
    elif m_name == "linear_regression":
        return LinearRegression(**p)
    elif m_name == "logistic_regression":
        p.setdefault("max_iter", 1000)
        p.setdefault("random_state", seed)
        return LogisticRegression(**p)
    elif m_name == "ridge":
        p.setdefault("random_state", seed)
        return Ridge(**p) if is_reg else RidgeClassifier(**p)
    elif m_name == "lasso":
        p.setdefault("random_state", seed)
        return Lasso(**p)
    elif m_name == "elasticnet":
        p.setdefault("random_state", seed)
        return ElasticNet(**p) if is_reg else LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=1000, random_state=seed, **p)
    elif m_name == "lda":
        return LinearDiscriminantAnalysis(**p)
    elif m_name == "gaussian_nb":
        return GaussianNB(**p)
    elif m_name == "knn":
        return KNeighborsRegressor(**p) if is_reg else KNeighborsClassifier(**p)
    elif m_name == "svm_linear":
        return SVR(kernel="linear", **p) if is_reg else SVC(kernel="linear", probability=True, random_state=seed, **p)
    elif m_name == "svm_rbf":
        return SVR(kernel="rbf", **p) if is_reg else SVC(kernel="rbf", probability=True, random_state=seed, **p)
    elif m_name == "decision_tree":
        p.setdefault("random_state", seed)
        return DecisionTreeRegressor(**p) if is_reg else DecisionTreeClassifier(**p)

    # 3. Ensemble / Bagging
    elif m_name == "random_forest":
        p.setdefault("n_estimators", 100)
        p.setdefault("random_state", seed)
        p.setdefault("n_jobs", -1)
        return RandomForestRegressor(**p) if is_reg else RandomForestClassifier(**p)
    elif m_name == "extra_trees":
        p.setdefault("n_estimators", 100)
        p.setdefault("random_state", seed)
        p.setdefault("n_jobs", -1)
        return ExtraTreesRegressor(**p) if is_reg else ExtraTreesClassifier(**p)

    # 4. Boosting
    elif m_name == "gradient_boosting":
        p.setdefault("random_state", seed)
        return GradientBoostingRegressor(**p) if is_reg else GradientBoostingClassifier(**p)
    elif m_name == "hist_gradient_boosting":
        p.setdefault("random_state", seed)
        return HistGradientBoostingRegressor(**p) if is_reg else HistGradientBoostingClassifier(**p)
    elif m_name == "adaboost":
        p.setdefault("random_state", seed)
        return AdaBoostRegressor(**p) if is_reg else AdaBoostClassifier(**p)
    elif m_name == "xgboost":
        if not _XGB_AVAILABLE or xgb is None:
            raise ImportError("xgboost is not installed.")
        p.setdefault("random_state", seed)
        p.setdefault("n_jobs", -1)
        p.setdefault("verbosity", 0)
        return xgb.XGBRegressor(**p) if is_reg else xgb.XGBClassifier(**p)
    elif m_name == "lightgbm":
        if not _LGB_AVAILABLE or lgb is None:
            raise ImportError("lightgbm is not installed.")
        p.setdefault("random_state", seed)
        p.setdefault("n_jobs", -1)
        p.setdefault("verbosity", -1)
        return lgb.LGBMRegressor(**p) if is_reg else lgb.LGBMClassifier(**p)
    elif m_name == "catboost":
        if not _CB_AVAILABLE or cb is None:
            raise ImportError("catboost is not installed.")
        p.setdefault("random_seed", seed)
        p.setdefault("verbose", False)
        return cb.CatBoostRegressor(**p) if is_reg else cb.CatBoostClassifier(**p)

    # 5. Deep Learning
    elif m_name in ("tabular_mlp", "residual_mlp"):
        if not _TORCH_AVAILABLE or PyTorchTabularEstimator is None:
            raise ImportError("PyTorch is not available for tabular deep learning.")
        arch_type = "res_mlp" if m_name == "residual_mlp" else "mlp"
        return PyTorchTabularEstimator(
            arch=arch_type,
            track=track,
            seed=seed,
            **p,
        )

    # 6. Meta Ensembles
    elif m_name == "voting_ensemble":
        if is_reg:
            estimators = [
                ("rf", RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1)),
                ("ridge", Ridge(alpha=1.0, random_state=seed)),
                ("hgb", HistGradientBoostingRegressor(max_iter=50, random_state=seed)),
            ]
            return VotingRegressor(estimators=estimators, n_jobs=-1)
        else:
            estimators = [
                ("rf", RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)),
                ("lr", LogisticRegression(max_iter=500, random_state=seed)),
                ("hgb", HistGradientBoostingClassifier(max_iter=50, random_state=seed)),
            ]
            return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

    elif m_name == "stacking_ensemble":
        if is_reg:
            estimators = [
                ("rf", RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1)),
                ("ridge", Ridge(alpha=1.0, random_state=seed)),
            ]
            final_estimator = Ridge(alpha=1.0)
            return StackingRegressor(estimators=estimators, final_estimator=final_estimator, n_jobs=-1)
        else:
            estimators = [
                ("rf", RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)),
                ("lr", LogisticRegression(max_iter=500, random_state=seed)),
            ]
            final_estimator = LogisticRegression(max_iter=500)
            return StackingClassifier(estimators=estimators, final_estimator=final_estimator, n_jobs=-1)

    else:
        raise ValueError(f"Unknown model name '{model_name}'. Valid options: {list(MODEL_CATEGORIES.keys())}")


def sample_optuna_params(model_name: str, trial: Any) -> Dict[str, Any]:
    """Define search spaces for Optuna hyperparameter optimization.

    Args:
        model_name: Identifier of the model.
        trial: Optuna trial object.

    Returns:
        Dict[str, Any]: Sampled hyperparameter dictionary.
    """
    m = model_name.lower().strip()

    if m in ("ridge", "lasso"):
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True)}
    elif m == "elasticnet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
        }
    elif m == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    elif m == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    elif m in ("svm_linear", "svm_rbf"):
        return {"C": trial.suggest_float("C", 1e-2, 1e2, log=True)}
    elif m in ("decision_tree", "random_forest", "extra_trees"):
        params: Dict[str, Any] = {
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
        if m in ("random_forest", "extra_trees"):
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 150, step=25)
        return params
    elif m in ("gradient_boosting", "hist_gradient_boosting"):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
        }
    elif m == "adaboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 30, 120, step=30),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        }
    elif m == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    elif m == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        }
    elif m == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 50, 150, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 7),
        }
    elif m in ("tabular_mlp", "residual_mlp"):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.4),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        }
    return {}
