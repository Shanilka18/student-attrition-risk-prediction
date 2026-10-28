"""Regression model zoo registry, specification models, and estimator factory for MEI Track B.

Defines candidate architectures across 7 model families:
1. Baseline: mean regressor (DummyRegressor), single-signal OLS linear regression.
2. Linear: Linear Regression, Ridge, Lasso, ElasticNet, Bayesian Ridge, Huber Regressor.
3. Classical / Shallow: KNN Regressor, SVR (RBF), Decision Tree Regressor, shallow MLPRegressor, Extra-Trees Regressor.
4. Tree Ensembles & Boosting: Random Forest, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost, AdaBoost.
5. Proposed State Hybrids: XGBoost, LightGBM, CatBoost, Random Forest, and Ridge on [agg_states].
6. Deep Learning Regressors: MLP-Wide, MLP-Residual, GRU, LSTM, Bi-GRU, 1D-CNN, Tiny Transformer, Hybrid Multimodal.
7. Ensembles: Soft Voting Regressor (top 3 by CV), Stacking Regressor (Ridge meta-learner on top 5 diverse OOF predictions).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None


@dataclass
class RegressionModelSpec:
    """Standardized metadata and execution specification for a regression candidate model."""

    id: str
    display_name: str
    plain_name: str
    category: str
    task: str = "regression"
    feature_set: str = "agg"
    needs_scaling: bool = False
    twin_model_id: Optional[str] = None
    n_tuning_trials: int = 15
    time_budget_seconds: int = 45
    key_hyperparameter: Optional[str] = None
    key_hyperparameter_range: List[Any] = field(default_factory=list)
    search_space: Dict[str, Any] = field(default_factory=dict)
    fixed_params: Dict[str, Any] = field(default_factory=dict)


def get_all_regression_model_specs() -> Dict[str, RegressionModelSpec]:
    """Return dictionary of all registered regression model specifications."""
    specs: Dict[str, RegressionModelSpec] = {}

    # 1. Baselines
    specs["mean_regressor"] = RegressionModelSpec(
        id="mean_regressor",
        display_name="Mean Regressor",
        plain_name="Historical Mean Score Baseline",
        category="baseline",
        feature_set="agg",
        needs_scaling=False,
        n_tuning_trials=0,
        time_budget_seconds=5,
    )
    specs["single_signal_linear"] = RegressionModelSpec(
        id="single_signal_linear",
        display_name="Single-Signal Linear Regression",
        plain_name="Login-Frequency Linear Regressor",
        category="baseline",
        feature_set="single_signal",
        needs_scaling=True,
        n_tuning_trials=0,
        time_budget_seconds=5,
    )

    # 2. Linear Models
    specs["linear_regression"] = RegressionModelSpec(
        id="linear_regression",
        display_name="Linear Regression (OLS)",
        plain_name="Ordinary Least Squares Regressor",
        category="linear",
        feature_set="agg",
        needs_scaling=True,
        n_tuning_trials=0,
        time_budget_seconds=5,
    )
    specs["ridge_regressor"] = RegressionModelSpec(
        id="ridge_regressor",
        display_name="Ridge Regression",
        plain_name="L2-Penalized Linear Regressor",
        category="linear",
        feature_set="agg",
        needs_scaling=True,
        n_tuning_trials=15,
        time_budget_seconds=20,
        key_hyperparameter="alpha",
        key_hyperparameter_range=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        search_space={
            "alpha": {"type": "float", "low": 1e-3, "high": 1e3, "log": True}
        },
    )
    specs["lasso_regressor"] = RegressionModelSpec(
        id="lasso_regressor",
        display_name="Lasso Regression",
        plain_name="L1-Penalized Sparse Linear Regressor",
        category="linear",
        feature_set="agg",
        needs_scaling=True,
        n_tuning_trials=15,
        time_budget_seconds=20,
        key_hyperparameter="alpha",
        key_hyperparameter_range=[0.001, 0.01, 0.1, 1.0, 5.0],
        search_space={
            "alpha": {"type": "float", "low": 1e-3, "high": 5.0, "log": True}
        },
    )
    specs["elasticnet_regressor"] = RegressionModelSpec(
        id="elasticnet_regressor",
        display_name="ElasticNet",
        plain_name="Balanced L1/L2 Convex Combination Regressor",
        category="linear",
        feature_set="agg",
        needs_scaling=True,
        n_tuning_trials=15,
        time_budget_seconds=20,
        search_space={
            "alpha": {"type": "float", "low": 1e-3, "high": 5.0, "log": True},
            "l1_ratio": {"type": "float", "low": 0.1, "high": 0.9},
        },
    )
    specs["bayesian_ridge_regressor"] = RegressionModelSpec(
        id="bayesian_ridge_regressor",
        display_name="Bayesian Ridge",
        plain_name="Probabilistic Evidence-Maximizing Linear Regressor",
        category="linear",
        feature_set="agg",
        needs_scaling=True,
        n_tuning_trials=0,
        time_budget_seconds=10,
    )
    specs["huber_regressor"] = RegressionModelSpec(
        id="huber_regressor",
        display_name="Huber Regressor",
        plain_name="Outlier-Robust Linear Regressor",
        category="linear",
        feature_set="agg",
        needs_scaling=True,
        n_tuning_trials=15,
        time_budget_seconds=25,
        search_space={
            "epsilon": {"type": "float", "low": 1.1, "high": 1.9},
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
        },
    )

    # 3. Classical / Shallow
    specs["knn_regressor"] = RegressionModelSpec(
        id="knn_regressor",
        display_name="k-Nearest Neighbours Regressor",
        plain_name="Local Metric Space Average Regressor",
        category="classical",
        feature_set="agg",
        needs_scaling=True,
        n_tuning_trials=15,
        time_budget_seconds=20,
        key_hyperparameter="n_neighbors",
        key_hyperparameter_range=[3, 5, 7, 11, 15, 21, 25],
        search_space={
            "n_neighbors": {"type": "int", "low": 3, "high": 25},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
        },
    )
    specs["svr_rbf"] = RegressionModelSpec(
        id="svr_rbf",
        display_name="Support Vector Regression (RBF)",
        plain_name="Kernel Margin-Based Continuous Regressor",
        category="classical",
        feature_set="agg",
        needs_scaling=True,
        n_tuning_trials=15,
        time_budget_seconds=30,
        key_hyperparameter="C",
        key_hyperparameter_range=[0.5, 1.0, 5.0, 10.0, 25.0, 50.0],
        search_space={
            "C": {"type": "float", "low": 0.5, "high": 50.0, "log": True},
            "epsilon": {"type": "float", "low": 0.05, "high": 2.0},
        },
    )
    specs["decision_tree_regressor"] = RegressionModelSpec(
        id="decision_tree_regressor",
        display_name="Decision Tree Regressor",
        plain_name="Single Hierarchical Rule Partition Regressor",
        category="classical",
        feature_set="agg",
        needs_scaling=False,
        n_tuning_trials=15,
        time_budget_seconds=15,
        search_space={
            "max_depth": {"type": "int", "low": 3, "high": 9},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 30},
        },
    )
    specs["mlp_shallow_regressor"] = RegressionModelSpec(
        id="mlp_shallow_regressor",
        display_name="Shallow MLP Regressor",
        plain_name="Two-Layer Feedforward Neural Regressor",
        category="classical",
        feature_set="agg",
        needs_scaling=True,
        n_tuning_trials=10,
        time_budget_seconds=30,
        search_space={
            "alpha": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
        },
    )
    specs["extra_trees_regressor"] = RegressionModelSpec(
        id="extra_trees_regressor",
        display_name="Extra-Trees Regressor",
        plain_name="Extremely Randomized Tree Ensemble Regressor",
        category="classical",
        feature_set="agg",
        needs_scaling=False,
        n_tuning_trials=15,
        time_budget_seconds=25,
        search_space={
            "n_estimators": {"type": "int", "low": 50, "high": 150},
            "max_depth": {"type": "int", "low": 4, "high": 10},
        },
    )

    # 4. Boosting & Tree Ensembles
    specs["random_forest_regressor"] = RegressionModelSpec(
        id="random_forest_regressor",
        display_name="Random Forest Regressor",
        plain_name="Bootstrap Aggregated Decision Trees Regressor",
        category="boosting",
        feature_set="agg",
        needs_scaling=False,
        n_tuning_trials=15,
        time_budget_seconds=30,
        key_hyperparameter="max_depth",
        key_hyperparameter_range=[4, 6, 8, 10, 12],
        search_space={
            "n_estimators": {"type": "int", "low": 60, "high": 150},
            "max_depth": {"type": "int", "low": 4, "high": 10},
            "min_samples_leaf": {"type": "int", "low": 4, "high": 16},
        },
    )
    specs["gradient_boosting_regressor"] = RegressionModelSpec(
        id="gradient_boosting_regressor",
        display_name="Gradient Boosting Regressor",
        plain_name="Sequentially Boosted Decision Trees Regressor",
        category="boosting",
        feature_set="agg",
        needs_scaling=False,
        n_tuning_trials=15,
        time_budget_seconds=35,
        search_space={
            "n_estimators": {"type": "int", "low": 60, "high": 150},
            "learning_rate": {"type": "float", "low": 0.02, "high": 0.15, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 5},
        },
    )
    specs["hist_gradient_boosting_regressor"] = RegressionModelSpec(
        id="hist_gradient_boosting_regressor",
        display_name="HistGradientBoosting Regressor",
        plain_name="Binned Histogram Gradient Boosting Regressor",
        category="boosting",
        feature_set="agg",
        needs_scaling=False,
        n_tuning_trials=15,
        time_budget_seconds=25,
        search_space={
            "learning_rate": {"type": "float", "low": 0.02, "high": 0.15, "log": True},
            "max_iter": {"type": "int", "low": 60, "high": 150},
            "max_depth": {"type": "int", "low": 3, "high": 6},
        },
    )
    specs["xgboost_regressor"] = RegressionModelSpec(
        id="xgboost_regressor",
        display_name="XGBoost Regressor",
        plain_name="Exact Second-Order Gradient Boosting Regressor",
        category="boosting",
        feature_set="agg",
        needs_scaling=False,
        twin_model_id="xgboost_hybrid_regressor",
        n_tuning_trials=15,
        time_budget_seconds=30,
        search_space={
            "n_estimators": {"type": "int", "low": 60, "high": 150},
            "learning_rate": {"type": "float", "low": 0.02, "high": 0.15, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 5},
            "subsample": {"type": "float", "low": 0.7, "high": 1.0},
        },
    )
    specs["lightgbm_regressor"] = RegressionModelSpec(
        id="lightgbm_regressor",
        display_name="LightGBM Regressor",
        plain_name="Leaf-Wise Histogram Tree Boosting Regressor",
        category="boosting",
        feature_set="agg",
        needs_scaling=False,
        twin_model_id="lightgbm_hybrid_regressor",
        n_tuning_trials=15,
        time_budget_seconds=25,
        search_space={
            "n_estimators": {"type": "int", "low": 60, "high": 150},
            "learning_rate": {"type": "float", "low": 0.02, "high": 0.15, "log": True},
            "num_leaves": {"type": "int", "low": 15, "high": 31},
        },
    )
    specs["catboost_regressor"] = RegressionModelSpec(
        id="catboost_regressor",
        display_name="CatBoost Regressor",
        plain_name="Oblivious Symmetric Tree Boosting Regressor",
        category="boosting",
        feature_set="agg",
        needs_scaling=False,
        twin_model_id="catboost_hybrid_regressor",
        n_tuning_trials=15,
        time_budget_seconds=30,
        search_space={
            "iterations": {"type": "int", "low": 80, "high": 180},
            "learning_rate": {"type": "float", "low": 0.02, "high": 0.15, "log": True},
            "depth": {"type": "int", "low": 3, "high": 5},
        },
    )
    specs["adaboost_regressor"] = RegressionModelSpec(
        id="adaboost_regressor",
        display_name="AdaBoost Regressor",
        plain_name="Adaptive Residual Reweighting Regressor",
        category="boosting",
        feature_set="agg",
        needs_scaling=False,
        n_tuning_trials=15,
        time_budget_seconds=20,
        search_space={
            "n_estimators": {"type": "int", "low": 30, "high": 100},
            "learning_rate": {"type": "float", "low": 0.05, "high": 0.3},
        },
    )

    # 5. Proposed State Hybrids (evaluated on agg_states)
    specs["xgboost_hybrid_regressor"] = RegressionModelSpec(
        id="xgboost_hybrid_regressor",
        display_name="Hybrid XGBoost Regressor [agg_states]",
        plain_name="Extreme Boosting with Unsupervised State Features",
        category="hybrid",
        feature_set="agg_states",
        needs_scaling=False,
        twin_model_id="xgboost_regressor",
        n_tuning_trials=15,
        time_budget_seconds=30,
        search_space={
            "n_estimators": {"type": "int", "low": 60, "high": 150},
            "learning_rate": {"type": "float", "low": 0.02, "high": 0.15, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 5},
        },
    )
    specs["lightgbm_hybrid_regressor"] = RegressionModelSpec(
        id="lightgbm_hybrid_regressor",
        display_name="Hybrid LightGBM Regressor [agg_states]",
        plain_name="LightGBM with Unsupervised State Features",
        category="hybrid",
        feature_set="agg_states",
        needs_scaling=False,
        twin_model_id="lightgbm_regressor",
        n_tuning_trials=15,
        time_budget_seconds=25,
        search_space={
            "n_estimators": {"type": "int", "low": 60, "high": 150},
            "learning_rate": {"type": "float", "low": 0.02, "high": 0.15, "log": True},
            "num_leaves": {"type": "int", "low": 15, "high": 31},
        },
    )
    specs["catboost_hybrid_regressor"] = RegressionModelSpec(
        id="catboost_hybrid_regressor",
        display_name="Hybrid CatBoost Regressor [agg_states]",
        plain_name="CatBoost with Unsupervised State Features",
        category="hybrid",
        feature_set="agg_states",
        needs_scaling=False,
        twin_model_id="catboost_regressor",
        n_tuning_trials=15,
        time_budget_seconds=30,
        search_space={
            "iterations": {"type": "int", "low": 80, "high": 180},
            "learning_rate": {"type": "float", "low": 0.02, "high": 0.15, "log": True},
            "depth": {"type": "int", "low": 3, "high": 5},
        },
    )
    specs["random_forest_hybrid_regressor"] = RegressionModelSpec(
        id="random_forest_hybrid_regressor",
        display_name="Hybrid Random Forest Regressor [agg_states]",
        plain_name="Random Forest with Unsupervised State Features",
        category="hybrid",
        feature_set="agg_states",
        needs_scaling=False,
        twin_model_id="random_forest_regressor",
        n_tuning_trials=15,
        time_budget_seconds=30,
        search_space={
            "n_estimators": {"type": "int", "low": 60, "high": 150},
            "max_depth": {"type": "int", "low": 4, "high": 10},
        },
    )
    specs["ridge_hybrid_regressor"] = RegressionModelSpec(
        id="ridge_hybrid_regressor",
        display_name="Hybrid Ridge Regressor [agg_states]",
        plain_name="Ridge Regression with Unsupervised State Features",
        category="hybrid",
        feature_set="agg_states",
        needs_scaling=True,
        twin_model_id="ridge_regressor",
        n_tuning_trials=15,
        time_budget_seconds=20,
        search_space={
            "alpha": {"type": "float", "low": 1e-3, "high": 1e3, "log": True}
        },
    )

    # 6. Deep Learning Regressors
    specs["mlp_wide_regressor"] = RegressionModelSpec(
        id="mlp_wide_regressor",
        display_name="Wide MLP Regressor",
        plain_name="Wide Feedforward Deep Regressor",
        category="deep",
        feature_set="agg_states",
        needs_scaling=True,
        n_tuning_trials=10,
        time_budget_seconds=30,
    )
    specs["mlp_residual_regressor"] = RegressionModelSpec(
        id="mlp_residual_regressor",
        display_name="Residual MLP Regressor",
        plain_name="Pre-Activation Skip-Connection Deep Regressor",
        category="deep",
        feature_set="agg_states",
        needs_scaling=True,
        n_tuning_trials=10,
        time_budget_seconds=30,
    )
    specs["gru_regressor"] = RegressionModelSpec(
        id="gru_regressor",
        display_name="Gated Recurrent Unit (GRU) Regressor",
        plain_name="Gated Sequential Recurrent Regressor",
        category="deep",
        feature_set="sequence",
        needs_scaling=False,
        n_tuning_trials=10,
        time_budget_seconds=35,
    )
    specs["lstm_regressor"] = RegressionModelSpec(
        id="lstm_regressor",
        display_name="Long Short-Term Memory (LSTM) Regressor",
        plain_name="Memory Cell Sequential Recurrent Regressor",
        category="deep",
        feature_set="sequence",
        needs_scaling=False,
        n_tuning_trials=10,
        time_budget_seconds=35,
    )
    specs["bi_gru_regressor"] = RegressionModelSpec(
        id="bi_gru_regressor",
        display_name="Bidirectional GRU Regressor",
        plain_name="Bidirectional Temporal Sequential Regressor",
        category="deep",
        feature_set="sequence",
        needs_scaling=False,
        n_tuning_trials=10,
        time_budget_seconds=35,
    )
    specs["cnn_1d_regressor"] = RegressionModelSpec(
        id="cnn_1d_regressor",
        display_name="1D Temporal CNN Regressor",
        plain_name="Temporal Convolutional Sequence Regressor",
        category="deep",
        feature_set="sequence",
        needs_scaling=False,
        n_tuning_trials=10,
        time_budget_seconds=30,
    )
    specs["tiny_transformer_regressor"] = RegressionModelSpec(
        id="tiny_transformer_regressor",
        display_name="Tiny Transformer Regressor",
        plain_name="Self-Attention Sequential Regressor",
        category="deep",
        feature_set="sequence",
        needs_scaling=False,
        n_tuning_trials=10,
        time_budget_seconds=30,
    )
    specs["hybrid_multimodal_regressor"] = RegressionModelSpec(
        id="hybrid_multimodal_regressor",
        display_name="Hybrid Multimodal Deep Regressor",
        plain_name="Joint Recurrent & Tabular State Deep Regressor",
        category="deep",
        feature_set="hybrid_multimodal",
        needs_scaling=False,
        n_tuning_trials=10,
        time_budget_seconds=35,
    )

    # 7. Ensembles
    specs["voting_regressor"] = RegressionModelSpec(
        id="voting_regressor",
        display_name="Average Voting Regressor",
        plain_name="Mean Continuous Prediction Ensemble",
        category="ensemble",
        feature_set="agg",
        needs_scaling=False,
        n_tuning_trials=0,
        time_budget_seconds=10,
    )
    specs["stacking_regressor"] = RegressionModelSpec(
        id="stacking_regressor",
        display_name="Stacking Regressor",
        plain_name="Meta-Learner Out-of-Fold Ridge Stacking Ensemble",
        category="ensemble",
        feature_set="agg",
        needs_scaling=False,
        n_tuning_trials=0,
        time_budget_seconds=15,
    )

    return specs


def create_base_regressor(
    spec: RegressionModelSpec,
    hyperparams: Optional[Dict[str, Any]] = None,
    random_seed: int = 42,
) -> BaseEstimator:
    """Instantiate a concrete scikit-learn, XGBoost, LightGBM, or CatBoost regressor."""
    params = dict(spec.fixed_params)
    if hyperparams:
        params.update(hyperparams)

    mid = spec.id

    # 1. Baselines
    if mid == "mean_regressor":
        return DummyRegressor(strategy="mean")
    if mid == "single_signal_linear":
        return LinearRegression()

    # 2. Linear Models
    if mid == "linear_regression":
        return LinearRegression()
    if mid in ["ridge_regressor", "ridge_hybrid_regressor"]:
        alpha = float(params.get("alpha", 1.0))
        return Ridge(alpha=alpha, random_state=random_seed)
    if mid == "lasso_regressor":
        alpha = float(params.get("alpha", 0.1))
        return Lasso(alpha=alpha, max_iter=2000, random_state=random_seed)
    if mid == "elasticnet_regressor":
        alpha = float(params.get("alpha", 0.1))
        l1_ratio = float(params.get("l1_ratio", 0.5))
        return ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=random_seed
        )
    if mid == "bayesian_ridge_regressor":
        return BayesianRidge(max_iter=300)
    if mid == "huber_regressor":
        epsilon = float(params.get("epsilon", 1.35))
        alpha = float(params.get("alpha", 0.0001))
        return HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=1000)

    # 3. Classical / Shallow
    if mid == "knn_regressor":
        k = int(params.get("n_neighbors", 7))
        weights = str(params.get("weights", "distance"))
        return KNeighborsRegressor(n_neighbors=k, weights=weights, n_jobs=-1)
    if mid == "svr_rbf":
        c_val = float(params.get("C", 10.0))
        eps = float(params.get("epsilon", 1.0))
        return SVR(C=c_val, epsilon=eps, kernel="rbf")
    if mid == "decision_tree_regressor":
        max_d = int(params.get("max_depth", 5))
        min_leaf = int(params.get("min_samples_leaf", 10))
        return DecisionTreeRegressor(
            max_depth=max_d, min_samples_leaf=min_leaf, random_state=random_seed
        )
    if mid == "mlp_shallow_regressor":
        alpha = float(params.get("alpha", 0.01))
        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            alpha=alpha,
            max_iter=400,
            early_stopping=True,
            random_state=random_seed,
        )
    if mid == "extra_trees_regressor":
        n_est = int(params.get("n_estimators", 100))
        max_d = int(params.get("max_depth", 8))
        return ExtraTreesRegressor(
            n_estimators=n_est, max_depth=max_d, random_state=random_seed, n_jobs=-1
        )

    # 4. Boosting & Tree Ensembles
    if mid in ["random_forest_regressor", "random_forest_hybrid_regressor"]:
        n_est = int(params.get("n_estimators", 100))
        max_d = int(params.get("max_depth", 8))
        min_leaf = int(params.get("min_samples_leaf", 5))
        return RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_leaf=min_leaf,
            random_state=random_seed,
            n_jobs=-1,
        )
    if mid == "gradient_boosting_regressor":
        n_est = int(params.get("n_estimators", 100))
        lr = float(params.get("learning_rate", 0.05))
        max_d = int(params.get("max_depth", 4))
        return GradientBoostingRegressor(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=max_d,
            random_state=random_seed,
        )
    if mid == "hist_gradient_boosting_regressor":
        lr = float(params.get("learning_rate", 0.05))
        max_iter = int(params.get("max_iter", 100))
        max_d = int(params.get("max_depth", 5))
        return HistGradientBoostingRegressor(
            learning_rate=lr,
            max_iter=max_iter,
            max_depth=max_d,
            random_state=random_seed,
        )
    if mid in ["xgboost_regressor", "xgboost_hybrid_regressor"]:
        if xgb is None:
            raise ImportError("xgboost package is not installed.")
        n_est = int(params.get("n_estimators", 100))
        lr = float(params.get("learning_rate", 0.05))
        max_d = int(params.get("max_depth", 4))
        sub = float(params.get("subsample", 0.9))
        return xgb.XGBRegressor(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=max_d,
            subsample=sub,
            random_state=random_seed,
            n_jobs=-1,
        )
    if mid in ["lightgbm_regressor", "lightgbm_hybrid_regressor"]:
        if lgb is None:
            raise ImportError("lightgbm package is not installed.")
        n_est = int(params.get("n_estimators", 100))
        lr = float(params.get("learning_rate", 0.05))
        leaves = int(params.get("num_leaves", 20))
        return lgb.LGBMRegressor(
            n_estimators=n_est,
            learning_rate=lr,
            num_leaves=leaves,
            random_state=random_seed,
            verbose=-1,
            n_jobs=-1,
        )
    if mid.startswith("catboost_regressor") or mid.startswith(
        "catboost_hybrid_regressor"
    ):
        if cb is None:
            raise ImportError("catboost package is not installed.")
        iters = int(params.get("iterations", 150))
        lr = float(params.get("learning_rate", 0.05))
        depth = int(params.get("depth", 4))
        l2_reg = float(params.get("l2_leaf_reg", 3.0))
        return cb.CatBoostRegressor(
            iterations=iters,
            learning_rate=lr,
            depth=depth,
            l2_leaf_reg=l2_reg,
            random_seed=random_seed,
            verbose=False,
        )
    if mid == "adaboost_regressor":
        n_est = int(params.get("n_estimators", 50))
        lr = float(params.get("learning_rate", 0.1))
        return AdaBoostRegressor(
            n_estimators=n_est, learning_rate=lr, random_state=random_seed
        )

    raise ValueError(f"Unknown regression model id: '{mid}'")
