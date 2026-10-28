"""Deep Learning models and training package for Multimodal Student Engagement Intelligence."""

from mei.models.deep.architectures import (
    BiGRUModel,
    CNN1DModel,
    GaussianNoise,
    GRUModel,
    HybridMultimodalModel,
    LSTMModel,
    MLPResidual,
    MLPWide,
    OverparameterizedMLP,
    TinyTransformer,
    build_deep_model,
)
from mei.models.deep.engine import (
    compute_seed_variance_deep,
    evaluate_single_deep_model,
    prepare_fold_datasets,
    train_overfit_demonstration_model,
)
from mei.models.deep.figures import (
    plot_all_deep_validation_loss_comparison,
    plot_cross_fold_variance,
    plot_deep_calibration_curves,
    plot_deep_twin_curves,
    plot_overfit_demonstration,
)
from mei.models.deep.trainer import (
    DeepEngagementDataset,
    evaluate_deep_epoch,
    predict_proba_deep,
    train_two_stage,
)
from mei.models.deep.tuner import tune_deep_model

__all__ = [
    "GaussianNoise",
    "MLPWide",
    "MLPResidual",
    "GRUModel",
    "LSTMModel",
    "BiGRUModel",
    "CNN1DModel",
    "TinyTransformer",
    "HybridMultimodalModel",
    "OverparameterizedMLP",
    "build_deep_model",
    "DeepEngagementDataset",
    "evaluate_deep_epoch",
    "train_two_stage",
    "predict_proba_deep",
    "tune_deep_model",
    "prepare_fold_datasets",
    "evaluate_single_deep_model",
    "train_overfit_demonstration_model",
    "compute_seed_variance_deep",
    "plot_deep_twin_curves",
    "plot_all_deep_validation_loss_comparison",
    "plot_overfit_demonstration",
    "plot_deep_calibration_curves",
    "plot_cross_fold_variance",
]
