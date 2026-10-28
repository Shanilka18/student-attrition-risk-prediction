"""Candidate models, model zoo registry, tuning, benchmarking engine, and diagnostic figures."""

from mei.models.engine import (
    assemble_and_save_leaderboard,
    build_pipeline_for_spec,
    compute_boosting_iteration_losses,
    compute_learning_curve_data,
    compute_validation_curve_data,
    evaluate_single_model,
    evaluate_stacking_ensemble,
    evaluate_voting_ensemble,
    run_smote_vs_class_weights,
    save_model_artifacts,
)
from mei.models.figures import (
    plot_boosting_loss_curves,
    plot_calibration_curves,
    plot_generalization_gap,
    plot_leaderboard_bars,
    plot_learning_curves,
    plot_metric_heatmap,
    plot_pr_curves_by_category,
    plot_roc_curves_by_category,
    plot_smote_comparison,
    plot_validation_curves,
)
from mei.models.registry import (
    ModelSpec,
    RuleLoginImitationClassifier,
    create_base_estimator,
    get_all_model_specs,
    load_model_zoo_config,
)
from mei.models.tuner import tune_model

__all__ = [
    "ModelSpec",
    "RuleLoginImitationClassifier",
    "load_model_zoo_config",
    "get_all_model_specs",
    "create_base_estimator",
    "tune_model",
    "build_pipeline_for_spec",
    "compute_learning_curve_data",
    "compute_validation_curve_data",
    "compute_boosting_iteration_losses",
    "evaluate_single_model",
    "evaluate_voting_ensemble",
    "evaluate_stacking_ensemble",
    "run_smote_vs_class_weights",
    "save_model_artifacts",
    "assemble_and_save_leaderboard",
    "plot_leaderboard_bars",
    "plot_generalization_gap",
    "plot_metric_heatmap",
    "plot_roc_curves_by_category",
    "plot_pr_curves_by_category",
    "plot_calibration_curves",
    "plot_boosting_loss_curves",
    "plot_learning_curves",
    "plot_validation_curves",
    "plot_smote_comparison",
]
