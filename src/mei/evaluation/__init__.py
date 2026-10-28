"""Evaluation routines, metrics calculations, and diagnostics."""

from mei.evaluation.metrics import (
    CurveData,
    EvaluationResult,
    ProbabilisticMetrics,
    ThresholdMetrics,
    compute_expected_calibration_error,
    compute_leaderboard_score,
    compute_probabilistic_metrics_and_curves,
    compute_specificity,
    compute_threshold_metrics,
    evaluate_predictions,
    find_optimal_threshold,
)
from mei.evaluation.regression_metrics import (
    ClassificationAt60,
    RegressionMetrics,
    compute_adjusted_r2,
    compute_breusch_pagan_test,
    compute_classification_at_60,
    compute_regression_metrics,
    compute_top_models_wilcoxon,
    compute_track_a_track_b_consistency,
    compute_track_b_leaderboard_score,
)

__all__ = [
    "ThresholdMetrics",
    "ProbabilisticMetrics",
    "CurveData",
    "EvaluationResult",
    "compute_expected_calibration_error",
    "compute_specificity",
    "compute_threshold_metrics",
    "find_optimal_threshold",
    "compute_probabilistic_metrics_and_curves",
    "evaluate_predictions",
    "compute_leaderboard_score",
    "RegressionMetrics",
    "ClassificationAt60",
    "compute_adjusted_r2",
    "compute_regression_metrics",
    "compute_classification_at_60",
    "compute_track_b_leaderboard_score",
    "compute_breusch_pagan_test",
    "compute_top_models_wilcoxon",
    "compute_track_a_track_b_consistency",
]
