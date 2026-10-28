"""Feature Engineering, Preprocessing, and Evaluation Splits Module."""

from mei.features.diagnostics import (
    audit_leakage,
    compute_feature_diagnostics,
    compute_vif_report,
    render_feature_diagnostic_figures,
    run_cv_benchmark,
)
from mei.features.pipeline import (
    ColumnSelector,
    build_feature_pipeline,
    compute_cohort_class_weights,
    extract_features,
)
from mei.features.splits import (
    create_cv_folds,
    create_outer_split,
    load_cv_folds,
    load_test_indices,
    load_train_indices,
    load_val_indices,
    save_cv_folds,
    save_splits,
)
from mei.features.transformers import (
    ConstantNearConstantFilter,
    EngagementFeatures,
    EngagementImputer,
    PeriodReshaper,
    get_canonical_28_columns,
)

__all__ = [
    "create_outer_split",
    "save_splits",
    "load_train_indices",
    "load_val_indices",
    "load_test_indices",
    "create_cv_folds",
    "save_cv_folds",
    "load_cv_folds",
    "PeriodReshaper",
    "EngagementImputer",
    "EngagementFeatures",
    "ConstantNearConstantFilter",
    "get_canonical_28_columns",
    "ColumnSelector",
    "build_feature_pipeline",
    "extract_features",
    "compute_cohort_class_weights",
    "audit_leakage",
    "compute_feature_diagnostics",
    "compute_vif_report",
    "run_cv_benchmark",
    "render_feature_diagnostic_figures",
]
