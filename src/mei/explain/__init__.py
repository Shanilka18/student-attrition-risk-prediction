"""Explainability subsystem for the Multimodal Engagement Intelligence (MEI) framework."""

from mei.explain.engine import (
    compute_cross_method_comparison,
    compute_global_shap_audit,
    compute_pattern_shap_breakdown,
    compute_shap_consistency_across_folds,
    explain_dataframe,
    get_champion_explainer,
)
from mei.explain.mapping import compute_grouped_importance, get_feature_group_mapping
from mei.explain.reasons import (
    convert_shap_to_mei_points,
    generate_student_reasons,
    load_reason_templates,
)

__all__ = [
    "get_champion_explainer",
    "compute_global_shap_audit",
    "compute_shap_consistency_across_folds",
    "compute_cross_method_comparison",
    "compute_pattern_shap_breakdown",
    "explain_dataframe",
    "get_feature_group_mapping",
    "compute_grouped_importance",
    "load_reason_templates",
    "convert_shap_to_mei_points",
    "generate_student_reasons",
]
