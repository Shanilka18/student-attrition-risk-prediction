"""Psychosocial Risk Index (PRI) Explainability and Attribution Engine.

Implements:
- Automated SHAP Explainer selection (Tree, Linear, Kernel, Permutation).
- Global explanations: summary bar, beeswarm, factor-level aggregations, dependence plots.
- Local explanations: per-student top 3 risk-increasing and top 3 risk-decreasing factors.
- Deterministic, template-driven plain-language narrative generator with non-clinical disclaimers.
- Rigorous sanity checks: additivity verification, stability, permutation agreement, synthetic ground-truth recovery.
"""

from src.explain.narrative import generate_student_narrative, DISCLAIMER_TEXT
from src.explain.explainer import build_explainer, explain_dataset, save_shap_artifacts
from src.explain.local_explain import explain_student_instance, export_student_waterfall_json
from src.explain.global_explain import (
    compute_global_importance,
    aggregate_factor_importance,
    plot_global_summary_bar,
    plot_beeswarm,
    plot_dependence,
)
from src.explain.sanity_checks import run_all_sanity_checks

__all__ = [
    "build_explainer",
    "explain_dataset",
    "save_shap_artifacts",
    "explain_student_instance",
    "export_student_waterfall_json",
    "compute_global_importance",
    "aggregate_factor_importance",
    "plot_global_summary_bar",
    "plot_beeswarm",
    "plot_dependence",
    "generate_student_narrative",
    "DISCLAIMER_TEXT",
    "run_all_sanity_checks",
]
