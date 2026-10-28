"""Feature mapping utilities for the MEI explainability subsystem.

Maps 100+ engineered features back to canonical indicators, academic periods,
and feature families (raw answers, longitudinal aggregates/drift, state features).
"""

from typing import Any, Dict, List

import yaml

from mei.config import CONFIG_DIR


def get_feature_group_mapping(feature_names: List[str]) -> Dict[str, Dict[str, str]]:
    """Map every feature name to its indicator, academic period, and feature family.

    Returns:
        Dict mapping feature_name -> {
            'indicator': str,
            'period': str,
            'family': 'raw_answers' | 'aggregates_drift' | 'state_features',
            'plain_name': str
        }
    """
    templates_path = CONFIG_DIR / "reason_templates.yaml"
    templates: Dict[str, Any] = {}
    if templates_path.exists():
        with open(templates_path, "r", encoding="utf-8") as f:
            templates = yaml.safe_load(f) or {}

    mapping = {}

    indicators = [
        "resource_access",
        "study_time",
        "review_time",
        "quiz_practice",
        "assignment_engagement",
        "study_consistency",
        "participation",
    ]

    for feat in feature_names:
        plain_name = templates.get(feat, {}).get(
            "plain_name", feat.replace("_", " ").title()
        )

        # 1. State features
        if (
            feat.startswith("hmm_")
            or feat.startswith("dist_cluster_")
            or feat.startswith("cluster_affinity_")
        ):
            mapping[feat] = {
                "indicator": "engagement_state",
                "period": "Overall",
                "family": "state_features",
                "plain_name": plain_name,
            }
            continue

        # 2. Raw answers (indicators P1-P4 and missingness)
        is_raw = False
        for ind in indicators:
            for p in [1, 2, 3, 4]:
                if feat == f"{ind}_p{p}":
                    mapping[feat] = {
                        "indicator": ind,
                        "period": f"P{p}",
                        "family": "raw_answers",
                        "plain_name": plain_name,
                    }
                    is_raw = True
                    break
            if is_raw:
                break
        if is_raw:
            continue

        if feat.startswith("missing_"):
            clean = feat.replace("missing_", "")
            period_token = clean.split("_")[-1].upper()  # P1..P4
            ind_base = clean.replace(f"_{clean.split('_')[-1]}", "")
            mapping[feat] = {
                "indicator": ind_base,
                "period": period_token,
                "family": "raw_answers",
                "plain_name": plain_name,
            }
            continue

        # 3. Composite indicators
        if feat.startswith("composite_"):
            sub = feat.replace("composite_", "")
            if sub in ["p1", "p2", "p3", "p4"]:
                mapping[feat] = {
                    "indicator": "composite",
                    "period": sub.upper(),
                    "family": "aggregates_drift",
                    "plain_name": plain_name,
                }
            else:
                mapping[feat] = {
                    "indicator": "composite",
                    "period": "Overall",
                    "family": "aggregates_drift",
                    "plain_name": plain_name,
                }
            continue

        # 4. Aggregated indicator dynamics
        matched_ind = False
        for ind in indicators:
            if feat.startswith(f"{ind}_"):
                mapping[feat] = {
                    "indicator": ind,
                    "period": "Overall",
                    "family": "aggregates_drift",
                    "plain_name": plain_name,
                }
                matched_ind = True
                break

        if not matched_ind:
            mapping[feat] = {
                "indicator": "other",
                "period": "Overall",
                "family": "aggregates_drift",
                "plain_name": plain_name,
            }

    return mapping


def compute_grouped_importance(
    mean_abs_shap: Dict[str, float],
    feature_mapping: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate mean |SHAP| values by indicator, period, and feature family.

    Returns:
        Dict with keys 'by_indicator', 'by_period', 'by_family' containing
        total importance and percentage of total importance.
    """
    total_shap = sum(mean_abs_shap.values()) or 1.0

    by_indicator: Dict[str, float] = {}
    by_period: Dict[str, float] = {}
    by_family: Dict[str, float] = {}

    for feat, val in mean_abs_shap.items():
        meta = feature_mapping.get(
            feat,
            {
                "indicator": "other",
                "period": "Overall",
                "family": "aggregates_drift",
            },
        )
        ind = meta["indicator"]
        p = meta["period"]
        fam = meta["family"]

        by_indicator[ind] = by_indicator.get(ind, 0.0) + val
        by_period[p] = by_period.get(p, 0.0) + val
        by_family[fam] = by_family.get(fam, 0.0) + val

    def to_shares(d: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        res = {}
        for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True):
            res[k] = {
                "total_shap": round(float(v), 6),
                "share_pct": round(float(v / total_shap * 100.0), 2),
            }
        return res

    return {
        "by_indicator": to_shares(by_indicator),
        "by_period": to_shares(by_period),
        "by_family": to_shares(by_family),
    }
