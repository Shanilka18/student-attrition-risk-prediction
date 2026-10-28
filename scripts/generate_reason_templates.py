"""Script to generate config/reason_templates.yaml with plain, empathetic,
non-judgemental explanations for every engineered feature in the MEI ecosystem.
"""

from pathlib import Path
import json
import yaml

INDICATOR_METADATA = {
    "resource_access": {
        "noun": "course learning materials and resources",
        "action": "accessing course materials",
        "short": "learning material access",
    },
    "study_time": {
        "noun": "dedicated independent study hours",
        "action": "independent study time",
        "short": "study hours",
    },
    "review_time": {
        "noun": "lecture review and recap time",
        "action": "reviewing course content after lectures",
        "short": "review time",
    },
    "quiz_practice": {
        "noun": "practice quizzes and self-assessment exercises",
        "action": "engaging with practice quizzes",
        "short": "practice quiz activity",
    },
    "assignment_engagement": {
        "noun": "assignment preparation and timely completion",
        "action": "working steadily on assignments",
        "short": "assignment preparation",
    },
    "study_consistency": {
        "noun": "study schedule regularity and weekly pacing",
        "action": "maintaining a consistent study routine",
        "short": "study schedule consistency",
    },
    "participation": {
        "noun": "discussion, forum, and classroom interactive activities",
        "action": "participating in interactive sessions and discussions",
        "short": "interactive participation",
    },
}

def generate_templates():
    features_path = Path("artifacts/runs/run_20261026_124514_1407b3/champion/features.json")
    if features_path.exists():
        with open(features_path, "r", encoding="utf-8") as f:
            features = json.load(f)
    else:
        features = []

    templates = {}

    for feat in features:
        # Check raw indicator per period
        raw_match = False
        for ind_key, meta in INDICATOR_METADATA.items():
            for p in [1, 2, 3, 4]:
                if feat == f"{ind_key}_p{p}":
                    raw_match = True
                    templates[feat] = {
                        "plain_name": f"{meta['short'].capitalize()} (Period {p})",
                        "risk_increasing": f"Engagement with {meta['noun']} was relatively low during period {p}",
                        "risk_decreasing": f"Maintained healthy engagement with {meta['noun']} during period {p}",
                    }
                    break
            if raw_match:
                break
        if raw_match:
            continue

        # Check missing indicators
        if feat.startswith("missing_"):
            clean_name = feat.replace("missing_", "")
            period_str = clean_name.split("_")[-1]  # p1..p4
            p_num = period_str.replace("p", "")
            base_ind = clean_name.replace(f"_{period_str}", "")
            meta = INDICATOR_METADATA.get(base_ind, {"noun": "weekly review responses", "short": "review responses"})
            templates[feat] = {
                "plain_name": f"Missing Response on {meta['short'].capitalize()} (Period {p_num})",
                "risk_increasing": f"Did not submit response regarding {meta['noun']} in period {p_num}",
                "risk_decreasing": f"Provided complete survey feedback on {meta['noun']} in period {p_num}",
            }
            continue

        # Check composite per period
        if feat in ["composite_p1", "composite_p2", "composite_p3", "composite_p4"]:
            p = feat.split("_")[-1].replace("p", "")
            templates[feat] = {
                "plain_name": f"Overall Engagement Level (Period {p})",
                "risk_increasing": f"Composite engagement across course activities was low in period {p}",
                "risk_decreasing": f"Maintained positive composite engagement across activities in period {p}",
            }
            continue

        # Check composite trajectory summary statistics
        if feat.startswith("composite_"):
            suffix = feat.replace("composite_", "")
            if suffix == "mean":
                templates[feat] = {
                    "plain_name": "Overall Term Engagement Level",
                    "risk_increasing": "Overall multi-term engagement level across all course activities was lower than average",
                    "risk_decreasing": "Maintained a strong overall multi-term engagement level across all course activities",
                }
            elif suffix == "std":
                templates[feat] = {
                    "plain_name": "Overall Engagement Volatility",
                    "risk_increasing": "Overall course engagement experienced substantial variability across terms",
                    "risk_decreasing": "Maintained steady, even engagement levels across terms without sudden swings",
                }
            elif suffix == "slope":
                templates[feat] = {
                    "plain_name": "Overall Engagement Trajectory",
                    "risk_increasing": "Overall engagement trajectory shows a downward trend over the course terms",
                    "risk_decreasing": "Overall engagement trajectory shows stable or increasing momentum over terms",
                }
            elif suffix == "max_drop":
                templates[feat] = {
                    "plain_name": "Maximum Term-to-Term Drop",
                    "risk_increasing": "Experienced a noticeable decline in engagement between consecutive terms",
                    "risk_decreasing": "Avoided sharp engagement declines between consecutive terms",
                }
            elif suffix == "reversals":
                templates[feat] = {
                    "plain_name": "Engagement Trend Fluctuations",
                    "risk_increasing": "Overall engagement pattern shifted back and forth across terms",
                    "risk_decreasing": "Engagement pattern followed a consistent and steady progression across terms",
                }
            elif suffix == "auc":
                templates[feat] = {
                    "plain_name": "Cumulative Term Engagement",
                    "risk_increasing": "Total cumulative engagement across all academic periods was lower than peer average",
                    "risk_decreasing": "Accumulated substantial continuous engagement across all academic periods",
                }
            elif suffix == "last_value":
                templates[feat] = {
                    "plain_name": "Recent Overall Engagement (Period 4)",
                    "risk_increasing": "Most recent overall engagement level (period 4) ended below average",
                    "risk_decreasing": "Concluded the final term (period 4) with solid, positive overall engagement",
                }
            elif suffix == "drop_from_best_to_last":
                templates[feat] = {
                    "plain_name": "Drop from Peak Engagement",
                    "risk_increasing": "Recent engagement declined substantially compared to the student's highest-performing term",
                    "risk_decreasing": "Maintained recent engagement close to peak historical levels",
                }
            else:
                templates[feat] = {
                    "plain_name": f"Overall {suffix.replace('_', ' ').title()}",
                    "risk_increasing": f"Overall composite {suffix.replace('_', ' ')} indicates potential engagement decline",
                    "risk_decreasing": f"Overall composite {suffix.replace('_', ' ')} reflects positive engagement",
                }
            continue

        # Check indicator aggregate dynamics
        matched_agg = False
        for ind_key, meta in INDICATOR_METADATA.items():
            if feat.startswith(f"{ind_key}_"):
                matched_agg = True
                suffix = feat.replace(f"{ind_key}_", "")
                if suffix == "mean":
                    templates[feat] = {
                        "plain_name": f"Average {meta['short'].capitalize()}",
                        "risk_increasing": f"Average time and effort devoted to {meta['noun']} was below class average",
                        "risk_decreasing": f"Consistently devoted steady time and effort to {meta['noun']}",
                    }
                elif suffix == "std":
                    templates[feat] = {
                        "plain_name": f"Variability in {meta['short'].capitalize()}",
                        "risk_increasing": f"Activity in {meta['noun']} fluctuated considerably from term to term",
                        "risk_decreasing": f"Maintained stable and predictable participation in {meta['noun']}",
                    }
                elif suffix == "min":
                    templates[feat] = {
                        "plain_name": f"Lowest {meta['short'].capitalize()}",
                        "risk_increasing": f"Experienced at least one period with minimal engagement in {meta['noun']}",
                        "risk_decreasing": f"Maintained an acceptable baseline of engagement with {meta['noun']} even in low periods",
                    }
                elif suffix == "max":
                    templates[feat] = {
                        "plain_name": f"Peak {meta['short'].capitalize()}",
                        "risk_increasing": f"Peak engagement with {meta['noun']} remained modest across all terms",
                        "risk_decreasing": f"Demonstrated high peak capability and strong engagement with {meta['noun']}",
                    }
                elif suffix == "slope":
                    templates[feat] = {
                        "plain_name": f"Trajectory Slope of {meta['short'].capitalize()}",
                        "risk_increasing": f"Participation in {meta['noun']} showed a downward trend from period 1 to period 4",
                        "risk_decreasing": f"Participation in {meta['noun']} remained steady or improved from period 1 to period 4",
                    }
                elif suffix == "last_minus_first":
                    templates[feat] = {
                        "plain_name": f"Net Change in {meta['short'].capitalize()} (P4 vs P1)",
                        "risk_increasing": f"Engagement with {meta['noun']} ended noticeably lower in period 4 than in period 1",
                        "risk_decreasing": f"Engagement with {meta['noun']} in period 4 matched or surpassed period 1 levels",
                    }
                elif suffix == "last_minus_mean_first_two":
                    templates[feat] = {
                        "plain_name": f"Recent vs Early {meta['short'].capitalize()}",
                        "risk_increasing": f"Recent activity in {meta['noun']} dropped compared to the first half of the course",
                        "risk_decreasing": f"Recent activity in {meta['noun']} remained strong relative to early terms",
                    }
                elif suffix == "reversals":
                    templates[feat] = {
                        "plain_name": f"Direction Changes in {meta['short'].capitalize()}",
                        "risk_increasing": f"Effort in {meta['noun']} alternated up and down frequently between periods",
                        "risk_decreasing": f"Effort in {meta['noun']} followed a smooth, consistent trajectory",
                    }
                else:
                    templates[feat] = {
                        "plain_name": f"{meta['short'].capitalize()} ({suffix})",
                        "risk_increasing": f"Pattern in {meta['noun']} reflects reduced engagement",
                        "risk_decreasing": f"Pattern in {meta['noun']} reflects positive engagement",
                    }
                break

        if not matched_agg:
            # Fallback for any other feature
            clean_feat = feat.replace("_", " ").title()
            templates[feat] = {
                "plain_name": clean_feat,
                "risk_increasing": f"Metrics for {clean_feat} indicate lower than expected student participation",
                "risk_decreasing": f"Metrics for {clean_feat} indicate positive and steady student participation",
            }

    # Add state features in case hybrid model is explained
    for s in [0, 1, 2]:
        state_label = "Low" if s == 0 else ("Moderate" if s == 1 else "High")
        templates[f"hmm_prob_state_{s}"] = {
            "plain_name": f"Probability of {state_label} Engagement State",
            "risk_increasing": f"High probability of occupying the {state_label.lower()}-engagement behavioral state",
            "risk_decreasing": f"Low probability of falling into the {state_label.lower()}-engagement behavioral state",
        }
        templates[f"hmm_current_state_{s}"] = {
            "plain_name": f"Current {state_label} Engagement State Indicator",
            "risk_increasing": f"Currently categorized in the {state_label.lower()}-engagement state in the most recent term",
            "risk_decreasing": f"Not categorized in the {state_label.lower()}-engagement state in the most recent term",
        }
        templates[f"hmm_dwell_state_{s}"] = {
            "plain_name": f"Dwell Time in {state_label} State",
            "risk_increasing": f"Spent an extended duration in the {state_label.lower()}-engagement behavioral state",
            "risk_decreasing": f"Spent minimal or zero duration in the {state_label.lower()}-engagement behavioral state",
        }

    templates["hmm_n_transitions"] = {
        "plain_name": "State Transition Count",
        "risk_increasing": "Frequent transitions between differing engagement states across terms",
        "risk_decreasing": "Stable behavioral state continuity across academic terms",
    }
    templates["hmm_entropy"] = {
        "plain_name": "Engagement State Uncertainty (Entropy)",
        "risk_increasing": "High behavioral unpredictability across different engagement states",
        "risk_decreasing": "Clear and predictable behavioral state adherence across terms",
    }

    for k in range(6):
        templates[f"dist_cluster_{k}"] = {
            "plain_name": f"Distance to Behavioral Profile Cluster {k+1}",
            "risk_increasing": f"Longitudinal profile aligns closely with higher-risk historical cluster {k+1}",
            "risk_decreasing": f"Longitudinal profile aligns closely with resilient historical cluster {k+1}",
        }
        templates[f"cluster_affinity_{k}"] = {
            "plain_name": f"Affinity to Behavioral Profile Cluster {k+1}",
            "risk_increasing": f"High membership affinity with historical pattern cluster {k+1}",
            "risk_decreasing": f"Low membership affinity with vulnerable historical cluster {k+1}",
        }

    out_path = Path("config/reason_templates.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(templates, f, sort_keys=True, allow_unicode=True, indent=2)

    print(f"Generated {len(templates)} reason templates in {out_path}")

if __name__ == "__main__":
    generate_templates()
