"""
Leaderboard Construction, Champion Selection, and Research Question Adjudication.

Computes weighted composite ranks across the full metric suite, applies
eligibility constraints (NFR1 latency, seed stability), selects the overall
champion and per-category winners, and formally answers the proposal's
core research question with statistical backing.

All outputs are labelled SYNTHETIC until real data replaces them.

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.src.config import config

logger = logging.getLogger("AcademicIntelligence.Leaderboard")

# Models that are reference-only and ineligible for champion status
_REFERENCE_ONLY_KEYS = {"gpa_threshold"}


def build_leaderboard(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the master leaderboard from test evaluation results.

    Parameters
    ----------
    evaluation_results : dict
        The dict returned by ``run_full_evaluation()`` containing:
        - models: per-model evaluation dicts (with metrics, calibrated_probs, etc.)
        - statistical_tests: bootstrap CIs and paired tests
        - key_models: identified sequence champion, boosting baseline, etc.

    Returns
    -------
    dict with: leaderboard_df, champion, research_question_answer, category_champions
    """
    logger.info("=" * 72)
    logger.info("LEADERBOARD CONSTRUCTION & CHAMPION SELECTION")
    logger.info("DATA SOURCE: SYNTHETIC — Results demonstrate methodology only")
    logger.info("=" * 72)

    all_models = evaluation_results.get("models", {})
    # Prefer serializable_models if models contain numpy arrays
    serializable = evaluation_results.get("serializable_models", {})
    stat_tests = evaluation_results.get("statistical_tests", {})
    key_models = evaluation_results.get("key_models", {})

    # ── 1. Load leaderboard configuration ─────────────────────────────────
    lb_cfg = config._raw.get("leaderboard", {})
    weights = lb_cfg.get("weights", {
        "pr_auc": 0.25, "roc_auc": 0.20, "f2_score": 0.20,
        "brier_score": 0.15, "expected_calibration_error": 0.10,
        "lead_time_semesters": 0.10,
    })
    constraints = lb_cfg.get("constraints", {
        "max_inference_time_sec_per_student": 2.0,
        "max_seed_std": 0.05,
    })
    max_latency_ms = constraints.get("max_inference_time_sec_per_student", 2.0) * 1000.0
    max_seed_std = constraints.get("max_seed_std", 0.05)

    # ── 2. Build DataFrame ────────────────────────────────────────────────
    rows: List[Dict[str, Any]] = []

    for key, res in (serializable or all_models).items():
        metrics = res.get("metrics", {})
        clf = metrics.get("classification", {})
        riq = metrics.get("risk_index_quality", {})
        ew = metrics.get("early_warning", {})
        eff = metrics.get("efficiency", {})

        rows.append({
            "model_key": key,
            "category": res.get("category", "?"),
            "label": res.get("label", key),
            "is_proposal_model": res.get("is_proposal_model", False),
            "is_sequence_model": res.get("is_sequence_model", False),
            # Metrics used for ranking
            "test_roc_auc": clf.get("roc_auc", 0.0),
            "test_pr_auc": clf.get("pr_auc", 0.0),
            "test_f2_score": clf.get("f2_score", 0.0),
            "test_brier_score": riq.get("brier_score", 1.0),
            "test_ece": riq.get("expected_calibration_error", 1.0),
            "test_lead_time": ew.get("average_lead_time_semesters", 0.0),
            # Additional display metrics
            "test_f1": clf.get("f1_score", 0.0),
            "test_recall": clf.get("recall", 0.0),
            "test_precision": clf.get("precision", 0.0),
            "test_balanced_acc": clf.get("balanced_accuracy", 0.0),
            "test_mcc": clf.get("matthews_corrcoef", 0.0),
            "test_bss": riq.get("brier_skill_score", 0.0),
            "test_pseudo_r2": riq.get("pseudo_r2", 0.0),
            "test_spearman": riq.get("spearman_correlation", 0.0),
            # Efficiency
            "inference_time_ms": eff.get("inference_time_ms_per_student", 0.0),
            "model_size_kb": eff.get("model_size_kb", 0.0),
            # Stability
            "seed_std": res.get("seed_std", 0.0),
            "threshold": res.get("threshold", 0.5),
            "validation_roc_auc": res.get("validation_roc_auc", 0.0),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("No models to rank — leaderboard is empty.")
        return {"leaderboard_df": df, "champion": None, "research_question_answer": None}

    # ── 3. Compute metric ranks ───────────────────────────────────────────
    # Higher is better: rank descending (rank 1 = highest value)
    for col in ["test_roc_auc", "test_pr_auc", "test_f2_score", "test_lead_time"]:
        df[f"rank_{col}"] = df[col].rank(ascending=False, method="min")

    # Lower is better: rank ascending (rank 1 = lowest value)
    for col in ["test_brier_score", "test_ece"]:
        df[f"rank_{col}"] = df[col].rank(ascending=True, method="min")

    # ── 4. Weighted composite rank ────────────────────────────────────────
    metric_to_rank_col = {
        "pr_auc": "rank_test_pr_auc",
        "roc_auc": "rank_test_roc_auc",
        "f2_score": "rank_test_f2_score",
        "brier_score": "rank_test_brier_score",
        "expected_calibration_error": "rank_test_ece",
        "lead_time_semesters": "rank_test_lead_time",
    }

    df["composite_rank"] = 0.0
    for metric_name, w in weights.items():
        rank_col = metric_to_rank_col.get(metric_name)
        if rank_col and rank_col in df.columns:
            df["composite_rank"] += w * df[rank_col]

    # ── 5. Eligibility constraints ────────────────────────────────────────
    df["eligible"] = True
    df.loc[df["inference_time_ms"] > max_latency_ms, "eligible"] = False
    df.loc[df["seed_std"] > max_seed_std, "eligible"] = False
    df.loc[df["model_key"].isin(_REFERENCE_ONLY_KEYS), "eligible"] = False

    ineligible_count = int((~df["eligible"]).sum())
    if ineligible_count > 0:
        ineligible_keys = df.loc[~df["eligible"], "model_key"].tolist()
        logger.info(f"Ineligible models ({ineligible_count}): {ineligible_keys}")

    # ── 6. Sort and select champion ───────────────────────────────────────
    df = df.sort_values("composite_rank", ascending=True).reset_index(drop=True)
    df["overall_rank"] = range(1, len(df) + 1)

    eligible_df = df[df["eligible"]]
    if eligible_df.empty:
        logger.warning("No eligible models! Selecting best ineligible as fallback.")
        champion_row = df.iloc[0]
    else:
        champion_row = eligible_df.iloc[0]

    champion_key = champion_row["model_key"]
    logger.info(
        f"🏆 CHAMPION: {champion_key} "
        f"(ROC-AUC={champion_row['test_roc_auc']:.4f}, "
        f"PR-AUC={champion_row['test_pr_auc']:.4f}, "
        f"Composite Rank={champion_row['composite_rank']:.2f})"
    )

    # ── 7. Category champions ─────────────────────────────────────────────
    category_champions: Dict[str, Dict[str, Any]] = {}
    for cat in sorted(df["category"].unique()):
        cat_df = df[df["category"] == cat].sort_values("composite_rank")
        if not cat_df.empty:
            best = cat_df.iloc[0]
            category_champions[cat] = {
                "model_key": best["model_key"],
                "label": best["label"],
                "test_roc_auc": round(float(best["test_roc_auc"]), 4),
                "test_pr_auc": round(float(best["test_pr_auc"]), 4),
                "test_f2_score": round(float(best["test_f2_score"]), 4),
                "composite_rank": round(float(best["composite_rank"]), 2),
            }
            logger.info(f"  Category {cat} champion: {best['model_key']} (ROC-AUC={best['test_roc_auc']:.4f})")

    # ── 8. Research question adjudication ─────────────────────────────────
    seq_champion_key = key_models.get("sequence_champion")
    boost_baseline_key = key_models.get("boosting_baseline")

    research_answer = _adjudicate_research_question(
        df=df,
        seq_key=seq_champion_key,
        boost_key=boost_baseline_key,
        stat_tests=stat_tests,
    )

    # ── 9. Build champion detail record ───────────────────────────────────
    champion_detail = {
        "data_source": "SYNTHETIC",
        "model_key": champion_key,
        "category": str(champion_row["category"]),
        "label": str(champion_row["label"]),
        "is_proposal_model": bool(champion_row["is_proposal_model"]),
        "is_sequence_model": bool(champion_row["is_sequence_model"]),
        "overall_rank": int(champion_row["overall_rank"]),
        "composite_rank_score": round(float(champion_row["composite_rank"]), 4),
        "test_metrics": {
            "roc_auc": round(float(champion_row["test_roc_auc"]), 4),
            "pr_auc": round(float(champion_row["test_pr_auc"]), 4),
            "f2_score": round(float(champion_row["test_f2_score"]), 4),
            "f1_score": round(float(champion_row["test_f1"]), 4),
            "recall": round(float(champion_row["test_recall"]), 4),
            "precision": round(float(champion_row["test_precision"]), 4),
            "balanced_accuracy": round(float(champion_row["test_balanced_acc"]), 4),
            "mcc": round(float(champion_row["test_mcc"]), 4),
            "brier_score": round(float(champion_row["test_brier_score"]), 4),
            "brier_skill_score": round(float(champion_row["test_bss"]), 4),
            "ece": round(float(champion_row["test_ece"]), 4),
            "pseudo_r2": round(float(champion_row["test_pseudo_r2"]), 4),
            "lead_time_semesters": round(float(champion_row["test_lead_time"]), 2),
        },
        "efficiency": {
            "inference_time_ms_per_student": round(float(champion_row["inference_time_ms"]), 3),
            "model_size_kb": round(float(champion_row["model_size_kb"]), 2),
            "nfr1_compliant": bool(champion_row["inference_time_ms"] <= max_latency_ms),
        },
        "stability": {
            "seed_std_roc_auc": round(float(champion_row["seed_std"]), 4),
            "stable": bool(champion_row["seed_std"] <= max_seed_std),
        },
        "threshold": round(float(champion_row["threshold"]), 4),
    }

    # ── 10. Save artifacts ────────────────────────────────────────────────
    out_dir = config.leaderboard_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Drop internal rank columns for export
    export_cols = [c for c in df.columns if not c.startswith("rank_")]
    df_export = df[export_cols]

    # JSON
    lb_json_path = out_dir / "leaderboard.json"
    with open(lb_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "data_source": "SYNTHETIC",
            "ranking_weights": weights,
            "eligibility_constraints": constraints,
            "n_models": len(df),
            "n_eligible": int(df["eligible"].sum()),
            "champion": champion_key,
            "models": json.loads(df_export.to_json(orient="records")),
        }, f, indent=2)

    # CSV
    lb_csv_path = out_dir / "leaderboard.csv"
    df_export.to_csv(lb_csv_path, index=False)

    # Markdown
    lb_md_path = out_dir / "leaderboard.md"
    md_cols = [
        "overall_rank", "model_key", "category", "eligible",
        "test_roc_auc", "test_pr_auc", "test_f2_score",
        "test_brier_score", "test_ece", "test_lead_time",
        "composite_rank",
    ]
    available = [c for c in md_cols if c in df_export.columns]
    md_table = df_export[available].to_markdown(index=False)
    md_content = (
        "# Final Test-Set Leaderboard (SYNTHETIC DATA)\n\n"
        f"> **Champion**: {champion_key}\n\n"
        f"{md_table}\n"
    )
    with open(lb_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Champion JSON
    champ_path = out_dir / "champion.json"
    with open(champ_path, "w", encoding="utf-8") as f:
        json.dump(champion_detail, f, indent=2)

    # Research Question Answer
    rq_path = out_dir / "research_question_answer.json"
    with open(rq_path, "w", encoding="utf-8") as f:
        json.dump(research_answer, f, indent=2)

    # Category Champions
    cc_path = out_dir / "category_champions.json"
    with open(cc_path, "w", encoding="utf-8") as f:
        json.dump({"data_source": "SYNTHETIC", "categories": category_champions}, f, indent=2)

    logger.info(f"Leaderboard artifacts saved to: {out_dir}")

    return {
        "leaderboard_df": df,
        "champion": champion_detail,
        "research_question_answer": research_answer,
        "category_champions": category_champions,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Research Question Adjudication
# ──────────────────────────────────────────────────────────────────────────────

def _adjudicate_research_question(
    df: pd.DataFrame,
    seq_key: Optional[str],
    boost_key: Optional[str],
    stat_tests: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Formally answers: "Does sequence-aware modelling (LSTM/GRU) beat
    engineered trend features + Gradient Boosting?"

    Returns a structured JSON-serializable answer.
    """
    if seq_key is None or boost_key is None:
        return {
            "research_question": (
                "Does sequence-aware modelling (LSTM/GRU) beat engineered "
                "trend features + Gradient Boosting for academic attrition prediction?"
            ),
            "answer": "INCONCLUSIVE",
            "reason": "Could not identify both sequence champion and boosting baseline.",
            "data_source": "SYNTHETIC",
        }

    seq_row = df[df["model_key"] == seq_key]
    boost_row = df[df["model_key"] == boost_key]

    if seq_row.empty or boost_row.empty:
        return {
            "research_question": (
                "Does sequence-aware modelling (LSTM/GRU) beat engineered "
                "trend features + Gradient Boosting for academic attrition prediction?"
            ),
            "answer": "INCONCLUSIVE",
            "reason": "One or both models missing from evaluation results.",
            "data_source": "SYNTHETIC",
        }

    s = seq_row.iloc[0]
    b = boost_row.iloc[0]

    delta_roc = float(s["test_roc_auc"] - b["test_roc_auc"])
    delta_pr = float(s["test_pr_auc"] - b["test_pr_auc"])
    delta_f2 = float(s["test_f2_score"] - b["test_f2_score"])
    delta_brier = float(b["test_brier_score"] - s["test_brier_score"])  # lower is better, so positive = seq wins
    delta_lead = float(s["test_lead_time"] - b["test_lead_time"])

    # Statistical significance from paired tests
    paired_data = stat_tests.get("paired_bootstrap_seq_vs_boost", {})
    roc_test = paired_data.get("roc_auc", {})
    pr_test = paired_data.get("pr_auc", {})

    roc_sig = roc_test.get("statistically_significant", False)
    pr_sig = pr_test.get("statistically_significant", False)
    roc_p = roc_test.get("p_value", 1.0)
    pr_p = pr_test.get("p_value", 1.0)

    seq_wins_roc = delta_roc > 0
    seq_wins_pr = delta_pr > 0

    # Decision logic
    if seq_wins_roc and seq_wins_pr and (roc_sig or pr_sig):
        answer = "POSITIVE"
        practical = (
            f"Sequence model ({seq_key}) outperforms boosting baseline ({boost_key}) "
            f"on both ROC-AUC (+{delta_roc:+.4f}) and PR-AUC (+{delta_pr:+.4f}), "
            f"with at least one comparison reaching statistical significance."
        )
    elif not seq_wins_roc and not seq_wins_pr:
        answer = "NULL"
        practical = (
            f"Boosting baseline ({boost_key}) matches or outperforms sequence model ({seq_key}) "
            f"on both ROC-AUC ({delta_roc:+.4f}) and PR-AUC ({delta_pr:+.4f}). "
            f"Engineered trend features appear sufficient for this task."
        )
    elif (seq_wins_roc or seq_wins_pr) and not (roc_sig or pr_sig):
        answer = "NULL"
        practical = (
            f"Sequence model ({seq_key}) shows a directional advantage but the difference "
            f"is NOT statistically significant (ROC-AUC p={roc_p:.4f}, PR-AUC p={pr_p:.4f}). "
            f"Cannot reject the null hypothesis."
        )
    else:
        answer = "MIXED"
        practical = (
            f"Results are inconsistent across metrics. "
            f"ROC-AUC: {'seq wins' if seq_wins_roc else 'boost wins'} ({delta_roc:+.4f}), "
            f"PR-AUC: {'seq wins' if seq_wins_pr else 'boost wins'} ({delta_pr:+.4f})."
        )

    return {
        "research_question": (
            "Does sequence-aware modelling (LSTM/GRU) beat engineered "
            "trend features + Gradient Boosting for academic attrition prediction?"
        ),
        "answer": answer,
        "sequence_champion": {
            "model_key": seq_key,
            "test_roc_auc": round(float(s["test_roc_auc"]), 4),
            "test_pr_auc": round(float(s["test_pr_auc"]), 4),
            "test_f2_score": round(float(s["test_f2_score"]), 4),
            "test_brier": round(float(s["test_brier_score"]), 4),
            "test_lead_time": round(float(s["test_lead_time"]), 2),
        },
        "boosting_baseline": {
            "model_key": boost_key,
            "test_roc_auc": round(float(b["test_roc_auc"]), 4),
            "test_pr_auc": round(float(b["test_pr_auc"]), 4),
            "test_f2_score": round(float(b["test_f2_score"]), 4),
            "test_brier": round(float(b["test_brier_score"]), 4),
            "test_lead_time": round(float(b["test_lead_time"]), 2),
        },
        "deltas": {
            "roc_auc": round(delta_roc, 4),
            "pr_auc": round(delta_pr, 4),
            "f2_score": round(delta_f2, 4),
            "brier_score_improvement": round(delta_brier, 4),
            "lead_time_gain": round(delta_lead, 2),
        },
        "statistical_significance": {
            "roc_auc_p_value": round(roc_p, 5),
            "roc_auc_significant": roc_sig,
            "pr_auc_p_value": round(pr_p, 5),
            "pr_auc_significant": pr_sig,
        },
        "practical_significance": practical,
        "honest_assessment": (
            "IMPORTANT: These results are from SYNTHETIC data generated by a latent-process "
            "simulator. They demonstrate that the evaluation methodology is sound and the "
            "pipeline is end-to-end functional. They do NOT constitute real-world evidence "
            "for or against sequence-aware modelling. Final conclusions require evaluation "
            "on genuine institutional records."
        ),
        "data_source": "SYNTHETIC",
    }
