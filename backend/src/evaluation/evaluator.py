"""
Phase 6 Final Test-Set Evaluation Orchestrator.

Evaluates every trained model EXACTLY ONCE on the held-out test partition,
records cryptographic provenance, runs bootstrap confidence intervals and
paired hypothesis tests, and generates publication-quality plots.

NON-NEGOTIABLE: The test set is touched once. Provenance hash is logged.
All outputs are labelled SYNTHETIC until real data replaces them.

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from backend.src.config import config
from backend.src.models.base import ModelRegistry, CATEGORIES
from backend.src.training.calibration import ProbabilityCalibrator
from backend.src.evaluation.metrics import compute_comprehensive_metrics
from backend.src.evaluation.statistical_tests import (
    compute_grouped_bootstrap_cis,
    paired_bootstrap_test,
    mcnemar_test,
)
from backend.src.evaluation.provenance import record_test_evaluation_provenance
from backend.src.evaluation.plots import (
    plot_roc_curves_overlay,
    plot_pr_curves_overlay,
    plot_calibration_curves,
    plot_confusion_matrices,
    plot_metric_heatmap,
    plot_lead_time_distribution,
    plot_risk_index_distribution,
    plot_efficiency_vs_performance,
)

logger = logging.getLogger("AcademicIntelligence.Evaluator")

# ── Category → data representation routing ────────────────────────────────────
_SEQUENCE_CATEGORIES = {"E"}
_STATIC_CATEGORIES = {"F"}


# ──────────────────────────────────────────────────────────────────────────────
#  Data Loading Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_student_ids(split_name: str) -> List[str]:
    path = config.splits_dir / f"{split_name}_students.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_tabular_test(test_sids: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Returns (X_test, y_test, df_test_meta) for tabular models."""
    df = pd.read_csv(config.processed_data_dir / "tabular_features.csv")
    meta_cols = ["sample_id", "student_id", "window_t", "target"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df_test = df[df["student_id"].isin(test_sids)].reset_index(drop=True)
    X = df_test[feature_cols].values.astype(np.float32)
    y = df_test["target"].values.astype(np.int64)
    return X, y, df_test


def _load_static_test(test_sids: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Returns (X_test, y_test, df_test_meta) for static-snapshot models."""
    df = pd.read_csv(config.processed_data_dir / "static_features.csv")
    feature_cols = ["last_gpa", "last_attendance_pct"]
    df_test = df[df["student_id"].isin(test_sids)].reset_index(drop=True)
    X = df_test[feature_cols].values.astype(np.float32)
    y = df_test["target"].values.astype(np.int64)
    return X, y, df_test


def _load_sequence_test(test_sids: List[str]) -> Tuple[Dict[str, Any], np.ndarray, pd.DataFrame]:
    """Returns (X_dict, y_test, df_test_meta) for sequence models."""
    payload = torch.load(
        config.processed_data_dir / "sequence_tensors.pt",
        map_location="cpu", weights_only=False,
    )
    student_ids = np.array(payload["student_ids"])
    test_mask = np.isin(student_ids, test_sids)
    idx = torch.tensor(np.where(test_mask)[0], dtype=torch.long)

    X_dict = {
        "tensors": payload["tensors"][idx],
        "padding_mask": payload["padding_mask"][idx],
        "seq_lengths": payload["seq_lengths"][idx],
    }
    y = payload["targets"][idx].numpy().astype(np.int64)

    # Build a meta DataFrame aligned to the sequence samples
    df_meta = pd.DataFrame({
        "sample_id": np.array(payload["sample_ids"])[test_mask.nonzero()[0]],
        "student_id": student_ids[test_mask],
    })
    # Merge window_t and last_gpa from tabular (needed for early-warning metrics)
    df_tab = pd.read_csv(config.processed_data_dir / "tabular_features.csv")
    df_tab_sub = df_tab[df_tab["student_id"].isin(test_sids)][
        ["sample_id", "window_t", "target", "last_gpa"]
    ]
    df_meta = df_meta.merge(df_tab_sub, on="sample_id", how="left")
    return X_dict, y, df_meta


def _dir_size_kb(directory: Path) -> float:
    total = 0
    for f in directory.iterdir():
        if f.is_file():
            total += f.stat().st_size
    return total / 1024.0


# ──────────────────────────────────────────────────────────────────────────────
#  Single-Model Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_single_model(
    model_key: str,
    model_dir: Path,
    summary: Dict[str, Any],
    X_test: Any,
    y_test: np.ndarray,
    df_test_samples: pd.DataFrame,
    df_outcomes: pd.DataFrame,
    n_test_students: int,
) -> Dict[str, Any]:
    """
    Loads one trained model, calibrator, and threshold, predicts on test,
    computes comprehensive metrics.  Returns the full result dict.
    """
    category = summary["category"]
    label = summary.get("label", model_key)

    # 1. Load model
    model = ModelRegistry.load_model(model_key, model_dir)

    # 2. Load calibrator
    cal_path = model_dir / "calibrator.joblib"
    calibrator = ProbabilityCalibrator.load(cal_path)

    # 3. Load optimal threshold
    thresh_path = model_dir / "threshold_config.json"
    thresh_cfg = {}
    if thresh_path.exists():
        with open(thresh_path, "r", encoding="utf-8") as f:
            thresh_cfg = json.load(f)
    threshold = (
        thresh_cfg.get("selected_threshold")
        or thresh_cfg.get("optimal_threshold")
        or thresh_cfg.get("calibrated_threshold_optimization", {}).get("optimal_threshold")
        or summary.get("threshold", {}).get("optimal_threshold")
        or 0.5
    )
    threshold = float(threshold)

    # 4. Inference timing
    t0 = time.perf_counter()
    raw_probs = model.predict_proba(X_test)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    per_student_ms = total_ms / max(n_test_students, 1)

    # 5. Calibrate
    cal_probs = calibrator.transform(raw_probs)

    # 6. Model disk size
    size_kb = _dir_size_kb(model_dir)

    # 7. Comprehensive metrics
    metrics = compute_comprehensive_metrics(
        y_true=y_test,
        y_prob=cal_probs,
        threshold=threshold,
        df_samples=df_test_samples,
        df_outcomes=df_outcomes,
        inference_time_ms=per_student_ms,
        model_size_kb=size_kb,
    )

    # 8. Load stability info from Phase 5 summary
    stab = summary.get("stability_summary", {})
    seed_std = stab.get("roc_auc_std", 0.0)

    return {
        "model_key": model_key,
        "category": category,
        "label": label,
        "is_proposal_model": summary.get("is_proposal_model", False),
        "is_sequence_model": summary.get("is_sequence_model", False),
        "metrics": metrics,
        "calibrated_probs": cal_probs,
        "threshold": threshold,
        "seed_std": seed_std,
        "raw_probs": raw_probs,
        "validation_roc_auc": summary.get("validation_metrics", {}).get("roc_auc", 0.0),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Plot Generation
# ──────────────────────────────────────────────────────────────────────────────

def _generate_all_plots(
    all_results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    champion_key: str,
    plots_dir: Path,
) -> List[str]:
    """Generates the 8 publication-grade evaluation plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    # Select representative models for overlay plots
    # Best per category + proposal models
    cat_bests: Dict[str, str] = {}
    for key, res in all_results.items():
        cat = res["category"]
        auc = res["metrics"]["classification"]["roc_auc"]
        if cat not in cat_bests or auc > all_results[cat_bests[cat]]["metrics"]["classification"]["roc_auc"]:
            cat_bests[cat] = key

    overlay_keys = set(cat_bests.values())
    overlay_keys.add(champion_key)
    # Also add proposal models
    for key, res in all_results.items():
        if res.get("is_proposal_model", False):
            overlay_keys.add(key)
    overlay_keys = list(overlay_keys)[:8]  # cap at 8 for readability

    # 1. ROC Curves
    try:
        roc_data = {}
        for key in overlay_keys:
            res = all_results[key]
            roc_data[res["label"]] = (
                res["calibrated_probs"],
                {"roc_auc": res["metrics"]["classification"]["roc_auc"]},
            )
        plot_roc_curves_overlay(roc_data, y_test, plots_dir / "roc_curves_overlay.png")
        generated.append("roc_curves_overlay.png")
    except Exception as e:
        logger.warning(f"Failed to generate ROC overlay: {e}")

    # 2. PR Curves
    try:
        pr_data = {}
        for key in overlay_keys:
            res = all_results[key]
            pr_data[res["label"]] = (
                res["calibrated_probs"],
                {"pr_auc": res["metrics"]["classification"]["pr_auc"]},
            )
        plot_pr_curves_overlay(pr_data, y_test, plots_dir / "pr_curves_overlay.png")
        generated.append("pr_curves_overlay.png")
    except Exception as e:
        logger.warning(f"Failed to generate PR overlay: {e}")

    # 3. Calibration Curves
    try:
        cal_data = {}
        for key in overlay_keys:
            res = all_results[key]
            curve = res["metrics"]["risk_index_quality"].get("calibration_curve", [])
            if curve:
                cal_data[res["label"]] = curve
        plot_calibration_curves(cal_data, plots_dir / "calibration_curves.png")
        generated.append("calibration_curves.png")
    except Exception as e:
        logger.warning(f"Failed to generate calibration curves: {e}")

    # 4. Confusion Matrices (champion + key baselines)
    try:
        cm_keys = [champion_key]
        if cat_bests.get("C") and cat_bests["C"] != champion_key:
            cm_keys.append(cat_bests["C"])
        if cat_bests.get("F") and cat_bests["F"] != champion_key:
            cm_keys.append(cat_bests["F"])
        if cat_bests.get("A") and cat_bests["A"] != champion_key:
            cm_keys.append(cat_bests["A"])
        cm_dict = {}
        for key in cm_keys[:4]:
            res = all_results[key]
            cm_dict[res["label"]] = res["metrics"]["classification"]["confusion_matrix"]
        plot_confusion_matrices(cm_dict, plots_dir / "confusion_matrices.png")
        generated.append("confusion_matrices.png")
    except Exception as e:
        logger.warning(f"Failed to generate confusion matrices: {e}")

    # 5. Metric Heatmap
    try:
        rows = []
        for key, res in all_results.items():
            clf = res["metrics"]["classification"]
            rows.append({
                "Model Key": key,
                "Val ROC-AUC": res.get("validation_roc_auc", 0.0),
                "PR-AUC": clf["pr_auc"],
                "F1 Score": clf["f1_score"],
                "Recall": clf["recall"],
                "Brier (Cal)": res["metrics"]["risk_index_quality"]["brier_score"],
                "Threshold": res["threshold"],
            })
        heatmap_df = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)
        plot_metric_heatmap(heatmap_df, plots_dir / "metric_heatmap.png")
        generated.append("metric_heatmap.png")
    except Exception as e:
        logger.warning(f"Failed to generate metric heatmap: {e}")

    # 6. Lead-Time Distribution (champion)
    try:
        champ_ew = all_results[champion_key]["metrics"].get("early_warning", {})
        lead_times = champ_ew.get("lead_time_distribution", [])
        plot_lead_time_distribution(
            lead_times, plots_dir / "lead_time_distribution.png",
            champion_name=all_results[champion_key]["label"],
        )
        generated.append("lead_time_distribution.png")
    except Exception as e:
        logger.warning(f"Failed to generate lead time plot: {e}")

    # 7. Risk Index Distribution (champion)
    try:
        champ_probs = all_results[champion_key]["calibrated_probs"]
        ari = champ_probs * 100.0
        ari_pos = ari[y_test == 1]
        ari_neg = ari[y_test == 0]
        cutoff_ari = all_results[champion_key]["threshold"] * 100.0
        plot_risk_index_distribution(
            ari_pos, ari_neg, plots_dir / "risk_index_distribution.png",
            threshold_cutoff=cutoff_ari,
        )
        generated.append("risk_index_distribution.png")
    except Exception as e:
        logger.warning(f"Failed to generate risk index distribution: {e}")

    # 8. Efficiency Frontier
    try:
        eff_data = []
        for key, res in all_results.items():
            eff_data.append({
                "model_key": key,
                "category": res["category"],
                "inference_time_ms_per_student": res["metrics"]["efficiency"]["inference_time_ms_per_student"],
                "roc_auc": res["metrics"]["classification"]["roc_auc"],
            })
        plot_efficiency_vs_performance(eff_data, plots_dir / "metric_vs_complexity.png")
        generated.append("metric_vs_complexity.png")
    except Exception as e:
        logger.warning(f"Failed to generate efficiency frontier: {e}")

    logger.info(f"Generated {len(generated)}/{8} evaluation plots in {plots_dir}")
    return generated


# ──────────────────────────────────────────────────────────────────────────────
#  Main Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_full_evaluation() -> Dict[str, Any]:
    """
    Master entry-point for Phase 6.
    Evaluates ALL trained models ONCE on the frozen test partition.

    Returns a dict with:
      - models: per-model evaluation results
      - statistical_tests: bootstrap CIs and paired hypothesis tests
      - provenance: SHA-256 audit record
      - plots: list of generated plot files
    """
    logger.info("=" * 72)
    logger.info("PHASE 6: ONE-TOUCH FINAL TEST SET EVALUATION")
    logger.info("DATA SOURCE: SYNTHETIC — Results demonstrate methodology only")
    logger.info("=" * 72)

    # ── 0. Register all model classes ─────────────────────────────────────
    import backend.src.models  # Registers all models across Categories A-F (tabular, mlp, sequence)

    # ── 1. Load test partition ────────────────────────────────────────────
    test_sids = _load_student_ids("test")
    n_test_students = len(test_sids)
    logger.info(f"Test partition: {n_test_students} students loaded")

    # ── 2. Load all three data representations ────────────────────────────
    X_tab, y_tab, df_tab_test = _load_tabular_test(test_sids)
    X_static, y_static, df_static_test = _load_static_test(test_sids)
    X_seq, y_seq, df_seq_test = _load_sequence_test(test_sids)

    logger.info(
        f"Test data loaded: Tabular {X_tab.shape}, Static {X_static.shape}, "
        f"Sequence tensors {X_seq['tensors'].shape}"
    )

    # ── 3. Load outcomes (for early-warning metrics) ──────────────────────
    df_outcomes = pd.read_csv(config.root_dir / "data" / "synthetic" / "outcomes.csv")

    # ── 4. Record provenance ──────────────────────────────────────────────
    provenance_path = config.metrics_dir / "test_evaluation_provenance.json"
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    provenance = record_test_evaluation_provenance(
        output_path=provenance_path,
        df_test_samples=df_tab_test,
        test_student_ids=test_sids,
    )
    logger.info(
        f"Provenance recorded: SHA-256={provenance['current_fingerprint'][:16]}... | "
        f"Touch #{provenance['total_test_evaluations']}"
    )

    # ── 5. Discover trained models ────────────────────────────────────────
    model_dirs: List[Tuple[str, Path, Dict]] = []
    for cat_dir in sorted(config.models_dir.iterdir()):
        if not cat_dir.is_dir() or not cat_dir.name.startswith("Category_"):
            continue
        for m_dir in sorted(cat_dir.iterdir()):
            if not m_dir.is_dir():
                continue
            summary_path = m_dir / "summary.json"
            if not summary_path.exists():
                continue
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            if summary.get("status") != "COMPLETED":
                logger.warning(f"Skipping {m_dir.name}: status={summary.get('status')}")
                continue
            model_dirs.append((summary["model_key"], m_dir, summary))

    logger.info(f"Discovered {len(model_dirs)} completed models for test evaluation")

    # ── 6. Evaluate each model ────────────────────────────────────────────
    all_results: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, str]] = []

    for idx, (m_key, m_dir, summary) in enumerate(model_dirs, 1):
        cat = summary["category"]
        logger.info(f"[{idx}/{len(model_dirs)}] Evaluating {m_key} (Category {cat})...")

        try:
            # Route to correct data representation
            if cat in _SEQUENCE_CATEGORIES:
                X_test, y_test, df_meta = X_seq, y_seq, df_seq_test
            elif cat in _STATIC_CATEGORIES:
                X_test, y_test, df_meta = X_static, y_static, df_static_test
            else:
                X_test, y_test, df_meta = X_tab, y_tab, df_tab_test

            result = _evaluate_single_model(
                model_key=m_key,
                model_dir=m_dir,
                summary=summary,
                X_test=X_test,
                y_test=y_test,
                df_test_samples=df_meta,
                df_outcomes=df_outcomes,
                n_test_students=n_test_students,
            )
            all_results[m_key] = result

            auc = result["metrics"]["classification"]["roc_auc"]
            pr = result["metrics"]["classification"]["pr_auc"]
            f2 = result["metrics"]["classification"]["f2_score"]
            logger.info(f"  ✓ {m_key}: ROC-AUC={auc:.4f}  PR-AUC={pr:.4f}  F2={f2:.4f}")

        except Exception as e:
            logger.error(f"  ✗ FAILED to evaluate {m_key}: {e}", exc_info=True)
            failures.append({"model_key": m_key, "error": str(e)})

    logger.info(
        f"Evaluation complete: {len(all_results)} succeeded, {len(failures)} failed"
    )

    # ── 7. Identify key models for statistical tests ──────────────────────
    def _best_in_cat(cat: str) -> Optional[str]:
        best_key, best_auc = None, -1.0
        for k, r in all_results.items():
            if r["category"] == cat:
                auc = r["metrics"]["classification"]["roc_auc"]
                if auc > best_auc:
                    best_auc = auc
                    best_key = k
        return best_key

    seq_champion_key = _best_in_cat("E")
    boost_baseline_key = _best_in_cat("C")
    static_best_key = _best_in_cat("F")
    tab_best_key = _best_in_cat("C")  # boosting is also best tabular approach

    # Overall best (for champion candidate)
    overall_best_key = max(
        all_results.keys(),
        key=lambda k: all_results[k]["metrics"]["classification"]["roc_auc"],
    )

    logger.info(
        f"Key models — Seq Champion: {seq_champion_key}, "
        f"Boost Baseline: {boost_baseline_key}, Overall Best: {overall_best_key}"
    )

    # ── 8. Bootstrap CIs ─────────────────────────────────────────────────
    lb_cfg = config._raw.get("leaderboard", {})
    boot_cfg = lb_cfg.get("bootstrap", {})
    n_resamples = boot_cfg.get("n_resamples", 1000)
    boot_seed = boot_cfg.get("seed", 42)

    bootstrap_results: Dict[str, Any] = {}

    for bkey in [seq_champion_key, boost_baseline_key, overall_best_key]:
        if bkey is None or bkey not in all_results:
            continue
        res = all_results[bkey]
        cat = res["category"]
        if cat in _SEQUENCE_CATEGORIES:
            _, y_ref, df_ref = y_seq, y_seq, df_seq_test
        elif cat in _STATIC_CATEGORIES:
            _, y_ref, df_ref = y_static, y_static, df_static_test
        else:
            _, y_ref, df_ref = y_tab, y_tab, df_tab_test

        try:
            cis = compute_grouped_bootstrap_cis(
                df_samples=df_ref,
                y_true=y_ref,
                y_prob=res["calibrated_probs"],
                threshold=res["threshold"],
                n_resamples=n_resamples,
                seed=boot_seed,
            )
            bootstrap_results[bkey] = cis
            logger.info(
                f"  Bootstrap CIs for {bkey}: "
                f"ROC-AUC [{cis['roc_auc']['ci_lower']:.4f}, {cis['roc_auc']['ci_upper']:.4f}]"
            )
        except Exception as e:
            logger.warning(f"  Bootstrap CIs failed for {bkey}: {e}")

    # ── 9. Paired Hypothesis Tests ────────────────────────────────────────
    stat_tests: Dict[str, Any] = {"bootstrap_cis": bootstrap_results}

    # 9a. Sequence vs Boosting (THE core research question)
    if seq_champion_key and boost_baseline_key and seq_champion_key in all_results and boost_baseline_key in all_results:
        # Both are evaluated on the tabular/sequence y_test which is aligned by sample
        # We need them on the SAME samples — use tabular partition as common ground
        # Sequence models and tabular models evaluate the same samples (same students),
        # aligned by sample index in their respective test arrays.
        # The y values are identical across representations for the same students.
        try:
            seq_res = all_results[seq_champion_key]
            boost_res = all_results[boost_baseline_key]

            # Paired bootstrap on ROC-AUC
            paired_roc = paired_bootstrap_test(
                df_samples=df_tab_test,
                y_true=y_tab,
                y_prob_a=seq_res["calibrated_probs"],
                y_prob_b=boost_res["calibrated_probs"],
                metric_fn=roc_auc_score,
                n_resamples=n_resamples,
                seed=boot_seed,
            )
            # Paired bootstrap on PR-AUC
            paired_pr = paired_bootstrap_test(
                df_samples=df_tab_test,
                y_true=y_tab,
                y_prob_a=seq_res["calibrated_probs"],
                y_prob_b=boost_res["calibrated_probs"],
                metric_fn=average_precision_score,
                n_resamples=n_resamples,
                seed=boot_seed,
            )
            # McNemar on binary predictions
            y_pred_seq = (seq_res["calibrated_probs"] >= seq_res["threshold"]).astype(int)
            y_pred_boost = (boost_res["calibrated_probs"] >= boost_res["threshold"]).astype(int)
            mcnemar_result = mcnemar_test(y_tab, y_pred_seq, y_pred_boost)

            stat_tests["paired_bootstrap_seq_vs_boost"] = {
                "model_a": seq_champion_key,
                "model_b": boost_baseline_key,
                "roc_auc": paired_roc,
                "pr_auc": paired_pr,
                "mcnemar": mcnemar_result,
            }
            logger.info(
                f"  Paired test (Seq vs Boost): ΔROC-AUC={paired_roc['observed_difference']:+.4f} "
                f"p={paired_roc['p_value']:.4f} | "
                f"ΔPR-AUC={paired_pr['observed_difference']:+.4f} p={paired_pr['p_value']:.4f}"
            )
        except Exception as e:
            logger.warning(f"  Paired test seq vs boost failed: {e}", exc_info=True)

    # 9b. Best tabular (with trend features) vs Static snapshot
    if tab_best_key and static_best_key and tab_best_key in all_results and static_best_key in all_results:
        try:
            tab_res = all_results[tab_best_key]
            static_res = all_results[static_best_key]

            paired_tab_static = paired_bootstrap_test(
                df_samples=df_tab_test,
                y_true=y_tab,
                y_prob_a=tab_res["calibrated_probs"],
                y_prob_b=static_res["calibrated_probs"],
                metric_fn=roc_auc_score,
                n_resamples=n_resamples,
                seed=boot_seed,
            )
            stat_tests["paired_bootstrap_tabular_vs_static"] = {
                "model_a": tab_best_key,
                "model_b": static_best_key,
                "roc_auc": paired_tab_static,
                "interpretation": (
                    "Quantifies the value of engineered trend features over "
                    "naive static snapshot (last GPA + attendance only)."
                ),
            }
            logger.info(
                f"  Paired test (Tabular vs Static): ΔROC-AUC={paired_tab_static['observed_difference']:+.4f} "
                f"p={paired_tab_static['p_value']:.4f}"
            )
        except Exception as e:
            logger.warning(f"  Paired test tabular vs static failed: {e}", exc_info=True)

    # ── 10. Save detailed results ─────────────────────────────────────────
    config.metrics_dir.mkdir(parents=True, exist_ok=True)

    # Serializable version (strip numpy arrays)
    serializable_results: Dict[str, Any] = {}
    for k, r in all_results.items():
        entry = {
            "model_key": r["model_key"],
            "category": r["category"],
            "label": r["label"],
            "is_proposal_model": r["is_proposal_model"],
            "is_sequence_model": r["is_sequence_model"],
            "threshold": r["threshold"],
            "seed_std": r["seed_std"],
            "validation_roc_auc": r["validation_roc_auc"],
            "metrics": r["metrics"],
        }
        serializable_results[k] = entry

    test_results_payload = {
        "data_source": "SYNTHETIC",
        "warning": "All results are from SYNTHETIC data and demonstrate methodology only.",
        "n_test_students": n_test_students,
        "n_test_samples": int(len(y_tab)),
        "provenance_fingerprint": provenance["current_fingerprint"],
        "models": serializable_results,
        "statistical_tests": _make_serializable(stat_tests),
        "failures": failures,
    }

    json_path = config.metrics_dir / "test_evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(test_results_payload, f, indent=2, default=str)
    logger.info(f"Saved test evaluation results to: {json_path}")

    # CSV summary
    csv_rows = []
    for k, r in all_results.items():
        clf = r["metrics"]["classification"]
        riq = r["metrics"]["risk_index_quality"]
        ew = r["metrics"].get("early_warning", {})
        eff = r["metrics"]["efficiency"]
        csv_rows.append({
            "model_key": k,
            "category": r["category"],
            "label": r["label"],
            "is_proposal": r["is_proposal_model"],
            "test_roc_auc": clf["roc_auc"],
            "test_pr_auc": clf["pr_auc"],
            "test_f1": clf["f1_score"],
            "test_f2": clf["f2_score"],
            "test_recall": clf["recall"],
            "test_precision": clf["precision"],
            "test_balanced_acc": clf["balanced_accuracy"],
            "test_mcc": clf["matthews_corrcoef"],
            "test_brier": riq["brier_score"],
            "test_bss": riq["brier_skill_score"],
            "test_ece": riq["expected_calibration_error"],
            "test_pseudo_r2": riq["pseudo_r2"],
            "test_lead_time": ew.get("average_lead_time_semesters", 0.0),
            "test_recall_top10": ew.get("recall_at_top_10_pct", 0.0),
            "test_detected_before_gpa2": ew.get("detected_before_gpa_below_2_rate", 0.0),
            "inference_ms": eff["inference_time_ms_per_student"],
            "model_size_kb": eff["model_size_kb"],
            "seed_std": r["seed_std"],
            "threshold": r["threshold"],
        })
    df_csv = pd.DataFrame(csv_rows).sort_values("test_roc_auc", ascending=False)
    csv_path = config.metrics_dir / "test_evaluation_summary.csv"
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"Saved test evaluation CSV to: {csv_path}")

    # ── 11. Generate plots ────────────────────────────────────────────────
    eval_plots_dir = config.plots_dir / "evaluation"
    plot_files = _generate_all_plots(all_results, y_tab, overall_best_key, eval_plots_dir)

    # ── 12. Return full result ────────────────────────────────────────────
    return {
        "models": all_results,
        "serializable_models": serializable_results,
        "statistical_tests": stat_tests,
        "provenance": provenance,
        "plots": plot_files,
        "failures": failures,
        "key_models": {
            "sequence_champion": seq_champion_key,
            "boosting_baseline": boost_baseline_key,
            "static_best": static_best_key,
            "overall_best": overall_best_key,
        },
        "n_test_students": n_test_students,
        "n_test_samples": int(len(y_tab)),
    }


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
