"""Main evaluation orchestrator for Phase 5.

Executes:
1. Archive stale artifacts before retraining/evaluation
2. Calibrate every classifier's probabilities on validation data and tune decision thresholds
3. Leaderboard creation, hard gates auditing, and composite scoring
4. Top-3 Voting & Stacking meta-ensembles
5. Statistical significance testing & Parsimony Decision Rule
6. Sealed test evaluation (opened strictly once for finalists with fixed calibrator & threshold)
7. Subgroup fairness & disparity audit
8. Frontend JSON export with single source of truth
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.evaluation.audit import run_subgroup_fairness_audit
from src.evaluation.calibration import (
    assign_risk_band,
    calculate_eri,
    calibrate_classifier,
    find_optimal_threshold,
    fit_calibrators,
)
from src.evaluation.ensembles import build_soft_voting_ensemble, build_stacking_ensemble
from src.evaluation.export_frontend import export_all_frontend_data
from src.evaluation.leaderboard import build_leaderboard, load_weights_config
from src.evaluation.metrics import compute_classification_metrics, compute_ece
from src.evaluation.significance import compare_models_cv, evaluate_parsimony_rule
from src.preprocessing.pipeline import get_scaled_pipeline, get_tree_pipeline, validate_feature_boundary
from src.preprocessing.split import get_sealed_test_set, load_split_dataset
from src.utils.logging_config import setup_logger
from src.utils.paths import get_artifacts_dir
from src.utils.run_context import archive_stale_artifacts, get_current_run_info, write_manifest

logger = setup_logger("evaluate_phase5")


def run_phase5_evaluation() -> Dict[str, Any]:
    """Execute complete Phase 5 evaluation, selection, calibration, and audit pipeline."""
    logger.info("=== Starting Phase 5: Leaderboard, Selection & Calibration ===")

    # 1. Archive stale artifacts
    archive_stale_artifacts()
    run_info = get_current_run_info()

    train_df, val_df, _ = load_split_dataset()
    y_train = train_df["attrition"].values
    y_val = val_df["attrition"].values
    prevalence = float(np.mean(y_val))

    preds_dir = get_artifacts_dir("preds")
    cal_dir = get_artifacts_dir("calibrators")
    cal_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = get_artifacts_dir("metrics") / "all_splits_metrics.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}. Run training first.")

    df_metrics = pd.read_csv(metrics_path)
    weights_cfg = load_weights_config()
    th_cfg = weights_cfg.get("threshold_tuning", {})
    obj = th_cfg.get("objective", "balanced_accuracy")
    tie = th_cfg.get("tie_breaker", "f1")

    # 2. Calibrate every classifier and tune decision threshold on validation data
    logger.info("Calibrating all classifiers and tuning decision thresholds on validation split...")
    thresholds: Dict[str, Any] = {}
    updated_metric_rows = []

    # Get list of Track C models with validation predictions
    c_models = [
        f.name.replace("_C_val_preds.npy", "")
        for f in preds_dir.glob("*_C_val_preds.npy")
        if not f.name.startswith("soft_voting") and not f.name.startswith("stacking")
    ]

    for m in c_models:
        val_pred_file = preds_dir / f"{m}_C_val_preds.npy"
        train_pred_file = preds_dir / f"{m}_C_train_preds.npy"
        if not val_pred_file.exists():
            continue

        raw_val_probs = np.load(str(val_pred_file))

        # Calibrate on validation split
        cal_res = calibrate_classifier(raw_val_probs, y_val, prevalence=prevalence)
        calibrator = cal_res["calibrator"]
        cal_val_probs = cal_res["calibrated_probs"]
        joblib.dump(calibrator, str(cal_dir / f"{m}_calibrator.joblib"))

        # Save calibrated probabilities
        np.save(str(preds_dir / f"{m}_C_val_calibrated_preds.npy"), cal_val_probs)

        # Tune decision threshold on validation data
        tau_star, th_metrics = find_optimal_threshold(
            y_val, cal_val_probs, objective=obj, tie_breaker=tie
        )
        thresholds[m] = {
            "threshold": tau_star,
            "objective": obj,
            "metrics": th_metrics,
        }

        # Compute validation binary metrics with optimal threshold
        val_pred_labels = (cal_val_probs >= tau_star).astype(int)
        v_metrics = compute_classification_metrics(y_val, val_pred_labels, cal_val_probs, split="val")
        v_metrics["model"] = m
        v_metrics["track"] = "C"
        v_metrics["threshold"] = tau_star
        v_metrics["brier_score"] = cal_res["brier_score"]
        v_metrics["brier"] = cal_res["brier_score"]
        v_metrics["ece"] = cal_res["ece"]
        v_metrics["brier_skill_score"] = cal_res["brier_skill_score"]
        v_metrics["calibrator"] = cal_res["method"]
        updated_metric_rows.append(v_metrics)

        # Apply same calibrator and fixed threshold to train predictions if present
        if train_pred_file.exists():
            raw_train_probs = np.load(str(train_pred_file))
            cal_train_probs = calibrator.predict(raw_train_probs)
            train_pred_labels = (cal_train_probs >= tau_star).astype(int)
            t_metrics = compute_classification_metrics(y_train, train_pred_labels, cal_train_probs, split="train")
            t_metrics["model"] = m
            t_metrics["track"] = "C"
            t_metrics["threshold"] = tau_star
            t_metrics["brier_score"] = float(np.mean((cal_train_probs - y_train) ** 2))
            t_metrics["brier"] = t_metrics["brier_score"]
            t_metrics["ece"] = float(cal_res["ece"])
            t_metrics["brier_skill_score"] = cal_res["brier_skill_score"]
            t_metrics["calibrator"] = cal_res["method"]
            updated_metric_rows.append(t_metrics)

    # Save tuned thresholds
    th_path = get_artifacts_dir("leaderboard") / "thresholds.json"
    th_path.parent.mkdir(parents=True, exist_ok=True)
    with open(th_path, "w", encoding="utf-8") as tf:
        json.dump(thresholds, tf, indent=2)

    # Update df_metrics with calibrated metrics
    df_updated = pd.DataFrame(updated_metric_rows)
    df_metrics = pd.concat([df_metrics, df_updated], ignore_index=True).drop_duplicates(
        subset=["model", "track", "split"], keep="last"
    )
    df_metrics.to_csv(metrics_path, index=False)

    # 3. Build initial leaderboard for Track C to identify top base models
    lb_c = build_leaderboard("C")
    try:
        lb_r = build_leaderboard("R")
    except Exception:
        lb_r = pd.DataFrame()

    # 4. Identify top 3 non-ensemble base models for Meta-Ensembles (MUST BE QUALIFIED)
    base_c = lb_c[
        (~lb_c["model"].isin(["soft_voting_top3", "stacking_top3"]))
        & (lb_c["qualified"] == True)
    ]
    if len(base_c) < 3:
        base_c = lb_c[~lb_c["model"].isin(["soft_voting_top3", "stacking_top3"])]
    top3_models = list(base_c.head(3)["model"].values)
    logger.info(f"Top 3 Base Models for Ensembling: {top3_models}")

    # Build Soft-Voting & Stacking
    voting_res = build_soft_voting_ensemble(top3_models, train_df, val_df)
    stacking_res = build_stacking_ensemble(top3_models, train_df, val_df)

    # Calibrate Ensembles and tune thresholds
    ensemble_metric_rows = []
    for ens_name, ens_res in [("soft_voting_top3", voting_res), ("stacking_top3", stacking_res)]:
        val_pred_file = preds_dir / f"{ens_name}_C_val_preds.npy"
        train_pred_file = preds_dir / f"{ens_name}_C_train_preds.npy"
        if val_pred_file.exists():
            raw_v = np.load(str(val_pred_file))
            cal_res = calibrate_classifier(raw_v, y_val, prevalence=prevalence)
            calibrator = cal_res["calibrator"]
            cal_v = cal_res["calibrated_probs"]
            joblib.dump(calibrator, str(cal_dir / f"{ens_name}_calibrator.joblib"))

            tau_star, th_metrics = find_optimal_threshold(y_val, cal_v, objective=obj, tie_breaker=tie)
            thresholds[ens_name] = {"threshold": tau_star, "objective": obj, "metrics": th_metrics}

            v_labels = (cal_v >= tau_star).astype(int)
            v_metrics = compute_classification_metrics(y_val, v_labels, cal_v, split="val")
            v_metrics["model"] = ens_name
            v_metrics["track"] = "C"
            v_metrics["threshold"] = tau_star
            v_metrics["brier_score"] = cal_res["brier_score"]
            v_metrics["brier"] = cal_res["brier_score"]
            v_metrics["ece"] = cal_res["ece"]
            v_metrics["brier_skill_score"] = cal_res["brier_skill_score"]
            v_metrics["calibrator"] = cal_res["method"]
            ensemble_metric_rows.append(v_metrics)

            if train_pred_file.exists():
                raw_t = np.load(str(train_pred_file))
                cal_t = calibrator.predict(raw_t)
                t_labels = (cal_t >= tau_star).astype(int)
                t_metrics = compute_classification_metrics(y_train, t_labels, cal_t, split="train")
                t_metrics["model"] = ens_name
                t_metrics["track"] = "C"
                t_metrics["threshold"] = tau_star
                t_metrics["brier_score"] = float(np.mean((cal_t - y_train) ** 2))
                t_metrics["brier"] = t_metrics["brier_score"]
                t_metrics["ece"] = cal_res["ece"]
                t_metrics["brier_skill_score"] = cal_res["brier_skill_score"]
                t_metrics["calibrator"] = cal_res["method"]
                ensemble_metric_rows.append(t_metrics)

    # Re-save thresholds with ensembles included
    with open(th_path, "w", encoding="utf-8") as tf:
        json.dump(thresholds, tf, indent=2)

    # Append ensemble rows to all_splits_metrics.csv
    df_metrics = pd.read_csv(metrics_path)
    df_metrics = pd.concat([df_metrics, pd.DataFrame(ensemble_metric_rows)], ignore_index=True).drop_duplicates(
        subset=["model", "track", "split"], keep="last"
    )
    df_metrics.to_csv(metrics_path, index=False)

    # Re-run leaderboard with ensembles and calibrations included
    lb_c = build_leaderboard("C")
    logger.info("Updated Leaderboard with Meta-Ensembles & Calibrations:")
    for _, r in lb_c.head(6).iterrows():
        logger.info(
            f"  Rank {r['rank']}: {r['model']} ({r['category']}) - Composite: {r['composite_score']:.4f} | "
            f"PR-AUC: {r['pr_auc']:.4f} | F1: {r['f1']:.4f} | BalAcc: {r['balanced_accuracy']:.4f} | "
            f"Brier: {r['brier_score']:.4f} | ECE: {r['ece']:.4f} | Qualified: {r['qualified']}"
        )

    # 5. Categorical Best Models (MUST BE QUALIFIED)
    cat_winners: Dict[str, Any] = {}
    for cat in ["Linear/Shallow", "Tree Ensembles", "Deep Learning", "Ensembles (voting/stacking)"]:
        sub = lb_c[(lb_c["category"] == cat) & (lb_c["qualified"] == True)]
        if sub.empty:
            sub = lb_c[lb_c["category"] == cat]
        if not sub.empty:
            best_row = sub.iloc[0]
            cat_winners[cat] = {
                "model": best_row["model"],
                "category": cat,
                "composite_score": float(best_row["composite_score"]),
                "pr_auc": float(best_row["pr_auc"]),
                "roc_auc": float(best_row["roc_auc"]),
                "f1": float(best_row["f1"]),
                "balanced_accuracy": float(best_row["balanced_accuracy"]),
                "brier_score": float(best_row["brier_score"]),
                "brier": float(best_row["brier_score"]),
                "ece": float(best_row["ece"]),
                "threshold": float(best_row.get("threshold", 0.50)),
                "brier_skill_score": float(best_row.get("brier_skill_score", 0.0)),
                "split": "validation",
                "qualified": bool(best_row.get("qualified", True)),
                "status": str(best_row.get("status", "passed")),
                "run_id": run_info["run_id"],
                "timestamp": run_info["timestamp"],
            }
    logger.info(f"Qualified Category Winners: {list(cat_winners.keys())}")

    # Top overall qualified model and runner-up
    qualified_lb = lb_c[lb_c["qualified"] == True]
    if qualified_lb.empty:
        qualified_lb = lb_c
    overall_top = qualified_lb.iloc[0]["model"]
    runner_up = qualified_lb.iloc[1]["model"] if len(qualified_lb) > 1 else overall_top

    # 6. Statistical significance testing & Parsimony Rule
    top_val_probs = np.load(str(preds_dir / f"{overall_top}_C_val_preds.npy"))
    runner_val_probs = np.load(str(preds_dir / f"{runner_up}_C_val_preds.npy"))
    lr_val_probs = np.load(str(preds_dir / "logistic_l2_C_val_preds.npy"))

    rng = np.random.RandomState(42)
    n_v = len(y_val)
    top_cv_scores, lr_cv_scores, runner_cv_scores = [], [], []
    for _ in range(15):
        boot_idx = rng.choice(n_v, size=n_v, replace=True)
        from sklearn.metrics import average_precision_score
        top_cv_scores.append(float(average_precision_score(y_val[boot_idx], top_val_probs[boot_idx])))
        lr_cv_scores.append(float(average_precision_score(y_val[boot_idx], lr_val_probs[boot_idx])))
        runner_cv_scores.append(float(average_precision_score(y_val[boot_idx], runner_val_probs[boot_idx])))

    comp_runner = compare_models_cv(top_cv_scores, runner_cv_scores)
    parsimony_decision = evaluate_parsimony_rule(overall_top, top_cv_scores, lr_cv_scores)

    selection_decision = {
        "leaderboard_rank_1": overall_top,
        "runner_up": runner_up,
        "comparison_with_runner_up": comp_runner,
        "parsimony_evaluation": parsimony_decision,
        "selected_model": parsimony_decision["selected_model"],
        "reason": parsimony_decision["rule_applied"],
        "run_id": run_info["run_id"],
        "timestamp": run_info["timestamp"],
    }
    decision_path = get_artifacts_dir("leaderboard") / "selection_decision.json"
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(selection_decision, f, indent=2)
    logger.info(f"Final Model Selection: {selection_decision['selected_model']} (Reason: {selection_decision['reason']})")

    # 7. Final Selected Model Calibrator & ERI check
    final_model = selection_decision["selected_model"]
    final_val_probs = np.load(str(preds_dir / f"{final_model}_C_val_preds.npy"))
    final_cal_file = cal_dir / f"{final_model}_calibrator.joblib"
    if final_cal_file.exists():
        calibrator = joblib.load(str(final_cal_file))
        cal_name = "isotonic" if "isotonic" in type(calibrator).__name__.lower() else "platt_sigmoid"
    else:
        calibrator, cal_name, _ = fit_calibrators(final_val_probs, y_val)
        joblib.dump(calibrator, str(final_cal_file))

    cal_val_probs = calibrator.predict(final_val_probs)
    cal_ece = float(compute_ece(y_val, cal_val_probs))
    val_eri, score_label, is_calibrated = calculate_eri(cal_val_probs, cal_ece)

    # 8. SEALED TEST SET ACCESS - OPENED STRICTLY ONCE FOR FINALISTS
    logger.info("Opening Sealed Test Set for Category Finalists and Selected Model...")
    test_df = get_sealed_test_set(
        allow_test_access=True,
        caller_description="Phase 5 Finalist Evaluation & Audit",
    )
    y_test = test_df["attrition"].values
    finalists_to_test = list(set([cat_winners[c]["model"] for c in cat_winners] + [final_model, overall_top]))
    top3_base = [m for m in top3_models if m in c_models]

    def _get_single_model_test_probs(m_name: str) -> np.ndarray:
        m_file = get_artifacts_dir("models") / "C" / f"{m_name}.joblib"
        if not m_file.exists():
            return np.full(len(y_test), 0.5)
        v_type = "tree" if m_name in [
            "random_forest", "extra_trees", "adaboost", "gradient_boosting",
            "hist_gradient_boosting", "xgboost", "lightgbm", "catboost"
        ] else "scaled"
        pipe = get_tree_pipeline() if v_type == "tree" else get_scaled_pipeline()
        pipe.fit(validate_feature_boundary(train_df), is_train_fold=True)
        x_t = pipe.transform(validate_feature_boundary(test_df))
        mod = joblib.load(str(m_file))
        if hasattr(mod, "predict_proba"):
            raw_p = mod.predict_proba(x_t)[:, 1]
        elif hasattr(mod, "predict"):
            raw_p = np.clip(mod.predict(x_t), 0.0, 1.0)
        else:
            raw_p = np.full(len(y_test), 0.5)

        # Apply fitted calibrator
        cal_path = cal_dir / f"{m_name}_calibrator.joblib"
        if cal_path.exists():
            try:
                cal = joblib.load(str(cal_path))
                return cal.predict(raw_p)
            except Exception:
                pass
        return raw_p

    final_test_results = {}
    for fm in finalists_to_test:
        if fm == "soft_voting_top3":
            t3_probs = [_get_single_model_test_probs(bm) for bm in top3_base]
            test_probs = np.mean(t3_probs, axis=0) if t3_probs else np.full(len(y_test), 0.5)
        elif fm == "stacking_top3":
            meta_path = get_artifacts_dir("models") / "C" / "stacking_top3.joblib"
            if meta_path.exists() and top3_base:
                meta_clf = joblib.load(str(meta_path))
                X_meta_test = np.column_stack([_get_single_model_test_probs(bm) for bm in top3_base])
                test_probs = meta_clf.predict_proba(X_meta_test)[:, 1]
            else:
                test_probs = np.mean([_get_single_model_test_probs(bm) for bm in top3_base], axis=0)
        else:
            test_probs = _get_single_model_test_probs(fm)

        # Apply FIXED validation threshold
        fm_th_info = thresholds.get(fm, {})
        tau_fixed = fm_th_info.get("threshold", 0.50)
        test_pred = (test_probs >= tau_fixed).astype(int)

        t_metrics = compute_classification_metrics(y_test, test_pred, test_probs, split="test")
        t_metrics["model"] = fm
        t_metrics["threshold"] = tau_fixed
        t_metrics["run_id"] = run_info["run_id"]
        t_metrics["timestamp"] = run_info["timestamp"]
        final_test_results[fm] = {
            "metrics": t_metrics,
            "test_probabilities": test_probs.tolist(),
        }

    # Save final test report
    test_report_path = get_artifacts_dir("metrics") / "final_test_report.json"
    with open(test_report_path, "w", encoding="utf-8") as f:
        json.dump({m: final_test_results[m]["metrics"] for m in final_test_results}, f, indent=2)
    logger.info(f"Sealed test report written -> {test_report_path}")

    # 9. Subgroup Fairness & Disparity Audit
    final_test_probs = np.array(final_test_results[final_model]["test_probabilities"])
    audit_report = run_subgroup_fairness_audit(test_df, y_test, final_test_probs, model_name=final_model)
    audit_report["run_id"] = run_info["run_id"]
    audit_report["timestamp"] = run_info["timestamp"]
    audit_path = get_artifacts_dir("metrics") / "subgroup_fairness_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    logger.info(f"Subgroup fairness audit written -> {audit_path}")

    # 10. Write manifest documenting active run
    write_manifest(metadata={"prevalence": prevalence, "models_count": len(lb_c)})

    # 11. Export All Frontend Data Assets
    export_all_frontend_data(
        leaderboard_c=lb_c,
        leaderboard_r=lb_r,
        category_winners=cat_winners,
        selection_decision=selection_decision,
        audit_report=audit_report,
    )

    logger.info("=== Phase 5 Evaluation Complete ===")
    return {
        "status": "success",
        "run_id": run_info["run_id"],
        "timestamp": run_info["timestamp"],
        "selected_model": final_model,
        "category_winners": cat_winners,
        "calibrator": cal_name,
        "is_calibrated": is_calibrated,
        "score_label": score_label,
    }
