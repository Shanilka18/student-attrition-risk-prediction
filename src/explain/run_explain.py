"""CLI Pipeline Runner for Phase 6 Explainability Engine.

Executes:
- Automated SHAP explainer selection and caching.
- Global feature attribution ranking & factor-level aggregation.
- Global diagnostic visualizations (Bar, Beeswarm, Dependence).
- Local per-student explanations, waterfall data JSONs, and plain-language narratives.
- Scientific sanity checks (Additivity, Stability, Permutation Agreement, Ground-Truth Recovery).
- Outputs formatted global importance tables to stdout.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

from src.data.splits import load_split
from src.features.build_features import compute_factor_scores, extract_feature_set, extract_targets
from src.explain.explainer import build_explainer, explain_dataset, save_shap_artifacts
from src.explain.global_explain import (
    compute_global_importance,
    aggregate_factor_importance,
    plot_global_summary_bar,
    plot_beeswarm,
    plot_dependence,
)
from src.explain.local_explain import explain_student_instance, export_student_waterfall_json
from src.explain.narrative import generate_student_narrative
from src.explain.sanity_checks import run_all_sanity_checks
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("explain.run_explain")

FINAL_MODEL_DIR = Path("artifacts/models/final")
SHAP_DIR = Path("artifacts/shap")


def run_explainability_pipeline(
    track: str = "both",
    n_students: int = 5,
    seed: int = 42,
    output_dir: Path = SHAP_DIR,
) -> Dict[str, Any]:
    """Execute end-to-end explainability suite across champion models."""
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = ["regression", "classification"] if track == "both" else [track]
    val_df = load_split("val", allow_test=False)
    val_feat = compute_factor_scores(val_df)
    feature_set = "factor_scores"
    X_val = extract_feature_set(val_feat, feature_set)
    targets = extract_targets(val_feat)

    all_summaries: Dict[str, Any] = {}

    for trk in tracks:
        logger.info("=" * 65)
        logger.info("EXPLAINABILITY ENGINE — TRACK: %s", trk.upper())
        logger.info("=" * 65)

        # 1. Load champion model
        champ_path = FINAL_MODEL_DIR / f"{trk}_champion.joblib"
        if not champ_path.exists():
            # Fallback to artifacts/models/{track}
            champ_path = Path(f"artifacts/models/{trk}") / ("ridge.joblib" if trk == "regression" else "hist_gradient_boosting.joblib")
        
        if not champ_path.exists():
            logger.error("Champion model not found at %s. Please train Phase 4/5 models first.", champ_path)
            continue

        model = joblib.load(champ_path)
        model_name = "ridge" if trk == "regression" else "hist_gradient_boosting"
        logger.info("Loaded champion model '%s' from %s", model_name, champ_path)

        # 2. Build automated SHAP explainer
        explainer, explainer_type = build_explainer(model, X_val, seed=seed)

        # Determine target class index for classification (High Risk)
        target_c_idx = 0
        if hasattr(model, "classes_"):
            classes_list = list(model.classes_)
            if "High" in classes_list:
                target_c_idx = classes_list.index("High")

        # 3. Compute SHAP values
        logger.info("Computing SHAP attributions for %d validation students...", len(X_val))
        explanation = explain_dataset(explainer, X_val, track=trk, target_class_idx=target_c_idx)
        shap_values = explanation["values"]
        base_value = explanation["base_value"]
        feature_names = explanation["feature_names"]

        # 4. Cache artifacts
        save_shap_artifacts(explanation, trk, model_name, output_dir=output_dir)

        # 5. Global importance & aggregation
        global_df = compute_global_importance(shap_values, feature_names)
        factor_df = aggregate_factor_importance(global_df)

        global_csv = output_dir / f"global_importance_{trk}.csv"
        factor_csv = output_dir / f"factor_importance_{trk}.csv"
        global_df.to_csv(global_csv, index=False)
        factor_df.to_csv(factor_csv, index=False)

        # 6. Global Diagnostic Plots
        logger.info("Generating global SHAP visualizations...")
        plot_global_summary_bar(global_df, trk, model_name, output_dir=output_dir)
        try:
            plot_beeswarm(shap_values, X_val, feature_names, trk, model_name, output_dir=output_dir)
        except Exception as e:
            logger.warning("Beeswarm plot generation skipped: %s", e)

        # Top feature dependence plot
        top_feature = global_df.iloc[0]["feature"]
        try:
            plot_dependence(shap_values, X_val, feature_names, top_feature, trk, model_name, output_dir=output_dir)
        except Exception as e:
            logger.warning("Dependence plot generation skipped: %s", e)

        # 7. Local Per-Student Explanations & Plain-Language Narratives
        logger.info("Generating local explanations for sample cohort of %d students...", n_students)
        local_students = []
        for i in range(min(n_students, len(X_val))):
            s_id = val_df.iloc[i].get("study_id", f"STUDENT_{i:04d}")
            pri_val = targets["pri_score"].iloc[i] if "pri_score" in targets else 50.0
            risk_cat = targets["risk_category"].iloc[i] if "risk_category" in targets else "Moderate"

            local_exp = explain_student_instance(
                student_idx=i,
                shap_values=shap_values,
                X=X_val,
                feature_names=feature_names,
                base_value=base_value,
                study_id=s_id,
            )

            # Export local waterfall JSON
            waterfall_file = output_dir / f"student_{s_id}_waterfall_{trk}.json"
            export_student_waterfall_json(local_exp, waterfall_file)

            # Plain-language narrative
            narrative = generate_student_narrative(
                study_id=s_id,
                pri_score=float(pri_val),
                risk_category=str(risk_cat),
                top_risk_increasing=local_exp["top_risk_increasing"],
                top_risk_decreasing=local_exp["top_risk_decreasing"],
            )

            narrative_file = output_dir / f"student_{s_id}_narrative_{trk}.json"
            with open(narrative_file, "w", encoding="utf-8") as nf:
                json.dump(narrative, nf, indent=2)

            local_students.append({
                "study_id": s_id,
                "pri_score": pri_val,
                "risk_category": risk_cat,
                "narrative": narrative,
                "waterfall": local_exp,
            })

        # 8. Scientific Sanity Checks
        y_target = targets["pri_score"].values if trk == "regression" else targets["risk_category"].values

        def explainer_subsample_fn(X_sub):
            return explain_dataset(explainer, X_sub, track=trk, target_class_idx=target_c_idx)

        logger.info("Executing scientific sanity checks for %s...", trk)
        sanity_report = run_all_sanity_checks(
            model=model,
            X=X_val,
            y=y_target,
            shap_values=shap_values,
            base_value=base_value,
            feature_names=feature_names,
            empirical_factor_df=factor_df,
            explainer_func=explainer_subsample_fn,
            output_dir=output_dir,
        )

        all_summaries[trk] = {
            "model_name": model_name,
            "explainer_type": explainer_type,
            "global_importance": global_df,
            "factor_importance": factor_df,
            "local_samples": local_students,
            "sanity_checks": sanity_report,
        }

    return all_summaries


def main() -> None:
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(description="Phase 6: Explainability and SHAP Attribution Engine")
    parser.add_argument("--track", choices=["regression", "classification", "both"], default="both")
    parser.add_argument("--n-students", type=int, default=5, help="Number of student profiles to explain locally")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="artifacts/shap", help="Output directory")

    args = parser.parse_args()
    results = run_explainability_pipeline(
        track=args.track,
        n_students=args.n_students,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )

    print("\n" + "=" * 70)
    print("  PHASE 6: EXPLAINABILITY AND ATTRIBUTION ENGINE REPORT")
    print("=" * 70)

    for trk, data in results.items():
        print(f"\n[GLOBAL ATTRIBUTION TABLE - {trk.upper()} TRACK (Model: {data['model_name']})]")
        print("-" * 70)
        print(data["factor_importance"].to_string(index=False))
        print("-" * 70)

        # Sanity Checks
        sc = data["sanity_checks"]
        additivity = sc["additivity_check"]
        stability = sc["stability_check"]
        perm = sc["permutation_agreement_check"]
        gt = sc["ground_truth_recovery_check"]

        print(f"\n[SANITY CHECKS AUDIT - {trk.upper()}]")
        print(f"  - Additivity Axiom Check  : {additivity['status']} (Mean Error: {additivity['mean_absolute_error']})")
        print(f"  - Subsample Stability     : {stability['status']} (Spearman rho: {stability['spearman_rank_correlation']})")
        print(f"  - Permutation Agreement   : {perm['status']} (Spearman rho: {perm['spearman_rank_correlation']})")
        print(f"  - Ground Truth Recovery   : {gt['status']} (Spearman rho: {gt['spearman_rank_correlation']})")

        # Sample narrative
        if data["local_samples"]:
            sample = data["local_samples"][0]
            print(f"\n[SAMPLE STUDENT EXPLANATION NARRATIVE ({sample['study_id']})]")
            print(f"  Headline : {sample['narrative']['headline']}")
            print(f"  Drivers  : {sample['narrative']['risk_increasing_narrative']}")
            print(f"  Buffers  : {sample['narrative']['protective_narrative']}")
            print(f"  Advice   : {sample['narrative']['recommendations']}")
            print(f"  Notice   : {sample['narrative']['disclaimer']}")

    print("\nAll explanation artifacts, beeswarm plots, and waterfall contracts saved in artifacts/shap/.\n")


if __name__ == "__main__":
    main()
