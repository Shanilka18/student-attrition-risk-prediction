"""
Phase 7 Master Explainability & Risk Index Orchestrator.

Coordinates:
  1. Academic Risk Index (0-100) computation and clinical risk banding
  2. SHAP TreeExplainer on Champion CatBoost model (global + local attributions)
  3. Temporal Integrated Gradients & Attention on Sequence model
  4. Rules-based deterministic plain-language diagnostic narrative generation
  5. Faithfulness feature deletion test and stability check
  6. Fusion Framework contract validation and batch export (Pydantic)
  7. High-resolution publication explainability plots

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from backend.src.config import config
from backend.src.models.base import ModelRegistry
import backend.src.models  # Trigger model registrations
from backend.src.explain.risk_index import AcademicRiskIndexEngine, DEFAULT_RISK_BANDS
from backend.src.explain.shap_explainer import TabularTreeSHAPExplainer, FEATURE_METADATA
from backend.src.explain.sequence_explainer import SequenceModelExplainer
from backend.src.explain.sentence_generator import generate_explanation_narrative
from backend.src.explain.faithfulness import FaithfulnessAuditor
from backend.src.explain.fusion_contract import (
    FactorContribution,
    FusionAcademicRiskPayload,
    FusionBatchExport,
)
from backend.src.explain.plots import (
    plot_global_shap_importance,
    plot_temporal_attribution_heatmap,
    plot_faithfulness_deletion_curve,
    plot_risk_band_distribution,
    plot_local_waterfall_sample,
)

logger = logging.getLogger("AcademicIntelligence.ExplainOrchestrator")


def run_explainability_pipeline(
    champion_key: str = "catboost",
    sequence_key: str = "tcn",
    top_k_factors: int = 3,
) -> Dict[str, Any]:
    """
    Executes the complete Phase 7 explainability and risk index pipeline.
    """
    t0 = time.perf_counter()
    logger.info("=" * 72)
    logger.info("PHASE 7: EXPLAINABILITY & FINAL ACADEMIC RISK INDEX PIPELINE")
    logger.info("DATA SOURCE: SYNTHETIC — Demonstrates interpretability framework")
    logger.info("=" * 72)

    out_dir = config.explanations_dir
    plots_dir = config.plots_dir / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Partitions and Models ─────────────────────────────────────
    logger.info("Step 1: Loading test partition, champion model, and calibrator...")
    with open(config.splits_dir / "test_students.json", "r", encoding="utf-8") as f:
        test_sids = json.load(f)

    # Load Tabular Data
    df_tab = pd.read_csv(config.processed_data_dir / "tabular_features.csv")
    meta_cols = ["sample_id", "student_id", "window_t", "target"]
    feature_cols = [c for c in df_tab.columns if c not in meta_cols]
    df_test_tab = df_tab[df_tab["student_id"].isin(test_sids)].reset_index(drop=True)
    X_test_tab = df_test_tab[feature_cols].values.astype(np.float32)
    y_test = df_test_tab["target"].values.astype(np.int64)

    # Load Champion Model (CatBoost)
    catboost_dir = config.models_dir / "Category_C" / champion_key
    model_champ = ModelRegistry.load_model(champion_key, catboost_dir)
    calibrator = joblib.load(catboost_dir / "calibrator.joblib")

    # ── 2. Compute Academic Risk Index & Clinical Bands ───────────────────
    logger.info("Step 2: Computing Academic Risk Index and clinical risk tiers...")
    raw_probs = model_champ.predict_proba(X_test_tab)
    cal_probs = calibrator.transform(raw_probs)

    risk_engine = AcademicRiskIndexEngine(DEFAULT_RISK_BANDS)
    ari_scores = risk_engine.compute_risk_index(cal_probs)
    risk_bands = risk_engine.batch_assign_bands(ari_scores)
    band_summary = risk_engine.summarize_cohort(ari_scores)

    logger.info(
        f"ARI Cohort Summary: Mean={band_summary['mean_ari']:.1f}, "
        f"Median={band_summary['median_ari']:.1f} | Bands: {band_summary['band_percentages']}"
    )

    # ── 3. SHAP TreeExplainer on Tabular Champion ─────────────────────────
    logger.info("Step 3: Generating SHAP values for global and local attribution...")
    shap_explainer = TabularTreeSHAPExplainer(model_champ, feature_cols)
    shap_values = shap_explainer.explain(X_test_tab)
    df_global_imp = shap_explainer.get_global_importance(X_test_tab)

    df_global_imp.to_csv(out_dir / "global_feature_importance.csv", index=False)
    logger.info(f"Top 3 Global Features: {list(df_global_imp['display_name'].head(3))}")

    # ── 4. Sequence Model Temporal Explanation ────────────────────────────
    logger.info("Step 4: Computing temporal attribution matrix on sequence architecture...")
    seq_matrix_df = pd.DataFrame()
    try:
        seq_payload = torch.load(
            config.processed_data_dir / "sequence_tensors.pt",
            map_location="cpu", weights_only=False,
        )
        student_ids_seq = np.array(seq_payload["student_ids"])
        test_mask = np.isin(student_ids_seq, test_sids)
        idx = torch.tensor(np.where(test_mask)[0], dtype=torch.long)

        seq_test_dict = {
            "tensors": seq_payload["tensors"][idx],
            "padding_mask": seq_payload["padding_mask"][idx],
            "seq_lengths": seq_payload["seq_lengths"][idx],
        }

        # Load Sequence Model (TCN or LSTM Attention)
        seq_dir = config.models_dir / "Category_E" / sequence_key
        if not seq_dir.exists():
            seq_dir = config.models_dir / "Category_E" / "lstm_attention"
            sequence_key = "lstm_attention"

        model_seq = ModelRegistry.load_model(sequence_key, seq_dir)
        seq_explainer = SequenceModelExplainer(model_seq, device=config.device)
        seq_matrix_df = seq_explainer.get_semester_by_feature_matrix(seq_test_dict, steps=15)
        seq_matrix_df.to_csv(out_dir / "semester_feature_attribution_matrix.csv")
        logger.info("Extracted Semester x Feature attribution matrix successfully.")
    except Exception as e:
        logger.warning(f"Sequence temporal attribution skipped or encountered error: {e}", exc_info=True)

    # ── 5. Faithfulness Deletion Test & Stability Verification ────────────
    logger.info("Step 5: Conducting Faithfulness Deletion and Stability tests...")
    faithfulness_auditor = FaithfulnessAuditor(model_champ, calibrator, feature_cols)
    deletion_results = faithfulness_auditor.run_deletion_test(
        X_test_tab, shap_values, k_steps=[0, 1, 2, 3, 5, 8], min_risk_threshold=0.05
    )
    stability_results = faithfulness_auditor.run_stability_check(
        X_test_tab, df_global_imp, n_bootstrap=4, noise_std=0.02
    )

    with open(out_dir / "faithfulness_deletion_results.json", "w", encoding="utf-8") as f:
        json.dump(deletion_results, f, indent=2)

    with open(out_dir / "explanation_stability_results.json", "w", encoding="utf-8") as f:
        json.dump(stability_results, f, indent=2)

    logger.info(f"Faithfulness: {deletion_results['conclusion']}")
    logger.info(f"Stability: {stability_results['conclusion']}")

    # ── 6. Per-Student Narrative & Fusion Contract Generation ─────────────
    logger.info("Step 6: Generating validated Fusion Framework payloads for all test students...")
    fusion_payloads: List[FusionAcademicRiskPayload] = []

    for i in range(len(df_test_tab)):
        row_sid = str(df_test_tab.iloc[i]["student_id"])
        row_sample_id = str(df_test_tab.iloc[i]["sample_id"])
        row_win = int(df_test_tab.iloc[i]["window_t"])
        row_ari = float(ari_scores[i])
        row_band = risk_bands[i].name

        # Extract local factors
        local_exp = shap_explainer.get_local_explanation(i, top_k=top_k_factors)
        raw_feat_dict = {col: float(X_test_tab[i, c_idx]) for c_idx, col in enumerate(feature_cols)}

        # Build factor contribution objects
        factor_objs: List[FactorContribution] = []
        for factor in local_exp["top_factors"]:
            factor_objs.append(
                FactorContribution(
                    name=factor["name"],
                    display_name=factor["display_name"],
                    direction=factor["direction"],
                    contribution=factor["contribution"],
                    feature_value=factor["feature_value"],
                    unit=factor["unit"],
                    narrative=factor["display_name"],
                )
            )

        # Generate deterministic plain-language diagnostic narrative
        narrative_text = generate_explanation_narrative(
            academic_risk_index=row_ari,
            risk_band=row_band,
            top_factors=local_exp["top_factors"],
            raw_features=raw_feat_dict,
        )

        payload = FusionAcademicRiskPayload(
            participant_id=row_sid,
            sample_id=row_sample_id,
            window_semester=row_win,
            academic_risk_index=row_ari,
            risk_band=row_band,
            top_factors=factor_objs,
            model_name=champion_key,
            model_version="1.0.0",
            calibrated=True,
            explanation_text=narrative_text,
            provenance_tag="SYNTHETIC",
        )
        fusion_payloads.append(payload)

    # Save Fusion JSON exports
    fusion_json_path = out_dir / "fusion_payloads.json"
    sample_json_path = out_dir / "sample_fusion_contract.json"
    FusionBatchExport.save_batch(fusion_payloads, fusion_json_path, sample_json_path)

    # Save CSV table of all test student explanations
    summary_rows = []
    for p in fusion_payloads:
        f_names = [f.display_name for f in p.top_factors]
        summary_rows.append({
            "participant_id": p.participant_id,
            "sample_id": p.sample_id,
            "window_semester": p.window_semester,
            "academic_risk_index": p.academic_risk_index,
            "risk_band": p.risk_band,
            "primary_driver": f_names[0] if len(f_names) > 0 else "",
            "secondary_driver": f_names[1] if len(f_names) > 1 else "",
            "tertiary_driver": f_names[2] if len(f_names) > 2 else "",
            "explanation_text": p.explanation_text,
        })
    df_explanations = pd.DataFrame(summary_rows)
    df_explanations.to_csv(out_dir / "per_student_explanations.csv", index=False)

    # ── 7. Generate Explainability Plots ───────────────────────────────────
    logger.info("Step 7: Rendering publication-grade explainability plots...")
    plot_files = []

    # 1. Global SHAP Importance
    p1 = plot_global_shap_importance(df_global_imp, plots_dir / "shap_summary_global.png")
    plot_files.append(p1.name)

    # 2. Temporal Heatmap
    if not seq_matrix_df.empty:
        p2 = plot_temporal_attribution_heatmap(seq_matrix_df, plots_dir / "temporal_attention_heatmap.png")
        plot_files.append(p2.name)

    # 3. Faithfulness Deletion Curve
    p3 = plot_faithfulness_deletion_curve(
        deletion_results["curve"], plots_dir / "feature_deletion_faithfulness.png"
    )
    plot_files.append(p3.name)

    # 4. Risk Band Distribution
    p4 = plot_risk_band_distribution(band_summary, plots_dir / "risk_band_distribution.png")
    plot_files.append(p4.name)

    # 5. Local Waterfall Sample (find highest risk student in test set)
    high_risk_idx = int(np.argmax(ari_scores))
    high_student_id = str(df_test_tab.iloc[high_risk_idx]["student_id"])
    high_ari = float(ari_scores[high_risk_idx])
    high_local = shap_explainer.get_local_explanation(high_risk_idx, top_k=6)
    p5 = plot_local_waterfall_sample(
        high_local["all_factors"], high_student_id, high_ari, plots_dir / "waterfall_sample_high_risk.png"
    )
    plot_files.append(p5.name)

    logger.info(f"Generated {len(plot_files)} explainability plots in {plots_dir}")

    elapsed = time.perf_counter() - t0
    logger.info(f"Phase 7 Explainability Pipeline completed in {elapsed:.1f}s.")

    return {
        "status": "COMPLETED",
        "n_students_explained": len(fusion_payloads),
        "champion_model": champion_key,
        "band_summary": band_summary,
        "faithfulness": deletion_results,
        "stability": stability_results,
        "fusion_json": str(fusion_json_path),
        "sample_fusion_json": str(sample_json_path),
        "plots": plot_files,
        "runtime_seconds": round(elapsed, 2),
    }
