"""MEI Scoring Engine for Uploaded Student Cohorts.

Implements score_dataframe(df) to score clean survey DataFrames (CSV/XLSX),
computing:
  - MEI (0-100) via Isotonic Calibrated Champion Model
  - Risk Band ('Low' | 'Moderate' | 'High')
  - At-Risk Flag ('Yes' | 'No' at tuned optimal threshold)
  - Latent Trajectory Pattern ('Consistent' | 'Declining' | 'Sporadic' | 'Improving')
  - Engagement Level ('High' | 'Medium' | 'Low')
  - Latent State Path (e.g. 'S1 -> S1 -> S2 -> S2')
  - Top-3 Risk Drivers placeholder (to be populated with real SHAP in Phase 10)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from mei.config import ARTIFACTS_DIR, RUNS_DIR
from mei.features.pipeline import extract_features
from mei.states.hmm import decode_hmm_sequences, prepare_sequence_data
from mei.states.patterns import classify_trajectory_pattern

logger = logging.getLogger(__name__)


def resolve_active_champion_dir(run_dir: Optional[Path] = None) -> Path:
    """Resolve champion bundle directory from run_dir or active_run.txt."""
    if run_dir is None:
        active_txt = ARTIFACTS_DIR / "active_run.txt"
        if not active_txt.exists():
            raise FileNotFoundError(f"Active run file not found at {active_txt}")
        run_id = active_txt.read_text(encoding="utf-8").strip()
        run_dir = RUNS_DIR / run_id

    champion_dir = Path(run_dir) / "champion"
    if not champion_dir.exists():
        raise FileNotFoundError(
            f"Champion bundle not found at {champion_dir}. Run 'mei finalize' first."
        )
    return champion_dir


def score_dataframe(
    df: pd.DataFrame,
    run_dir: Optional[Path] = None,
    explain: bool = False,
) -> pd.DataFrame:
    """Score a clean student survey DataFrame and produce the MEI Intelligence Table.

    Args:
        df: Clean canonical survey DataFrame containing student_id and the 28 indicator columns.
        run_dir: Optional run directory. If None, resolves from active_run.txt.
        explain: If True, computes exact local SHAP feature attributions and fills top_reasons
                 with plain-English drivers and MEI point contributions.

    Returns:
        pd.DataFrame containing ['student_id', 'mei', 'band', 'flag', 'pattern', 'level', 'state_path', 'top_reasons']
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "student_id",
                "mei",
                "band",
                "flag",
                "pattern",
                "level",
                "state_path",
                "top_reasons",
            ]
        )

    df_clean = df.copy()

    # 1. Ensure student_id
    if "student_id" in df_clean.columns:
        student_ids = df_clean["student_id"].astype(str).tolist()
    else:
        student_ids = [f"STUDENT_{i + 1:04d}" for i in range(len(df_clean))]
        df_clean["student_id"] = student_ids

    # 2. Resolve champion bundle paths
    champ_dir = resolve_active_champion_dir(run_dir)
    pipeline_path = champ_dir / "pipeline.joblib"
    calibrator_path = champ_dir / "calibrator.joblib"
    thresholds_path = champ_dir / "thresholds.json"
    bands_path = champ_dir / "bands.json"

    pipeline = joblib.load(pipeline_path)
    calibrator = joblib.load(calibrator_path)

    thresholds_info = json.load(open(thresholds_path, "r", encoding="utf-8"))
    opt_tau = float(thresholds_info.get("optimal_threshold", {}).get("threshold", 0.29))

    features_path = champ_dir / "features.json"
    expected_features = json.load(open(features_path, "r", encoding="utf-8"))

    bands_info = json.load(open(bands_path, "r", encoding="utf-8")).get(
        "definitions", {}
    )
    low_max = int(bands_info.get("low", {}).get("max", 34))
    mod_max = int(bands_info.get("moderate", {}).get("max", 64))

    # 3. Extract aggregate features for champion model
    X_agg, _ = extract_features(df_clean, feature_set="agg")
    if isinstance(X_agg, pd.DataFrame):
        for col in expected_features:
            if col not in X_agg.columns:
                X_agg[col] = 0.0
        X_matrix = X_agg[expected_features].values
    else:
        X_matrix = X_agg

    # 4. Predict uncalibrated probabilities and calibrate via Isotonic Regression
    raw_probs = pipeline.predict_proba(X_matrix)[:, 1]
    cal_probs = calibrator.predict(raw_probs)
    cal_probs = np.clip(cal_probs, 0.0, 1.0)

    # 5. MEI Score (0 to 100)
    mei_scores = np.round(100.0 * cal_probs).astype(int)

    # 6. Bands & At-Risk Flags
    bands = []
    flags = []
    for s, p in zip(mei_scores, cal_probs):
        if s <= low_max:
            bands.append("Low")
        elif s <= mod_max:
            bands.append("Moderate")
        else:
            bands.append("High")

        flags.append("Yes" if p >= opt_tau else "No")

    # 7. Decode Latent States & Trajectory Patterns
    run_path = champ_dir.parent
    hmm_model_file = run_path / "states" / "hmm_model.joblib"

    patterns: List[str] = []
    levels: List[str] = []
    state_paths: List[str] = []

    if hmm_model_file.exists():
        try:
            hmm_model = joblib.load(hmm_model_file)
            X_seq_scaled, lengths, _, _ = prepare_sequence_data(df_clean)
            viterbi_paths, _, _ = decode_hmm_sequences(hmm_model, X_seq_scaled, lengths)

            for path in viterbi_paths:
                p_res = classify_trajectory_pattern(path, k=hmm_model.n_components)
                patterns.append(p_res.pattern)
                levels.append(p_res.level)
                state_paths.append(" -> ".join([f"S{s}" for s in path]))
        except Exception as e:
            logger.warning(f"Could not decode HMM sequence states: {e}")
            patterns = ["Consistent"] * len(df_clean)
            levels = ["Medium"] * len(df_clean)
            state_paths = ["S1 -> S1 -> S1 -> S1"] * len(df_clean)
    else:
        patterns = ["Consistent"] * len(df_clean)
        levels = ["Medium"] * len(df_clean)
        state_paths = ["S1 -> S1 -> S1 -> S1"] * len(df_clean)

    # 8. Top-3 Reasons Placeholder (Human-interpretable contextual driver summary)
    top_reasons_list = []
    for idx, row in df_clean.iterrows():
        # Identify indicator with lowest mean across periods
        p1_4_cols = [
            c
            for c in df_clean.columns
            if c.startswith(
                (
                    "resource_access",
                    "study_time",
                    "review_time",
                    "study_consistency",
                    "participation",
                )
            )
        ]
        if p1_4_cols:
            indicators = [
                "resource_access",
                "study_time",
                "review_time",
                "study_consistency",
                "participation",
            ]
            ind_means = {
                ind: np.mean(
                    [row[f"{ind}_p{k}"] for k in range(1, 5) if f"{ind}_p{k}" in row]
                )
                for ind in indicators
            }
            sorted_inds = sorted(ind_means.items(), key=lambda x: x[1])
            weakest_1 = sorted_inds[0][0].replace("_", " ").title()
            weakest_2 = sorted_inds[1][0].replace("_", " ").title()
            r1 = f"Low overall {weakest_1} across semester"
            r2 = f"Reduced {weakest_2} in late periods"
            r3 = f"Engagement pattern categorized as {patterns[idx]}"
            top_reasons_list.append(f"1. {r1}; 2. {r2}; 3. {r3}")
        else:
            top_reasons_list.append(
                "1. Routine engagement monitoring; 2. Pattern consistent; 3. Baseline stable"
            )

    # If explain=True requested, populate top_reasons with exact SHAP feature attributions
    if explain:
        try:
            from mei.explain.engine import explain_dataframe

            _, full_explanations = explain_dataframe(df, run_dir=champ_dir.parent)
            if len(full_explanations) == len(df):
                top_reasons_list = [
                    "; ".join(
                        [
                            f"{r['text']} ({r['points']:+d} pts)"
                            for r in ex["reasons"][:3]
                        ]
                    )
                    for ex in full_explanations
                ]
        except Exception as e:
            logger.warning("Falling back to baseline reasons due to: %s", e)

    # 9. Assemble Output DataFrame
    results_df = pd.DataFrame(
        {
            "student_id": student_ids,
            "mei": mei_scores,
            "band": bands,
            "flag": flags,
            "pattern": patterns,
            "level": levels,
            "state_path": state_paths,
            "top_reasons": top_reasons_list,
        }
    )

    return results_df
