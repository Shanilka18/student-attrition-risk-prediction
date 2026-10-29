"""Production Inference Engine for the Environmental Risk Intelligence (ERI) Framework.

Executes scoring on raw or survey datasets:
1. Ingests and aligns input schema.
2. Applies pre-fitted preprocessor, model, and probability calibrator.
3. Computes exact linear SHAP attributions and deterministic plain-language narratives.
4. Categorizes into standardized risk bands (Low/Moderate/High/Critical).
5. Generates per-student data quality notes when inputs are sparse.
6. Computes cohort-level summary statistics and conditional causal summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.evaluation.calibration import assign_risk_band, calculate_eri
from src.explain.narrative import generate_student_narrative
from src.inference.ingestion import align_dataframe_schema, load_raw_file
from src.preprocessing.codec import EnvFinCodec
from src.preprocessing.pipeline import validate_feature_boundary
from src.utils.logging_config import setup_logger

logger = setup_logger("inference_predict")


class ProductionInferenceEngine:
    """Production Predictor using saved preprocessor, winning classifier, and probability calibrator."""

    def __init__(
        self,
        model_name: str | None = None,
        artifacts_dir: Path | None = None,
    ):
        self.artifacts_dir = artifacts_dir or Path("artifacts")

        # 1. Resolve winning model name
        if model_name is None:
            decision_path = (
                self.artifacts_dir / "leaderboard" / "selection_decision.json"
            )
            if decision_path.exists():
                with open(decision_path, "r", encoding="utf-8") as f:
                    dec = json.load(f)
                    self.model_name = dec.get("selected_model", "logistic_l2")
            else:
                self.model_name = "logistic_l2"
        else:
            self.model_name = model_name

        logger.info(
            f"Initializing Production Inference Engine using model: '{self.model_name}'"
        )

        # 2. Load pre-fitted preprocessor
        prep_path = self.artifacts_dir / "scalers" / "production_preprocessor.joblib"
        if not prep_path.exists():
            raise FileNotFoundError(f"Fitted preprocessor not found at: {prep_path}")
        self.preprocessor = joblib.load(str(prep_path))
        model_inputs = getattr(self.preprocessor, "feature_names_in_", None)
        codec = EnvFinCodec()
        supported_inputs = set(codec.items) | set(codec.derived_indices)
        if model_inputs is None or not set(model_inputs).issubset(supported_inputs):
            raise RuntimeError(
                "The saved model feature schema is incompatible with the "
                "Environmental & Financial questionnaire. Retrain and publish "
                "a model using the current questionnaire schema before scoring."
            )

        # 3. Load trained model
        model_path = self.artifacts_dir / "models" / "C" / f"{self.model_name}.joblib"
        if not model_path.exists():
            candidates = [
                "logistic_l2",
                "catboost",
                "random_forest",
                "linear_svm",
                "xgboost",
                "lightgbm",
            ]
            for cand in candidates:
                cand_path = self.artifacts_dir / "models" / "C" / f"{cand}.joblib"
                if cand_path.exists():
                    logger.warning(
                        f"Trained model '{self.model_name}' has no direct single joblib artifact. "
                        f"Falling back to champion individual model '{cand}'."
                    )
                    self.model_name = cand
                    model_path = cand_path
                    break
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at: {model_path}")
        self.model = joblib.load(str(model_path))

        # 4. Load fitted probability calibrator
        cal_path = (
            self.artifacts_dir / "calibrators" / f"{self.model_name}_calibrator.joblib"
        )
        if cal_path.exists():
            self.calibrator = joblib.load(str(cal_path))
            self.has_calibrator = True
        else:
            logger.warning(
                f"No dedicated calibrator found at {cal_path}. Falling back to raw predict_proba."
            )
            self.calibrator = None
            self.has_calibrator = False

        # 5. Extract model linear weights for fast exact SHAP computation
        clf = getattr(self.model, "_clf", self.model)
        if hasattr(clf, "coef_"):
            self.coef = clf.coef_[0]
            self.intercept = (
                float(clf.intercept_[0]) if hasattr(clf, "intercept_") else 0.0
            )
            self.is_linear = True
        else:
            self.is_linear = False

        # Feature names post-preprocessing
        try:
            raw_names = self.preprocessor.get_feature_names_out()
            self.feature_names = [n.split("__")[-1] for n in raw_names]
        except (AttributeError, NotImplementedError, ValueError):
            self.feature_names = (
                [f"feat_{i}" for i in range(len(self.coef))] if self.is_linear else []
            )

    def score_dataframe(
        self, df_raw: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Score a raw or questionnaire dataframe and return student scores and cohort summary."""
        # Align schema and purge PII
        df_aligned, schema_report = align_dataframe_schema(df_raw)
        n_samples = len(df_aligned)
        participant_ids = df_aligned["participant_id"].values

        # Enforce feature set boundary
        X_boundary = validate_feature_boundary(df_aligned)

        # Preprocess features
        X_trans = self.preprocessor.transform(X_boundary)

        # Predict raw probabilities
        raw_probs = self.model.predict_proba(X_trans)[:, 1]

        # Calibrate probabilities -> ERI score
        if self.has_calibrator:
            calibrated_probs = self.calibrator.predict(raw_probs)
            calibrated_probs = np.clip(calibrated_probs, 0.0, 1.0)
        else:
            calibrated_probs = raw_probs

        eri_scores, _score_label, is_calibrated = calculate_eri(
            calibrated_probs, ece=0.04
        )

        # Calculate SHAP attributions per student
        # For linear models: phi_i = w_i * (x_i - mean(X_train))
        # Preprocessor standard scaler mean is available:
        if self.is_linear:
            # Reconstruct marginal contributions in log-odds space
            shap_matrix = X_trans * self.coef
        elif hasattr(self.model, "feature_importances_"):
            shap_matrix = X_trans * self.model.feature_importances_
        else:
            shap_matrix = np.zeros_like(X_trans)

        # Vectorize missing field counts for high-throughput batch scoring
        missing_counts = df_aligned.isna().sum(axis=1).to_numpy()

        student_records = []
        for i in range(n_samples):
            pid = str(participant_ids[i])
            eri_val = float(round(eri_scores[i], 1))
            band_info = assign_risk_band(eri_val)
            band_name = band_info["band"]

            # Check missing input count for data quality confidence note
            row_missing_count = int(missing_counts[i])
            if row_missing_count >= 5:
                confidence_note = f"LOW CONFIDENCE: Student record has {row_missing_count} missing environmental fields; scores reflect median-imputed baselines."
            elif row_missing_count >= 2:
                confidence_note = (
                    f"MODERATE CONFIDENCE: {row_missing_count} missing fields imputed."
                )
            else:
                confidence_note = "HIGH CONFIDENCE: Complete survey response."

            # Generate plain-language narrative
            shap_vec = shap_matrix[i]
            narrative_res = generate_student_narrative(
                participant_id=pid,
                eri_score=eri_val,
                feature_names=self.feature_names,
                shap_vector=shap_vec,
            )

            student_records.append(
                {
                    "participant_id": pid,
                    "eri_score": eri_val,
                    "risk_band": band_name,
                    "confidence_note": confidence_note,
                    "missing_fields_count": row_missing_count,
                    "narrative": narrative_res["narrative"],
                    "top_positive_drivers": narrative_res["main_drivers"],
                    "top_protective_factors": narrative_res["protective_factors"],
                }
            )

        df_scored = pd.DataFrame(student_records)

        # Cohort summary metrics
        risk_dist = df_scored["risk_band"].value_counts(normalize=True).to_dict()
        risk_dist_pct = {k: round(float(v) * 100.0, 1) for k, v in risk_dist.items()}

        mean_abs_shap = (
            np.mean(np.abs(shap_matrix), axis=0)
            if self.is_linear
            else np.zeros(len(self.feature_names))
        )
        top_cohort_driver_indices = np.argsort(mean_abs_shap)[::-1][:5]
        cohort_drivers = [
            {
                "feature": self.feature_names[idx],
                "mean_abs_shap": round(float(mean_abs_shap[idx]), 4),
            }
            for idx in top_cohort_driver_indices
        ]

        cohort_summary = {
            "model_version": f"{self.model_name}_v1.0",
            "is_calibrated_eri": is_calibrated,
            "total_students_scored": n_samples,
            "mean_eri": round(float(np.mean(eri_scores)), 1),
            "median_eri": round(float(np.median(eri_scores)), 1),
            "risk_band_distribution_pct": risk_dist_pct,
            "top_cohort_drivers": cohort_drivers,
            "schema_report": schema_report,
        }

        # Conditional causal summary
        if schema_report["has_outcome_column"] and n_samples >= 300:
            cohort_summary["causal_summary"] = (
                "Outcome column detected with N >= 300. Causal inference can be evaluated via `python -m src.causal.run_phase7`."
            )
        else:
            cohort_summary["causal_summary"] = (
                "Causal summary unavailable: Requires an uploaded dataset containing a verified outcome column "
                f"('attrition') and a minimum cohort size of N >= 300 (current rows: {n_samples})."
            )

        return df_scored, cohort_summary

    def predict_file(
        self, file_path: Path | str
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Ingest raw file and execute inference scoring."""
        logger.info(f"Running inference on file: {file_path}")
        df_raw = load_raw_file(file_path)
        return self.score_dataframe(df_raw)


def run_cli_inference(
    input_file: str,
    output_dir: str = "artifacts/predictions",
) -> None:
    """CLI runner function."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    engine = ProductionInferenceEngine()
    df_scored, summary = engine.predict_file(input_file)

    # Save outputs
    csv_file = out_path / "scored_students.csv"
    json_file = out_path / "fusion_payload.json"
    summary_file = out_path / "cohort_summary.json"

    # Minimal tabular dataframe for CSV/Excel
    df_export = df_scored[
        [
            "participant_id",
            "eri_score",
            "risk_band",
            "confidence_note",
            "missing_fields_count",
            "narrative",
        ]
    ].copy()
    df_export.to_csv(csv_file, index=False)

    from src.inference.export import export_inference_bundle

    export_inference_bundle(df_scored, summary, out_path)

    logger.info(f"Inference complete for {len(df_scored)} students.")
    logger.info(f"  CSV exported: {csv_file}")
    logger.info(f"  Fusion payload exported: {json_file}")
    logger.info(f"  Cohort summary: {summary_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ERI Production Inference CLI")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to input CSV or XLSX questionnaire file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/predictions",
        help="Directory to save predictions",
    )
    args = parser.parse_args()
    run_cli_inference(args.file, args.output_dir)
