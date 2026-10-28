"""Stage 10 & Orchestration: 10-stage survey data cleaning pipeline."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from mei.cleaning.encoder import encode_and_validate_indicators
from mei.cleaning.filters import (
    apply_missingness_policy,
    detect_careless_responding,
    filter_attention_check,
    filter_consent,
    resolve_duplicates,
)
from mei.cleaning.mapper import map_headers
from mei.cleaning.models import (
    DataQualityReport,
    RowQualityDetail,
    RuleLog,
    TrafficLightStatus,
)
from mei.cleaning.outcomes import process_outcomes
from mei.cleaning.reader import read_survey_file
from mei.config import DATA_DIR, RUNS_DIR
from mei.io.schema import (
    LabelsConfig,
    MappingConfig,
    SurveySchema,
    load_labels,
    load_mapping,
    load_schema,
)

logger = logging.getLogger(__name__)


class CleaningPipeline:
    """Production data-quality cleaning pipeline for training and scoring modes."""

    def __init__(
        self,
        schema: Optional[SurveySchema] = None,
        mapping: Optional[MappingConfig] = None,
        labels: Optional[LabelsConfig] = None,
    ) -> None:
        self.schema = schema or load_schema()
        self.mapping = mapping or load_mapping()
        self.labels = labels or load_labels()

    def clean(
        self,
        file_input: Union[str, Path, bytes, pd.DataFrame],
        mode: str = "training",
        run_id: Optional[str] = None,
        filter_attention: bool = True,
        drop_careless: bool = True,
        max_missing_pct: float = 0.25,
        save_artifacts: bool = True,
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Execute the complete 10-stage survey data cleaning pipeline."""
        if mode not in ["training", "scoring"]:
            raise ValueError(
                f"Invalid mode: '{mode}'. Must be 'training' or 'scoring'."
            )

        now_str = datetime.now(timezone.utc).isoformat()
        current_run_id = (
            run_id
            or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        )

        input_name = (
            str(file_input)
            if isinstance(file_input, (str, Path))
            else "in_memory_dataframe"
        )
        rules_applied: List[RuleLog] = []
        warnings: List[str] = []
        messages: List[str] = []

        # -------------------------------------------------------------
        # Stage 1: Ingestion & Normalization
        # -------------------------------------------------------------
        if isinstance(file_input, pd.DataFrame):
            df = file_input.copy()
            stage1_log = RuleLog(
                rule_id="R01_INGESTION_CLEANUP",
                rule_name="DataFrame Ingestion",
                rows_affected=0,
                columns_affected=list(df.columns),
                action="pass_through_dataframe",
                details="Ingested in-memory DataFrame.",
            )
        else:
            df, stage1_log = read_survey_file(file_input, filename_hint=input_name)

        rules_applied.append(stage1_log)
        rows_in = len(df)

        if rows_in == 0:
            report = DataQualityReport(
                run_id=current_run_id,
                timestamp=now_str,
                mode=mode,
                input_file=input_name,
                rows_in=0,
                rows_out=0,
                retention_rate=0.0,
                status=TrafficLightStatus.RED,
                rules_applied=rules_applied,
                warnings=["Input dataset contains zero rows."],
                messages=["Critical failure: Empty dataset."],
            )
            return pd.DataFrame(), report

        # Tracking for scoring mode diagnostics
        rejection_map: Dict[int, List[str]] = {idx: [] for idx in df.index}

        # -------------------------------------------------------------
        # Stage 2: Header Mapping
        # -------------------------------------------------------------
        mapped_df, mapping_res, stage2_log = map_headers(
            df,
            mapping_cfg=self.mapping,
            schema=self.schema,
            mode=mode,
        )
        rules_applied.append(stage2_log)

        if mapping_res.missing_required:
            warnings.append(
                f"Missing required canonical columns: {mapping_res.missing_required}"
            )
        if mapping_res.unknown_extra:
            warnings.append(
                f"Ignored unrecognized columns: {mapping_res.unknown_extra}"
            )
        if mapping_res.suggestions:
            s_count = len(mapping_res.suggestions)
            messages.append(f"Generated fuzzy suggestions for {s_count} columns.")

        # In training mode, fatal if required columns are missing
        if mode == "training" and mapping_res.missing_required:
            missing_req = mapping_res.missing_required
            report = DataQualityReport(
                run_id=current_run_id,
                timestamp=now_str,
                mode=mode,
                input_file=input_name,
                rows_in=rows_in,
                rows_out=0,
                retention_rate=0.0,
                status=TrafficLightStatus.RED,
                rules_applied=rules_applied,
                mapping_summary={
                    "mapped": mapping_res.mapped_columns,
                    "missing_required": mapping_res.missing_required,
                    "unknown_extra": mapping_res.unknown_extra,
                    "suggestions": mapping_res.suggestions,
                },
                warnings=warnings,
                messages=[
                    "Fatal error: Missing required columns in training mode: "
                    f"{missing_req}"
                ],
            )
            return pd.DataFrame(), report

        # -------------------------------------------------------------
        # Stage 3: Consent & Attention Check Filters
        # -------------------------------------------------------------
        # 3a. Consent
        stage3a_df, non_consent_idx, stage3a_log = filter_consent(mapped_df, mode=mode)
        rules_applied.append(stage3a_log)
        for idx in non_consent_idx:
            rejection_map[idx].append("consent_refused")

        # 3b. Attention Check
        stage3b_df, attn_fail_idx, stage3b_log = filter_attention_check(
            stage3a_df, mode=mode, enabled=filter_attention
        )
        rules_applied.append(stage3b_log)
        for idx in attn_fail_idx:
            rejection_map[idx].append("attention_check_failed")

        curr_df = stage3b_df

        # -------------------------------------------------------------
        # Stages 4 & 5: Typo-Tolerant Codec & Range Validation
        # -------------------------------------------------------------
        encoded_df, stage4_log, stage5_log = encode_and_validate_indicators(
            curr_df, schema=self.schema, typo_threshold=90.0
        )
        rules_applied.append(stage4_log)
        rules_applied.append(stage5_log)
        curr_df = encoded_df

        # -------------------------------------------------------------
        # Stage 6: Duplicate Resolution
        # -------------------------------------------------------------
        stage6_df, dup_idx, stage6_log = resolve_duplicates(curr_df, mode=mode)
        rules_applied.append(stage6_log)
        for idx in dup_idx:
            rejection_map[idx].append("duplicate_student_id")

        curr_df = stage6_df

        # -------------------------------------------------------------
        # Stage 7: Careless Responding (Straight-Liners)
        # -------------------------------------------------------------
        feature_names = self.schema.get_feature_names()
        present_features = [c for c in feature_names if c in curr_df.columns]

        stage7_df, straight_idx, stage7_log = detect_careless_responding(
            curr_df, mode=mode, drop=drop_careless, feature_cols=present_features
        )
        rules_applied.append(stage7_log)
        for idx in straight_idx:
            rejection_map[idx].append("straight_lining")

        curr_df = stage7_df

        # -------------------------------------------------------------
        # Stage 8: Missing Data Policy (>25% dropped; NO IMPUTATION)
        # -------------------------------------------------------------
        stage8_df, missing_summary, excess_miss_idx, stage8_log = (
            apply_missingness_policy(
                curr_df,
                mode=mode,
                max_missing_pct=max_missing_pct,
                feature_cols=present_features,
            )
        )
        rules_applied.append(stage8_log)
        for idx in excess_miss_idx:
            rejection_map[idx].append("excessive_missing_features")

        curr_df = stage8_df

        # -------------------------------------------------------------
        # Stage 9: Outcome Handling
        # -------------------------------------------------------------
        stage9_df, missing_out_idx, class_balance, stage9_log = process_outcomes(
            curr_df, labels_cfg=self.labels, mode=mode, threshold=60.0
        )
        rules_applied.append(stage9_log)
        for idx in missing_out_idx:
            rejection_map[idx].append("missing_outcome_items")

        curr_df = stage9_df

        # -------------------------------------------------------------
        # Stage 10: Canonical Output & Audit Report Generation
        # -------------------------------------------------------------
        # Assemble ordered canonical columns
        canonical_cols: List[str] = []
        if "student_id" in curr_df.columns:
            canonical_cols.append("student_id")
        if "timestamp" in curr_df.columns:
            canonical_cols.append("timestamp")

        # 28 canonical features
        for f in feature_names:
            if f in curr_df.columns:
                canonical_cols.append(f)

        # Outcomes if present
        for o in ["attrition_intent_score", "at_risk"]:
            if o in curr_df.columns:
                canonical_cols.append(o)

        clean_df = curr_df[canonical_cols].copy()
        rows_out = len(clean_df)
        retention_rate = rows_out / rows_in if rows_in > 0 else 0.0

        # Build rejected rows audit for scoring mode
        rejected_details: List[RowQualityDetail] = []
        id_series = df["student_id"] if "student_id" in df.columns else None
        for orig_idx, reasons in rejection_map.items():
            stu_id = (
                str(id_series.loc[orig_idx])
                if id_series is not None and orig_idx in id_series.index
                else f"Row_{orig_idx}"
            )
            scorable = (len(reasons) == 0) and (orig_idx in clean_df.index)
            if not scorable:
                rejected_details.append(
                    RowQualityDetail(
                        row_index=int(orig_idx),
                        student_id=stu_id,
                        scorable=scorable,
                        rejection_reasons=reasons or ["unspecified_filter"],
                    )
                )

        # Item marginal distributions
        item_distributions: Dict[str, Dict[str, Any]] = {}
        for col in present_features:
            s_counts = clean_df[col].dropna().value_counts().to_dict()
            total_valid = int(clean_df[col].dropna().count())
            dist_dict = {
                str(int(k)): {
                    "count": int(v),
                    "pct": round(v / max(1, total_valid) * 100, 2),
                }
                for k, v in sorted(s_counts.items())
            }
            item_distributions[col] = {
                "total_valid": total_valid,
                "missing_count": int(clean_df[col].isna().sum()),
                "codes": dist_dict,
            }

        # Quality warnings
        if rows_out < 300:
            warnings.append(
                f"Small sample size warning: {rows_out} clean rows (<300 target)."
            )
        if missing_summary.get("overall_missing_pct", 0) > 10.0:
            m_pct = missing_summary["overall_missing_pct"]
            warnings.append(f"Elevated missingness: {m_pct}% feature cells missing.")
        if class_balance:
            at_risk_p = class_balance["at_risk_pct"]
            if at_risk_p < 15.0 or at_risk_p > 45.0:
                warnings.append(
                    f"Class imbalance warning: at-risk prevalence {at_risk_p}% "
                    f"(typical range 15%–45%)."
                )

        # Traffic light status evaluation
        if rows_out == 0 or retention_rate < 0.50:
            status = TrafficLightStatus.RED
            messages.append("Status RED: High data loss or zero valid records.")
        elif (
            rows_out < 300
            or missing_summary.get("overall_missing_pct", 0) > 10.0
            or (
                class_balance
                and (
                    class_balance["at_risk_pct"] < 15.0
                    or class_balance["at_risk_pct"] > 45.0
                )
            )
            or mapping_res.unknown_extra
            or retention_rate < 0.70
        ):
            status = TrafficLightStatus.AMBER
            messages.append("Status AMBER: Dataset is usable with quality advisories.")
        else:
            status = TrafficLightStatus.GREEN
            messages.append(
                "Status GREEN: High quality survey dataset ready for pipeline modeling."
            )

        report = DataQualityReport(
            run_id=current_run_id,
            timestamp=now_str,
            mode=mode,
            input_file=input_name,
            rows_in=rows_in,
            rows_out=rows_out,
            retention_rate=retention_rate,
            status=status,
            rules_applied=rules_applied,
            missingness_summary=missing_summary,
            class_balance=class_balance,
            item_distributions=item_distributions,
            mapping_summary={
                "mapped_columns": mapping_res.mapped_columns,
                "missing_required": mapping_res.missing_required,
                "unknown_extra": mapping_res.unknown_extra,
                "suggestions": mapping_res.suggestions,
            },
            rejected_rows=rejected_details,
            warnings=warnings,
            messages=messages,
        )

        # Persist artifacts if requested
        if save_artifacts:
            self._save_artifacts(clean_df, report, current_run_id)

        return clean_df, report

    def _save_artifacts(
        self, clean_df: pd.DataFrame, report: DataQualityReport, run_id: str
    ) -> None:
        """Save clean parquet and JSON data quality report."""
        interim_dir = DATA_DIR / "interim"
        interim_dir.mkdir(parents=True, exist_ok=True)

        # 1. Parquet artifact
        parquet_path = interim_dir / f"clean_{run_id}.parquet"
        clean_df.to_parquet(parquet_path, engine="fastparquet", index=False)

        # 2. Latest clean parquet pointer / copy
        latest_parquet = interim_dir / "clean_latest.parquet"
        clean_df.to_parquet(latest_parquet, engine="fastparquet", index=False)

        # 3. JSON Quality Report
        run_artifacts_dir = RUNS_DIR / run_id
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = run_artifacts_dir / "data_quality_report.json"
        interim_report_path = interim_dir / "data_quality_report.json"

        rep_dict = report.to_dict()
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(rep_dict, f, indent=2)
        with open(interim_report_path, "w", encoding="utf-8") as f:
            json.dump(rep_dict, f, indent=2)

        logger.info(
            "Saved cleaned dataset to %s and report to %s",
            parquet_path,
            report_json_path,
        )


io_bytes = Union[bytes, Any]
