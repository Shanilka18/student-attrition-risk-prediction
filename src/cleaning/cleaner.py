"""Deterministic survey data cleaner with audit logging and recovery verification.

Performs:
1. Type coercion and whitespace stripping.
2. Category standardization (case-folding, whitespace trimming).
3. Mapping sentinel strings ('Prefer not to say', 'N/A') to NaN.
4. Duplicate row detection and removal.
5. Boundary clipping for impossible physical values (commute, work hours).
6. Cross-variable logical reconciliation (employment status vs work hours).
7. Difference audit against pristine ground truth.
8. Serialization to data/interim/clean.csv and artifacts/metrics/cleaning_log.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.utils.paths import get_data_dir, get_artifacts_dir, ensure_dir
from src.utils.logging_config import setup_logger

logger = setup_logger("cleaner")


class SurveyCleaner:
    """Deterministic, logged survey data cleaner for undergraduate student records."""

    def __init__(self):
        self.log_actions: List[Dict[str, Any]] = []

    def log_action(self, action_name: str, rows_affected: int, details: str):
        """Record a cleaning action for transparent audit trails."""
        entry = {
            "action": action_name,
            "rows_affected": int(rows_affected),
            "details": details,
        }
        self.log_actions.append(entry)
        logger.info(f"[{action_name}] Affected rows: {rows_affected} - {details}")

    def clean(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean raw survey dataset with full determinism and logging.

        Parameters
        ----------
        df_raw : pd.DataFrame
            The raw, corrupted survey dataset.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Cleaned DataFrame and full cleaning audit dictionary.
        """
        df = df_raw.copy()
        n_initial = len(df)
        self.log_actions.clear()
        self.log_action("initial_load", n_initial, f"Raw dataset loaded with {n_initial} rows and {df.shape[1]} columns.")

        # 1. Exact Duplicate Removal
        duplicate_mask = df.duplicated()
        n_duplicates = int(duplicate_mask.sum())
        if n_duplicates > 0:
            df = df.drop_duplicates(keep="first").reset_index(drop=True)
            self.log_action("remove_duplicates", n_duplicates, f"Removed {n_duplicates} exact duplicate records.")
        else:
            self.log_action("remove_duplicates", 0, "No duplicate records detected.")

        n_after_dedup = len(df)

        # 2. String Cleaning, Whitespace Trimming & Sentinel Value Normalization
        sentinel_tokens = {
            "prefer not to say",
            "not applicable",
            "n/a",
            "na",
            "?",
            "none",
            "null",
            "undefined",
            "",
        }

        # Columns known to contain sentinel non-responses
        for col in df.columns:
            if df[col].dtype == object or isinstance(df[col].iloc[0], str):
                # Clean whitespace
                s_str = df[col].astype(str).str.strip()
                s_lower = s_str.str.lower()

                # Detect sentinels (excluding legitimate categories like device_type 'none')
                if col == "device_type":
                    # 'none' is a valid device category, sentinels are n/a, prefer not to say
                    sentinel_mask = s_lower.isin(["prefer not to say", "not applicable", "n/a", "?"])
                else:
                    sentinel_mask = s_lower.isin(sentinel_tokens)

                n_sentinels = int(sentinel_mask.sum())
                if n_sentinels > 0:
                    df.loc[sentinel_mask, col] = np.nan
                    self.log_action(
                        f"sentinel_to_nan_{col}",
                        n_sentinels,
                        f"Converted {n_sentinels} sentinel entries in '{col}' to NaN.",
                    )

        # 3. Standardize Categorical Encodings (Case Folding & Whitelist Validation)
        # district_type: urban, semi-urban, rural
        if "district_type" in df.columns:
            raw_district = df["district_type"].astype(str).str.strip().str.lower()
            changed_mask = raw_district != df["district_type"]
            df["district_type"] = raw_district
            self.log_action(
                "standardize_district_type",
                int(changed_mask.sum()),
                "Normalized district_type to lowercase stripped tokens ('urban', 'semi-urban', 'rural').",
            )

        # housing_type: home, boarding, hostel, relatives
        if "housing_type" in df.columns:
            raw_housing = df["housing_type"].astype(str).str.strip().str.lower()
            changed_mask = raw_housing != df["housing_type"]
            df["housing_type"] = raw_housing
            self.log_action(
                "standardize_housing_type",
                int(changed_mask.sum()),
                "Normalized housing_type to lowercase stripped tokens ('home', 'boarding', 'hostel', 'relatives').",
            )

        # faculty
        if "faculty" in df.columns:
            df["faculty"] = df["faculty"].astype(str).str.strip()

        # 4. Explicit Type Coercion for Numeric Variables
        numeric_int_cols = [
            "household_income_bracket",
            "financial_stress",
            "scholarship_bursary",
            "dependents_count",
            "internet_reliability",
            "internet_affordability",
            "device_quality",
            "quiet_study_space",
            "part_time_employment",
            "year_of_study",
            "later_gpa_decline",
            "later_engagement_decline",
            "financial_hardship",
            "long_commute",
            "inadequate_digital_access",
            "heavy_part_time_work",
            "attrition",
            "intention_to_withdraw",
        ]

        numeric_float_cols = [
            "commute_time_min",
            "commute_cost_lkr_week",
            "work_hours_per_week",
            "baseline_engagement_score",
            "latent_risk_score",
        ]

        for col in numeric_float_cols:
            if col in df.columns:
                orig_series = df[col]
                converted = pd.to_numeric(orig_series, errors="coerce")
                n_coerced = int((orig_series.notna() & converted.isna()).sum())
                df[col] = converted
                if n_coerced > 0:
                    self.log_action(
                        f"coerce_float_{col}",
                        n_coerced,
                        f"Coerced non-numeric string values in '{col}' to NaN.",
                    )

        for col in numeric_int_cols:
            if col in df.columns:
                orig_series = df[col]
                # Coerce to float first to allow NaN preservation, cast to Int64 (nullable int)
                converted = pd.to_numeric(orig_series, errors="coerce")
                n_coerced = int((orig_series.notna() & converted.isna()).sum())
                # Use pandas nullable integer type Int64 so missing values don't break integer semantics
                df[col] = converted.astype("Int64")
                if n_coerced > 0:
                    self.log_action(
                        f"coerce_int_{col}",
                        n_coerced,
                        f"Coerced non-numeric string values in '{col}' to nullable Int64.",
                    )

        # 5. Boundary Clipping & Impossible Values Validation
        # commute_time_min: must be within [8.0, 240.0]
        if "commute_time_min" in df.columns:
            negative_commute_mask = df["commute_time_min"] < 8.0
            excessive_commute_mask = df["commute_time_min"] > 240.0
            n_neg = int(negative_commute_mask.sum())
            n_excess = int(excessive_commute_mask.sum())

            # Clip impossible negative values to reasonable minimum (10.0 mins)
            df.loc[negative_commute_mask, "commute_time_min"] = 10.0
            # Clip excessive outliers (e.g. 900 mins) to maximum plausible transit (240.0 mins)
            df.loc[excessive_commute_mask, "commute_time_min"] = 240.0

            if n_neg > 0:
                self.log_action(
                    "clip_negative_commute_time",
                    n_neg,
                    "Clipped negative commute times (< 8.0) to 10.0 minutes.",
                )
            if n_excess > 0:
                self.log_action(
                    "clip_excessive_commute_time",
                    n_excess,
                    "Clipped impossible commute times (> 240.0) to 240.0 minutes.",
                )

        # work_hours_per_week: must be within [0.0, 50.0]
        if "work_hours_per_week" in df.columns:
            excess_work_mask = df["work_hours_per_week"] > 50.0
            n_work_excess = int(excess_work_mask.sum())
            df.loc[excess_work_mask, "work_hours_per_week"] = 50.0
            if n_work_excess > 0:
                self.log_action(
                    "clip_excessive_work_hours",
                    n_work_excess,
                    "Clipped extreme working hours (> 50.0 hrs/wk) to 50.0 hours/week.",
                )

        # 6. Logical Contradiction Resolution
        # Rule: If part_time_employment == 0, work_hours_per_week MUST be 0.0
        if "part_time_employment" in df.columns and "work_hours_per_week" in df.columns:
            contradiction_mask = (df["part_time_employment"] == 0) & (df["work_hours_per_week"] > 0)
            n_contradictions = int(contradiction_mask.sum())
            if n_contradictions > 0:
                df.loc[contradiction_mask, "work_hours_per_week"] = 0.0
                self.log_action(
                    "resolve_employment_hours_contradiction",
                    n_contradictions,
                    "Resolved contradiction: student marked unemployed but reported positive work hours; reset hours to 0.0.",
                )

            # Rule: If part_time_employment == 1 and work_hours_per_week == 0, set to median employed hours (18.0)
            employed_zero_mask = (df["part_time_employment"] == 1) & (df["work_hours_per_week"] == 0)
            n_emp_zero = int(employed_zero_mask.sum())
            if n_emp_zero > 0:
                df.loc[employed_zero_mask, "work_hours_per_week"] = 18.0
                self.log_action(
                    "resolve_employed_zero_hours",
                    n_emp_zero,
                    "Resolved contradiction: student marked employed but reported 0 hours; imputed to median 18.0 hrs/wk.",
                )

        # 7. Recompute Derived Exposures Deterministically
        # Guarantees that exposures reflect cleaned, consistent values
        df["financial_hardship"] = (
            (df["household_income_bracket"] <= 2) & (df["financial_stress"] >= 4)
        ).astype("Int64")
        df["long_commute"] = (df["commute_time_min"] > 120.0).astype("Int64")
        df["inadequate_digital_access"] = (
            (df["internet_reliability"] <= 2)
            | df["device_type"].astype(str).str.lower().isin(["phone only", "none"])
        ).astype("Int64")
        df["heavy_part_time_work"] = (df["work_hours_per_week"] > 15.0).astype("Int64")

        audit_summary = {
            "initial_row_count": n_initial,
            "cleaned_row_count": len(df),
            "duplicates_removed": n_duplicates,
            "total_cleaning_actions": len(self.log_actions),
            "actions_log": self.log_actions,
        }

        return df, audit_summary


def compute_recovery_metrics(
    df_clean: pd.DataFrame, df_truth: pd.DataFrame
) -> Dict[str, Any]:
    """Compare cleaned dataset against pristine ground truth to quantify recovery performance.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned dataset produced by SurveyCleaner.
    df_truth : pd.DataFrame
        Pristine pre-corruption dataset.

    Returns
    -------
    Dict[str, Any]
        Recovery metrics including row match rate, feature concordance, and mean absolute errors.
    """
    logger.info("Computing ground truth recovery metrics...")
    # Merge on participant_id
    merged = pd.merge(
        df_clean,
        df_truth,
        on="participant_id",
        suffixes=("_cleaned", "_truth"),
        how="inner",
    )

    n_matched_rows = len(merged)
    row_recovery_rate = float(n_matched_rows / len(df_truth))

    feature_metrics = {}
    numeric_compare_cols = [
        "commute_time_min",
        "commute_cost_lkr_week",
        "work_hours_per_week",
        "baseline_engagement_score",
    ]

    for col in numeric_compare_cols:
        col_c = f"{col}_cleaned"
        col_t = f"{col}_truth"
        if col_c in merged.columns and col_t in merged.columns:
            valid_mask = merged[col_c].notna() & merged[col_t].notna()
            mae = float(np.mean(np.abs(merged.loc[valid_mask, col_c] - merged.loc[valid_mask, col_t])))
            corr = float(np.corrcoef(merged.loc[valid_mask, col_c], merged.loc[valid_mask, col_t])[0, 1])
            feature_metrics[col] = {
                "mae": round(mae, 4),
                "pearson_correlation": round(corr, 4),
                "valid_comparisons": int(valid_mask.sum()),
            }

    categorical_compare_cols = ["district_type", "housing_type", "faculty"]
    for col in categorical_compare_cols:
        col_c = f"{col}_cleaned"
        col_t = f"{col}_truth"
        if col_c in merged.columns and col_t in merged.columns:
            exact_matches = int((merged[col_c].astype(str) == merged[col_t].astype(str)).sum())
            feature_metrics[col] = {
                "exact_match_rate": round(exact_matches / n_matched_rows, 4),
                "matched_count": exact_matches,
            }

    recovery_report = {
        "ground_truth_rows": len(df_truth),
        "cleaned_rows": len(df_clean),
        "matched_rows": n_matched_rows,
        "row_recovery_percentage": round(row_recovery_rate * 100.0, 2),
        "feature_metrics": feature_metrics,
    }
    logger.info(f"Recovery rate: {recovery_report['row_recovery_percentage']}% matched rows.")
    return recovery_report


def run_cleaning_pipeline() -> Tuple[Path, Path, Dict[str, Any]]:
    """Execute complete cleaning pipeline and save cleaned interim data and logs."""
    synthetic_dir = get_data_dir("synthetic")
    interim_dir = get_data_dir("interim")
    metrics_dir = get_artifacts_dir("metrics")

    raw_path = synthetic_dir / "eri_synthetic_raw_latest.csv" if (synthetic_dir / "eri_synthetic_raw_latest.csv").is_file() else synthetic_dir / "eri_synthetic_raw.csv"
    truth_path = synthetic_dir / "eri_synthetic_clean_truth_latest.csv" if (synthetic_dir / "eri_synthetic_clean_truth_latest.csv").is_file() else synthetic_dir / "eri_synthetic_clean_truth.csv"

    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw synthetic data not found at {raw_path}. Run generate.py first.")

    df_raw = pd.read_csv(raw_path)
    cleaner = SurveyCleaner()
    df_clean, audit_summary = cleaner.clean(df_raw)

    # Save cleaned interim data
    clean_path = interim_dir / "clean.csv"
    df_clean.to_csv(clean_path, index=False)

    # Compare recovery if ground truth is present
    if truth_path.is_file():
        df_truth = pd.read_csv(truth_path)
        recovery_report = compute_recovery_metrics(df_clean, df_truth)
        audit_summary["recovery_against_truth"] = recovery_report

    # Save cleaning log
    log_path = metrics_dir / "cleaning_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    logger.info(f"Cleaned dataset saved to: {clean_path}")
    logger.info(f"Cleaning audit log saved to: {log_path}")

    return clean_path, log_path, audit_summary


if __name__ == "__main__":
    clean_p, log_p, summary = run_cleaning_pipeline()
    print(f"\n[SUCCESS] Cleaning pipeline completed:")
    print(f"  Clean interim CSV: {clean_p}")
    print(f"  Audit log JSON:    {log_p}")
    print(f"  Rows before:       {summary['initial_row_count']}")
    print(f"  Rows after:        {summary['cleaned_row_count']}")
    if "recovery_against_truth" in summary:
        print(f"  Row Recovery Rate: {summary['recovery_against_truth']['row_recovery_percentage']}%\n")
