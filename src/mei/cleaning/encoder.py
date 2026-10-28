"""Stages 4 and 5: Typo-tolerant label encoding and code standardization."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from mei.cleaning.models import RuleLog
from mei.io.codec import SurveyCodec
from mei.io.schema import SurveySchema, load_schema


def encode_single_value(
    val: Any,
    col_name: str,
    codec: SurveyCodec,
    allowed_labels_dict: Dict[int, str],
    typo_threshold: float = 90.0,
) -> Tuple[Optional[int], bool, bool]:
    """Encode a single survey cell into an ordinal integer code in [1, 5].

    Returns:
        (code_or_none, typo_corrected, unconvertible_flag)
    """
    if pd.isna(val) or val is None:
        return None, False, False

    # 1. Already an integer or float
    if isinstance(val, (int, float, np.integer, np.floating)):
        int_v = int(val)
        if 1 <= int_v <= 5:
            return int_v, False, False
        else:
            return None, False, False  # Out of range

    # 2. String representation
    s = str(val).strip()
    if not s or s.lower() in ["nan", "none", "null", ""]:
        return None, False, False

    # If it's a numeric string like "1", "2.0"
    try:
        float_v = float(s)
        int_v = int(float_v)
        if 1 <= int_v <= 5:
            return int_v, False, False
        else:
            return None, False, False
    except ValueError:
        pass

    # 3. Codec exact / normalized match
    encoded = codec.encode(col_name, s, strict=False)
    if encoded is not None and 1 <= encoded <= 5:
        return encoded, False, False

    # 4. Fuzzy typo matching against allowed text labels for this column
    if allowed_labels_dict:
        norm_s = s.lower().replace("\u2013", "-").replace("“", '"').strip()
        best_code = None
        best_score = 0.0

        for code, label in allowed_labels_dict.items():
            norm_label = label.lower().replace("\u2013", "-").replace("“", '"').strip()
            score = fuzz.ratio(norm_s, norm_label)
            if score > best_score:
                best_score = score
                best_code = code

        if best_score >= typo_threshold and best_code is not None:
            return best_code, True, False

    # 5. Unconvertible value
    return None, False, True


def encode_and_validate_indicators(
    df: pd.DataFrame,
    schema: Optional[SurveySchema] = None,
    typo_threshold: float = 90.0,
) -> Tuple[pd.DataFrame, RuleLog, RuleLog]:
    """Execute Stage 4 (Typo-Tolerant Encoding) and Stage 5 (Range Validation).

    Returns the transformed DataFrame, Stage 4 RuleLog, and Stage 5 RuleLog.
    """
    sch = schema or load_schema()
    codec = SurveyCodec(sch)

    # Build schema text labels map for indicators and outcomes
    schema_labels_map: Dict[str, Dict[int, str]] = {}
    for col in sch.columns:
        if col.text_labels:
            schema_labels_map[col.name] = {
                int(k): v for k, v in col.text_labels.items()
            }

    out_df = df.copy()

    # Identify indicator and outcome columns present in out_df
    target_cols = [
        c
        for c in out_df.columns
        if c in schema_labels_map and c not in ["consent", "attention_check"]
    ]

    total_typos = 0
    total_unconvertible = 0
    total_out_of_range = 0
    cells_converted = 0

    for col in target_cols:
        col_labels_dict = schema_labels_map.get(col, {})
        series = out_df[col]

        # Fast path if already numeric and all within valid bounds [1, 5]
        if pd.api.types.is_numeric_dtype(series):
            in_range = series.isna() | series.between(1, 5)
            if in_range.all():
                out_df[col] = series.astype(float)
                cells_converted += int(series.notna().sum())
                continue

        # Fast unique-value lookup cache: encode each unique representation once
        unique_vals = series.unique()
        encoding_cache: Dict[Any, Tuple[Optional[float], bool, bool, bool]] = {}

        for val in unique_vals:
            if isinstance(val, (int, float, np.integer, np.floating)):
                if not pd.isna(val) and (val < 1 or val > 5):
                    encoding_cache[val] = (np.nan, False, False, True)
                    continue

            code, is_typo, is_unconvertible = encode_single_value(
                val,
                col,
                codec,
                col_labels_dict,
                typo_threshold=typo_threshold,
            )
            c_val = float(code) if code is not None else np.nan
            encoding_cache[val] = (c_val, is_typo, is_unconvertible, False)

        # Count occurrences using value_counts
        val_counts = series.value_counts(dropna=False)
        for val, count in val_counts.items():
            if val in encoding_cache:
                c_val, is_typo, is_unconvertible, is_oor = encoding_cache[val]
                if is_oor:
                    total_out_of_range += count
                if is_typo:
                    total_typos += count
                if is_unconvertible:
                    total_unconvertible += count
                if not np.isnan(c_val):
                    cells_converted += count

        code_map = {k: v[0] for k, v in encoding_cache.items()}
        out_df[col] = series.map(code_map).astype(float)

    # Stage 4 Log
    stage4_log = RuleLog(
        rule_id="R04_CODEC_TYPO_ENCODING",
        rule_name="Text-to-Code Conversion & Typo Tolerance",
        rows_affected=total_typos + total_unconvertible,
        columns_affected=target_cols,
        action="encode_labels_with_fuzzy_tolerance",
        details=(
            f"Converted {cells_converted} cells. "
            f"Corrected {total_typos} typos (threshold >= {typo_threshold}%). "
            f"Converted {total_unconvertible} unrecognizable entries to NaN."
        ),
    )

    # Stage 5 Log
    stage5_log = RuleLog(
        rule_id="R05_RANGE_TYPE_VALIDATION",
        rule_name="Range and Ordinal Type Validation (1-5)",
        rows_affected=total_out_of_range,
        columns_affected=target_cols,
        action="convert_out_of_range_to_nan",
        details=(
            f"Enforced ordinal scale [1, 5]. "
            f"Converted {total_out_of_range} out-of-range entries to NaN."
        ),
    )

    return out_df, stage4_log, stage5_log
