"""Stage 2: Schema-driven header mapping and column auditing."""

from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from rapidfuzz import fuzz

from mei.cleaning.models import HeaderMappingResult, RuleLog
from mei.io.schema import MappingConfig, SurveySchema, load_mapping, load_schema


def normalize_header(header: str) -> str:
    """Normalize a header string for invariant matching."""
    s = header.lower().strip()
    # Normalize dashes and quotes
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    # Remove period formatting differences e.g. [period 1: weeks 1-4] vs [p1]
    # Replace multiple spaces with single space
    s = " ".join(s.split())
    return s


def map_headers(
    df: pd.DataFrame,
    mapping_cfg: Optional[MappingConfig] = None,
    schema: Optional[SurveySchema] = None,
    mode: str = "training",
    confidence_threshold: float = 88.0,
) -> Tuple[pd.DataFrame, HeaderMappingResult, RuleLog]:
    """Map raw dataframe headers to canonical column names.

    Matches in order:
    1. Exact match against canonical name or variants.
    2. Normalized match (casing, dashes, spacing).
    3. Fuzzy suggestions (similarity >= threshold) returned to caller.

    Returns the mapped DataFrame, HeaderMappingResult, and Stage 2 RuleLog.
    """
    mapping = mapping_cfg or load_mapping()
    sch = schema or load_schema()

    raw_headers = list(df.columns)
    column_rename_map: Dict[str, str] = {}
    matched_canonical: Set[str] = set()
    suggestions: Dict[str, List[Dict[str, float]]] = {}
    unknown_extra: List[str] = []

    # Build lookup dictionaries from mapping.yaml
    exact_lookup: Dict[str, str] = {}
    norm_lookup: Dict[str, str] = {}
    all_variants_flat: List[
        Tuple[str, str, str]
    ] = []  # (canonical, variant, norm_variant)

    for canonical_name, variants in mapping.columns.items():
        exact_lookup[canonical_name] = canonical_name
        norm_lookup[normalize_header(canonical_name)] = canonical_name

        for var in variants:
            exact_lookup[var] = canonical_name
            norm_v = normalize_header(var)
            norm_lookup[norm_v] = canonical_name
            all_variants_flat.append((canonical_name, var, norm_v))

    # Perform matching for each raw header
    for raw_h in raw_headers:
        # 1. Exact match
        if raw_h in exact_lookup:
            canon = exact_lookup[raw_h]
            column_rename_map[raw_h] = canon
            matched_canonical.add(canon)
            continue

        # 2. Normalized match
        norm_h = normalize_header(raw_h)
        if norm_h in norm_lookup:
            canon = norm_lookup[norm_h]
            column_rename_map[raw_h] = canon
            matched_canonical.add(canon)
            continue

        # 3. Fuzzy matching (candidate suggestions)
        raw_suggestions: List[Dict[str, float]] = []
        for canon, orig_var, norm_var in all_variants_flat:
            score = fuzz.token_sort_ratio(norm_h, norm_var)
            if score >= confidence_threshold:
                raw_suggestions.append(
                    {
                        "canonical": canon,
                        "variant": orig_var,
                        "confidence": round(float(score), 1),
                    }
                )

        if raw_suggestions:
            # Sort descending by confidence and deduplicate by canonical
            raw_suggestions.sort(key=lambda x: x["confidence"], reverse=True)
            seen_canon = set()
            top_suggestions = []
            for item in raw_suggestions:
                if item["canonical"] not in seen_canon:
                    seen_canon.add(item["canonical"])
                    top_suggestions.append(item)
            suggestions[raw_h] = top_suggestions[:3]
        else:
            unknown_extra.append(raw_h)

    # Determine required columns by mode
    feature_names = sch.get_feature_names()
    outcome_names = sch.get_outcome_names()

    required_canonical: List[str] = ["student_id"] + list(feature_names)
    if mode == "training":
        required_canonical += ["consent"] + list(outcome_names)

    missing_required = [c for c in required_canonical if c not in matched_canonical]

    # Apply rename to matched columns
    mapped_df = df.rename(columns=column_rename_map)

    result = HeaderMappingResult(
        mapped_columns=column_rename_map,
        missing_required=missing_required,
        unknown_extra=unknown_extra,
        suggestions=suggestions,
    )

    log = RuleLog(
        rule_id="R02_HEADER_MAPPING",
        rule_name="Header Mapping & Column Auditing",
        rows_affected=0,
        columns_affected=list(column_rename_map.keys()),
        action="rename_mapped_columns",
        details=(
            f"Mapped {len(column_rename_map)}/{len(raw_headers)} headers. "
            f"Missing required: {len(missing_required)}. "
            f"Unknown extra: {len(unknown_extra)}. "
            f"Fuzzy suggestions: {len(suggestions)}."
        ),
    )

    return mapped_df, result, log
