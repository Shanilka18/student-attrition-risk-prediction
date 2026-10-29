"""Environmental & Financial Survey Codec (EF-01).

Provides bidirectional conversion between survey answer labels and canonical integer codes,
robust header mapping with fuzzy suggestion support, and pressure score calculation.
Tolerant of:
- Case variations (lower / UPPER / TitleCase)
- Whitespace anomalies (leading/trailing/multiple spaces)
- Punctuation variants (en-dash '–', em-dash '—', hyphen '-')
- Currency notations ('LKR', 'lkr', Rs., thousand commas: '50,000' vs '50000')
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
import numpy as np
import pandas as pd


class EnvFinCodec:
    """Production codec for Environmental & Financial survey items."""

    def __init__(
        self,
        schema_path: Optional[Union[str, Path]] = None,
        mapping_path: Optional[Union[str, Path]] = None,
    ):
        # Resolve schema path (support config/ and configs/)
        if schema_path is None:
            for candidate in [
                Path("config/schema_envfin.yaml"),
                Path("configs/schema_envfin.yaml"),
                Path("../config/schema_envfin.yaml"),
            ]:
                if candidate.exists():
                    schema_path = candidate
                    break

        if mapping_path is None:
            for candidate in [
                Path("config/mapping_envfin.yaml"),
                Path("configs/mapping_envfin.yaml"),
                Path("../config/mapping_envfin.yaml"),
            ]:
                if candidate.exists():
                    mapping_path = candidate
                    break

        if not schema_path or not Path(schema_path).exists():
            raise FileNotFoundError(f"Cannot find schema_envfin.yaml at {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = yaml.safe_load(f)

        self.mapping = {}
        if mapping_path and Path(mapping_path).exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                self.mapping = yaml.safe_load(f)

        self.items: Dict[str, Any] = self.schema.get("items", {})
        self.outcomes: Dict[str, Any] = self.schema.get("outcomes", {})
        self.shared_meta: Dict[str, Any] = self.schema.get("shared_meta", {})
        self.derived_indices: Dict[str, Any] = self.schema.get("derived_indices", {})

        # Build normalized lookups: (item_name, normalized_label) -> code
        self._label_to_code: Dict[str, Dict[str, int]] = {}
        self._code_to_label: Dict[str, Dict[int, str]] = {}
        self._build_code_tables()

    @staticmethod
    def normalize_text(text: Any) -> str:
        """Normalizes input string for robust matching."""
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return ""
        s = str(text).strip().lower()
        # Normalize Unicode dashes to standard ASCII hyphen
        s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", s)
        # Remove currency prefixes and thousands commas
        s = re.sub(r"\blkr\b|\brs\.?\b", "", s)
        s = s.replace(",", "")
        # Collapse multiple spaces
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _build_code_tables(self) -> None:
        """Precomputes lookup dictionaries for all items."""
        for name, meta in self.items.items():
            self._label_to_code[name] = {}
            self._code_to_label[name] = {}

            # Standard levels
            levels = meta.get("levels", [])
            for lvl in levels:
                code = lvl["code"]
                label = lvl["label"]
                norm_lbl = self.normalize_text(label)
                self._label_to_code[name][norm_lbl] = code
                self._code_to_label[name][code] = label

                # Additional normalized variants (e.g. without punctuation)
                clean_lbl = re.sub(r"[^\w\s-]", "", norm_lbl)
                self._label_to_code[name][clean_lbl] = code

            # Prefer not to say
            pns = meta.get("prefer_not_to_say")
            if pns:
                pns_code = pns["code"]
                pns_lbl = pns["label"]
                self._label_to_code[name][self.normalize_text(pns_lbl)] = pns_code
                if pns_code not in self._code_to_label[name]:
                    self._code_to_label[name][pns_code] = pns_lbl

            # Structural N/A
            sna = meta.get("structural_na")
            if sna:
                sna_code = sna["code"]
                sna_lbl = sna["label"]
                self._label_to_code[name][self.normalize_text(sna_lbl)] = sna_code
                if sna_code not in self._code_to_label[name]:
                    self._code_to_label[name][sna_code] = sna_lbl

                # Common shorthand for structural N/A
                self._label_to_code[name]["n/a"] = sna_code
                self._label_to_code[name]["not applicable"] = sna_code

    def encode(self, item_name: str, value: Any) -> int:
        """Encodes a survey response string, float or integer into canonical integer code.
        
        Raises ValueError if item unknown or label unmappable.
        """
        if item_name not in self.items:
            raise KeyError(f"Item '{item_name}' is not defined in schema_envfin.yaml.")

        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise ValueError(f"Cannot encode NaN/None value for item '{item_name}'.")

        # If already an integer or integer-like float
        if isinstance(value, (int, np.integer)):
            if int(value) in self._code_to_label[item_name]:
                return int(value)
        if isinstance(value, float) and value.is_integer():
            if int(value) in self._code_to_label[item_name]:
                return int(value)

        norm_val = self.normalize_text(value)
        table = self._label_to_code[item_name]

        if norm_val in table:
            return table[norm_val]

        # Try stripping punctuation
        clean_val = re.sub(r"[^\w\s-]", "", norm_val)
        if clean_val in table:
            return table[clean_val]

        # Prefix matching (e.g. "Less than 30" -> "Less than 30 minutes")
        for stored_norm, code in table.items():
            if norm_val and (norm_val in stored_norm or stored_norm in norm_val):
                return code

        valid_labels = list(self._code_to_label[item_name].values())
        raise ValueError(
            f"Unrecognized response '{value}' for item '{item_name}'. Valid options: {valid_labels}"
        )

    def decode(self, item_name: str, code: int) -> str:
        """Decodes a canonical integer code back to its primary survey label."""
        if item_name not in self.items:
            raise KeyError(f"Item '{item_name}' is not defined in schema_envfin.yaml.")

        code_int = int(code)
        table = self._code_to_label[item_name]
        if code_int in table:
            return table[code_int]

        raise ValueError(
            f"Invalid code '{code}' for item '{item_name}'. Valid codes: {list(table.keys())}"
        )

    def get_pressure(self, item_name: str, value: Any) -> float:
        """Calculates normalized pressure score in [0.0, 1.0] for a given item response.
        
        Returns np.nan for prefer not to say / missing data.
        Returns 0.0 for structural N/A (e.g. fully online students).
        """
        code = self.encode(item_name, value)
        meta = self.items[item_name]
        direction = meta.get("pressure_direction")

        # 1. Structural N/A handling
        sna = meta.get("structural_na")
        if sna and code == sna["code"]:
            return 0.0

        # 2. Prefer not to say handling
        pns = meta.get("prefer_not_to_say")
        if pns and code == pns["code"]:
            return np.nan

        # 3. Nominal pressure mapping
        if direction == "nominal_mapped":
            pmap = meta.get("pressure_map", {})
            label = self.decode(item_name, code)
            norm_label = self.normalize_text(label)
            # Find best match in pressure map
            for map_key, map_val in pmap.items():
                if map_key == "_note":
                    continue
                if map_key in norm_label or norm_label in map_key:
                    return float(map_val)
            return 0.5  # Neutral fallback

        # 4. Ordinal reversed (higher code = lower pressure, e.g. income)
        n_levels = meta["n_levels"]
        if direction == "reversed":
            return float((n_levels - 1 - code) / (n_levels - 1))

        # 5. Ordinal standard (higher code = higher pressure)
        return float(code / (n_levels - 1))

    def map_header(self, raw_header: str) -> Tuple[Optional[str], float, str]:
        """Maps a raw table or Google Forms column header to canonical column name.
        
        Returns (canonical_name, confidence, match_type: 'exact'|'variant'|'suggestion'|'none')
        """
        norm_h = self.normalize_text(raw_header)
        canon_mappings = self.mapping.get("canonical_mappings", {})

        # 1. Check exact match on canonical names
        all_canon_names = list(self.items.keys()) + list(self.shared_meta.keys()) + list(self.outcomes.keys())
        for cname in all_canon_names:
            if norm_h == self.normalize_text(cname):
                return cname, 100.0, "exact"

        # 2. Check canonical mappings dictionary
        for cname, details in canon_mappings.items():
            gf_header = details.get("google_forms_header", "")
            if norm_h == self.normalize_text(gf_header):
                return cname, 100.0, "exact"

            for variant in details.get("accepted_variants", []):
                if norm_h == self.normalize_text(variant):
                    return cname, 100.0, "variant"

        # 3. Fuzzy search fallback (Rapidfuzz or difflib)
        best_name = None
        best_score = 0.0
        try:
            from rapidfuzz import fuzz
            for cname, details in canon_mappings.items():
                candidates = [details.get("google_forms_header", "")] + details.get("accepted_variants", [])
                for cand in candidates:
                    score = fuzz.token_sort_ratio(norm_h, self.normalize_text(cand))
                    if score > best_score:
                        best_score = score
                        best_name = cname
        except ImportError:
            import difflib
            for cname, details in canon_mappings.items():
                candidates = [details.get("google_forms_header", "")] + details.get("accepted_variants", [])
                for cand in candidates:
                    ratio = difflib.SequenceMatcher(None, norm_h, self.normalize_text(cand)).ratio() * 100
                    if ratio > best_score:
                        best_score = ratio
                        best_name = cname

        threshold = self.mapping.get("matching_engine", {}).get("threshold", 88)
        if best_score >= threshold:
            return best_name, float(best_score), "suggestion"

        return None, float(best_score), "none"

    def compute_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes the 4 canonical derived pressure indices on [0.0, 1.0]."""
        result = df.copy()

        # Helper to get series of pressure scores for an item
        def _get_pressure_col(col_name: str) -> pd.Series:
            if col_name not in result.columns:
                return pd.Series(np.nan, index=result.index)
            return result[col_name].apply(
                lambda v: self.get_pressure(col_name, v) if pd.notna(v) else np.nan
            )

        # 1. Financial Stress
        fin_items = [
            "income_band",
            "aid_status",
            "family_support_duty",
            "cost_difficulty",
            "financial_shock",
            "money_worry",
            "job_hours",
        ]
        fin_pressures = pd.DataFrame({it: _get_pressure_col(it) for it in fin_items})
        result["financial_stress_index"] = fin_pressures.mean(axis=1, skipna=True)

        # 2. Commute Burden (0 if fully online)
        commute_items = ["commute_time", "commute_spend", "transport_mode"]
        commute_pressures = pd.DataFrame({it: _get_pressure_col(it) for it in commute_items})
        raw_commute = commute_pressures.mean(axis=1, skipna=True)
        # Detect online students
        is_online = pd.Series(False, index=result.index)
        if "commute_time" in result.columns:
            is_online = is_online | result["commute_time"].astype(str).str.contains("online|applicable|-2", case=False)
        result["commute_burden_index"] = np.where(is_online, 0.0, raw_commute)

        # 3. Digital Access Gap
        dig_items = ["home_internet", "main_device", "mobile_data_plan"]
        dig_pressures = pd.DataFrame({it: _get_pressure_col(it) for it in dig_items})
        result["digital_access_gap_index"] = dig_pressures.mean(axis=1, skipna=True)

        # 4. Study Environment Pressure
        env_items = ["quiet_study_place", "living_situation"]
        env_pressures = pd.DataFrame({it: _get_pressure_col(it) for it in env_items})
        result["study_environment_pressure_index"] = env_pressures.mean(axis=1, skipna=True)

        return result
