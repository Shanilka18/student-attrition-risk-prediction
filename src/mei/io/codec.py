"""Bidirectional conversion between survey text labels and numeric codes."""

import re
from typing import Any, Dict, Optional, Union

import pandas as pd

from mei.io.schema import SurveySchema, load_schema


def normalize_label(text: Union[str, Any]) -> str:
    """Normalize label text for robust matching.

    Tolerant of:
    - Variations in case (case-insensitive)
    - Consecutive/irregular whitespace
    - Unicode dashes: en-dash (\u2013), em-dash (\u2014), minus (\u2212) vs hyphen (-)
    - Leading/trailing punctuation/quotes
    """
    if not isinstance(text, str):
        text = str(text)

    # Normalize unicode dashes and quotes to standard ASCII
    s = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")

    # Lowercase and clean spacing
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


class SurveyCodec:
    """Encoder and decoder between survey text labels and ordinal codes."""

    def __init__(self, schema: Optional[SurveySchema] = None) -> None:
        self.schema = schema or load_schema()
        self._code_to_label: Dict[str, Dict[int, str]] = {}
        self._norm_label_to_code: Dict[str, Dict[str, int]] = {}

        self._build_lookup_tables()

    def _build_lookup_tables(self) -> None:
        """Construct bidirectional lookup tables with normalized text variants."""
        for col in self.schema.columns:
            if not col.text_labels:
                continue

            self._code_to_label[col.name] = {}
            self._norm_label_to_code[col.name] = {}

            for code, label in col.text_labels.items():
                int_code = int(code)
                self._code_to_label[col.name][int_code] = label

                # Index exact normalized label
                norm = normalize_label(label)
                self._norm_label_to_code[col.name][norm] = int_code

                # Also add common practical variations
                # E.g. "0 days (never)" -> "0 days", "never"
                if "(" in label and ")" in label:
                    inside = re.search(r"\((.*?)\)", label)
                    outside = re.sub(r"\s*\(.*?\)", "", label)
                    if inside:
                        self._norm_label_to_code[col.name][
                            normalize_label(inside.group(1))
                        ] = int_code
                    if outside:
                        self._norm_label_to_code[col.name][normalize_label(outside)] = (
                            int_code
                        )

                # E.g. "1-2 days" -> "1 to 2 days"
                if "-" in norm:
                    self._norm_label_to_code[col.name][norm.replace("-", " to ")] = (
                        int_code
                    )

    def encode(self, col_name: str, value: Any, strict: bool = False) -> Optional[int]:
        """Convert a text label or raw value to its canonical numeric code."""
        if pd.isna(value) or value is None:
            return None

        # If already an integer or integer-like float
        if isinstance(value, (int, float)):
            try:
                int_val = int(value)
                col_schema = self.schema.get_column(col_name)
                if col_schema and col_schema.allowed_codes:
                    if int_val in col_schema.allowed_codes:
                        return int_val
            except (ValueError, OverflowError):
                pass

        val_str = str(value).strip()

        # If the string is a pure digit: e.g. "3" or "3.0"
        try:
            float_val = float(val_str)
            if float_val.is_integer():
                int_val = int(float_val)
                col_schema = self.schema.get_column(col_name)
                if col_schema and col_schema.allowed_codes:
                    if int_val in col_schema.allowed_codes:
                        return int_val
        except ValueError:
            pass

        # Look up via normalized text dictionary
        col_table = self._norm_label_to_code.get(col_name, {})
        norm_val = normalize_label(val_str)

        if norm_val in col_table:
            return col_table[norm_val]

        if strict:
            allowed = list(col_table.keys())
            raise ValueError(
                f"Unknown value '{value}' for column '{col_name}'. Allowed: {allowed}"
            )
        return None

    def decode(self, col_name: str, code: Any, strict: bool = False) -> Optional[str]:
        """Convert a numeric code to its canonical survey text label."""
        if pd.isna(code) or code is None:
            return None

        try:
            int_code = int(code)
        except (ValueError, TypeError):
            if strict:
                raise ValueError(f"Code '{code}' cannot be converted to integer.")
            return None

        col_table = self._code_to_label.get(col_name, {})
        if int_code in col_table:
            return col_table[int_code]

        if strict:
            allowed = list(col_table.keys())
            raise ValueError(
                f"Unknown code '{code}' for column '{col_name}'. Allowed: {allowed}"
            )
        return None

    def encode_dataframe(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """Encode all recognizable text columns in a DataFrame into numeric codes."""
        out = df if inplace else df.copy()
        for col_name in out.columns:
            if col_name in self._norm_label_to_code:
                out[col_name] = out[col_name].apply(
                    lambda v: self.encode(col_name, v, strict=False)
                )
        return out

    def decode_dataframe(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """Decode all coded columns in a DataFrame into canonical text labels."""
        out = df if inplace else df.copy()
        for col_name in out.columns:
            if col_name in self._code_to_label:
                out[col_name] = out[col_name].apply(
                    lambda v: self.decode(col_name, v, strict=False)
                )
        return out
