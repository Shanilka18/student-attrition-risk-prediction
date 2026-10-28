"""Stage 1: Multi-format survey file ingestion and normalization."""

import io
import re
from pathlib import Path
from typing import Any, Tuple, Union

import pandas as pd

from mei.cleaning.models import RuleLog


def normalize_string_value(val: Any) -> Any:
    """Normalize string values: strip whitespace, standardize dashes and quotes."""
    if not isinstance(val, str):
        return val

    # Standardize unicode dashes and minus
    s = val.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    # Standardize smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    # Collapse multiple whitespace and strip edges
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detect_header_row(raw_lines: list[str]) -> int:
    """Detect if header row starts after banner/blank introductory rows."""
    for idx, line in enumerate(raw_lines[:10]):
        stripped = line.strip()
        if not stripped:
            continue
        # If line contains delimiters and multiple non-empty tokens
        tokens = [t.strip() for t in re.split(r"[,;\t]", stripped) if t.strip()]
        if len(tokens) >= 3:
            return idx
    return 0


def read_survey_file(
    file_input: Union[str, Path, io.BytesIO, bytes],
    filename_hint: str = "",
    sheet_name: Union[str, int] = 0,
) -> Tuple[pd.DataFrame, RuleLog]:
    """Read CSV, XLSX, or XLS file with encoding fallback and structural cleanup.

    Returns the cleaned DataFrame and a Stage 1 RuleLog entry.
    """
    path_obj = Path(file_input) if isinstance(file_input, (str, Path)) else None
    ext = path_obj.suffix.lower() if path_obj else Path(filename_hint).suffix.lower()

    df: pd.DataFrame
    encoding_used = "utf-8"

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_input, sheet_name=sheet_name)
    else:
        # CSV reading with encoding detection fallback
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
        last_exc: Exception = Exception("Unknown read error")

        if isinstance(file_input, (str, Path)):
            with open(file_input, "rb") as f:
                content = f.read()
        elif isinstance(file_input, io.BytesIO):
            content = file_input.getvalue()
        elif isinstance(file_input, bytes):
            content = file_input
        else:
            raise ValueError(f"Unsupported file input type: {type(file_input)}")

        success = False
        for enc in encodings:
            try:
                text_content = content.decode(enc)
                lines = text_content.splitlines()
                header_idx = detect_header_row(lines)

                df = pd.read_csv(
                    io.StringIO(text_content),
                    skiprows=header_idx,
                    dtype=str,
                )
                encoding_used = enc
                success = True
                break
            except Exception as exc:
                last_exc = exc
                continue

        if not success:
            raise ValueError(
                f"Failed to decode CSV across encodings {encodings}. Error: {last_exc}"
            )

    rows_before = len(df)
    cols_before = len(df.columns)

    # 1. Drop fully empty rows and columns
    df = df.dropna(how="all", axis=0)
    df = df.dropna(how="all", axis=1)
    df = df.reset_index(drop=True)

    # 2. Normalize headers
    clean_cols = [
        str(normalize_string_value(c)) if pd.notna(c) else f"Unnamed_{i}"
        for i, c in enumerate(df.columns)
    ]
    df.columns = pd.Index(clean_cols)

    # 3. Normalize string cell contents
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].map(normalize_string_value)

    # Convert empty strings to NA
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    rows_affected = rows_before - len(df)
    cols_affected = cols_before - len(df.columns)

    log = RuleLog(
        rule_id="R01_INGESTION_CLEANUP",
        rule_name="Read File & Structural Normalization",
        rows_affected=rows_affected,
        columns_affected=list(df.columns),
        action="drop_empty_and_normalize_text",
        details=(
            f"Encoding: {encoding_used}. Dropped {rows_affected} empty rows "
            f"and {cols_affected} empty columns."
        ),
    )

    return df, log
