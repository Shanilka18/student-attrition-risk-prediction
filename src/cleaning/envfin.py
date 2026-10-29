"""Reusable Environmental & Financial survey ingestion and cleaning pipeline."""

from __future__ import annotations

import csv
import difflib
import io
import json
import math
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from charset_normalizer import from_bytes

from src.preprocessing.codec import EnvFinCodec
from src.utils.paths import get_project_root


@dataclass(frozen=True)
class CleaningResult:
    """Cleaned canonical rows, report, and run artifact paths."""

    data: pd.DataFrame
    report: dict[str, Any]
    parquet_path: Path | None = None
    report_path: Path | None = None


class EnvironmentalFinancialCleaner:
    """Configuration-driven cleaner shared by training, scoring, and uploads."""

    def __init__(
        self,
        codec: EnvFinCodec | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        root = get_project_root()
        self.codec = codec or EnvFinCodec(
            schema_path=root / "config" / "schema_envfin.yaml",
            mapping_path=root / "config" / "mapping_envfin.yaml",
        )
        self.settings = settings or self._load_settings(root)
        self.items = list(self.codec.items)
        self.outcomes = list(self.codec.outcomes)
        self.shared = list(self.codec.shared_meta)
        self.logs: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.row_reasons: dict[Any, list[str]] = {}
        self.mapping_suggestions: list[dict[str, Any]] = []
        self.missing_required: list[str] = []
        self.unknown_columns: list[str] = []

    @staticmethod
    def _load_settings(root: Path) -> dict[str, Any]:
        import yaml

        config_path = root / "config" / "project.yaml"
        with config_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config.get("cleaning", {})

    def _log(
        self,
        rule: str,
        before: pd.DataFrame,
        after: pd.DataFrame,
        action: str,
    ) -> dict[str, Any]:
        shared_indices = before.index.intersection(after.index)
        shared_columns = before.columns.union(after.columns)
        before_aligned = before.reindex(index=shared_indices, columns=shared_columns)
        after_aligned = after.reindex(index=shared_indices, columns=shared_columns)
        before_text = before_aligned.astype("string").fillna("<NA>")
        after_text = after_aligned.astype("string").fillna("<NA>")
        changed_rows = (
            before_text.ne(after_text).any(axis=1).sum() if len(shared_indices) else 0
        )
        affected = int(abs(len(after) - len(before)) + changed_rows)
        entry = {
            "rule": rule,
            "rows": affected,
            "columns": list(after.columns),
            "action": action,
            "rows_before": len(before),
            "rows_after": len(after),
        }
        self.logs.append(entry)
        return entry

    def _add_reason(self, index: Any, reason: str) -> None:
        self.row_reasons.setdefault(index, []).append(reason)

    @staticmethod
    def _clean_text(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        value = value.strip()
        value = re.sub(r"[\u2010-\u2015\u2212]", "-", value)
        value = value.translate(
            str.maketrans(
                {
                    "\u2018": "'",
                    "\u2019": "'",
                    "\u201c": '"',
                    "\u201d": '"',
                }
            )
        )
        return value if value else np.nan

    def _header_match(self, header: Any) -> tuple[str | None, float, str]:
        normalized_header = self.codec.normalize_text(header).replace("_", " ")
        normalized_header = re.sub(r"[^\w\s]", " ", normalized_header)
        normalized_header = re.sub(r"\s+", " ", normalized_header).strip()
        for item, details in self.codec.mapping.get("canonical_mappings", {}).items():
            candidates = [item, details.get("google_forms_header", "")]
            candidates.extend(details.get("accepted_variants", []))
            for index, candidate in enumerate(candidates):
                normalized_candidate = self.codec.normalize_text(candidate).replace(
                    "_", " "
                )
                normalized_candidate = re.sub(r"[^\w\s]", " ", normalized_candidate)
                normalized_candidate = re.sub(r"\s+", " ", normalized_candidate).strip()
                if normalized_header == normalized_candidate:
                    return item, 100.0, "exact" if index == 0 else "variant"
        canonical, confidence, match_type = self.codec.map_header(str(header))
        if match_type in {"exact", "variant"}:
            return canonical, confidence, match_type
        normalized_header = self.codec.normalize_text(header)
        candidates: list[tuple[str, str]] = []
        for item, details in self.codec.mapping.get("canonical_mappings", {}).items():
            candidates.append((item, details.get("google_forms_header", "")))
            candidates.extend(
                (item, variant) for variant in details.get("accepted_variants", [])
            )
        scores: list[tuple[float, str]] = []
        try:
            from rapidfuzz import fuzz

            scores = [
                (
                    float(
                        fuzz.token_sort_ratio(
                            normalized_header, self.codec.normalize_text(candidate)
                        )
                    ),
                    item,
                )
                for item, candidate in candidates
            ]
        except ImportError:
            scores = [
                (
                    difflib.SequenceMatcher(
                        None,
                        normalized_header,
                        self.codec.normalize_text(candidate),
                    ).ratio()
                    * 100,
                    item,
                )
                for item, candidate in candidates
            ]
        if not scores:
            return None, 0.0, "none"
        best_score, suggested = max(scores)
        self.mapping_suggestions.append(
            {
                "source": str(header),
                "suggested_canonical": suggested,
                "confidence": round(float(best_score), 2),
                "accepted": False,
                "meets_threshold": best_score
                >= float(
                    self.codec.mapping.get("matching_engine", {}).get("threshold", 88)
                ),
            }
        )
        return None, float(best_score), "suggestion" if best_score >= 88 else "none"

    def _score_header_row(self, values: list[Any]) -> tuple[int, int]:
        matched = 0
        nonempty = 0
        for value in values:
            if pd.isna(value) or not str(value).strip():
                continue
            nonempty += 1
            canonical, _, kind = self._header_match_without_record(value)
            if canonical and kind in {"exact", "variant"}:
                matched += 1
        return matched, nonempty

    def _header_match_without_record(
        self, header: Any
    ) -> tuple[str | None, float, str]:
        return self.codec.map_header(str(header))

    def _detect_encoding(self, content: bytes) -> str:
        if content.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        if content.startswith((b"\xff\xfe", b"\xfe\xff")):
            return "utf-16"
        match = from_bytes(content).best()
        return match.encoding if match is not None and match.encoding else "utf-8"

    def _detect_csv_header(self, content: bytes, encoding: str) -> int:
        best_index = 0
        best_score = (-1, -1)
        delimiter = self._detect_delimiter(content, encoding)
        for index in range(int(self.settings.get("header_scan_rows", 15))):
            try:
                columns = pd.read_csv(
                    io.BytesIO(content),
                    encoding=encoding,
                    skiprows=index,
                    nrows=0,
                    sep=delimiter,
                ).columns.tolist()
            except (csv.Error, pd.errors.ParserError, UnicodeError, ValueError):
                continue
            score = self._score_header_row(list(columns))
            if score[0] > best_score[0] or (
                score[0] == best_score[0] and score[1] > best_score[1]
            ):
                best_index, best_score = index, score
            if score[0] >= len(self.items) + 2:
                break
        return best_index

    @staticmethod
    def _detect_delimiter(content: bytes, encoding: str) -> str:
        sample = content.decode(encoding, errors="replace")[:8192]
        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except csv.Error:
            return ","

    def _detect_excel_header(self, content: bytes, suffix: str) -> int:
        best_index = 0
        best_score = (-1, -1)
        for index in range(int(self.settings.get("header_scan_rows", 15))):
            try:
                columns = pd.read_excel(
                    io.BytesIO(content),
                    header=index,
                    nrows=0,
                    engine="xlrd" if suffix == ".xls" else "openpyxl",
                ).columns.tolist()
            except (ImportError, ValueError, OSError, KeyError):
                continue
            score = self._score_header_row(list(columns))
            if score[0] > best_score[0] or (
                score[0] == best_score[0] and score[1] > best_score[1]
            ):
                best_index, best_score = index, score
            if score[0] >= len(self.items) + 2:
                break
        return best_index

    def read_input(
        self,
        source: str | Path | bytes | bytearray | pd.DataFrame,
        filename: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Read a supported file or frame, detect its header, and normalize text."""
        before = pd.DataFrame()
        if isinstance(source, pd.DataFrame):
            frame = source.copy()
            source_name = filename or "<dataframe>"
            encoding = "dataframe"
            header_row = 0
        else:
            if isinstance(source, (bytes, bytearray)):
                content = bytes(source)
                source_name = filename or "uploaded.csv"
            else:
                path = Path(source)
                if not path.is_file():
                    raise FileNotFoundError(f"Input file not found: {path}")
                content = path.read_bytes()
                source_name = filename or path.name
            suffix = Path(source_name).suffix.lower()
            if suffix == ".csv":
                encoding = self._detect_encoding(content)
                header_row = self._detect_csv_header(content, encoding)
                delimiter = self._detect_delimiter(content, encoding)
                frame = pd.read_csv(
                    io.BytesIO(content),
                    encoding=encoding,
                    skiprows=header_row,
                    sep=delimiter,
                )
            elif suffix in {".xlsx", ".xls"}:
                encoding = "binary workbook"
                header_row = self._detect_excel_header(content, suffix)
                frame = pd.read_excel(
                    io.BytesIO(content),
                    header=header_row,
                    engine="xlrd" if suffix == ".xls" else "openpyxl",
                )
            else:
                raise ValueError(
                    f"Unsupported input type: {suffix or '<no extension>'}"
                )

        original_rows, original_columns = frame.shape
        frame.columns = [
            self._clean_text(column) if isinstance(column, str) else column
            for column in frame.columns
        ]
        frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            frame[column] = frame[column].map(self._clean_text)
        log = self._log(
            "read_normalize",
            before,
            frame,
            (
                f"Read {source_name}; encoding={encoding}; header_row={header_row}; "
                f"dropped {original_rows - len(frame)} empty rows and "
                f"{original_columns - len(frame.columns)} empty columns; "
                "trimmed spaces and normalized dashes/quotes."
            ),
        )
        return frame.reset_index(drop=True), log

    def map_headers(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Map exact/normalized headers; fuzzy matches remain suggestions only."""
        before = frame.copy()
        self.mapping_suggestions.clear()
        mapping: dict[Any, str] = {}
        self.unknown_columns = []
        for column in frame.columns:
            canonical, _, match_type = self._header_match(column)
            if canonical and match_type in {"exact", "variant"}:
                if canonical not in mapping.values():
                    mapping[column] = canonical
                else:
                    self.warnings.append(
                        f"Multiple input columns map to {canonical}; later duplicate header {column!r} was retained as extra."
                    )
                    self.unknown_columns.append(str(column))
            else:
                self.unknown_columns.append(str(column))

        renamed = frame.rename(columns=mapping)
        for canonical in mapping.values():
            duplicates = [column for column in renamed.columns if column == canonical]
            if len(duplicates) > 1:
                renamed[canonical] = renamed[duplicates].bfill(axis=1).iloc[:, 0]
                renamed = renamed.loc[:, ~renamed.columns.duplicated(keep="first")]
        self.missing_required = [
            item for item in self.items if item not in renamed.columns
        ]
        self._log(
            "map_headers",
            before,
            renamed,
            f"Mapped {len(mapping)} recognized columns; fuzzy matches are suggestions only.",
        )
        if self.unknown_columns:
            self.warnings.append(
                f"{len(self.unknown_columns)} unknown or suggested header(s) were not auto-mapped."
            )
        if self.missing_required:
            self.warnings.append(
                "Missing required questionnaire columns: "
                + ", ".join(self.missing_required)
            )
        mapping_report = {
            "mapped_columns": {str(key): value for key, value in mapping.items()},
            "suggestions": list(self.mapping_suggestions),
            "missing_required": list(self.missing_required),
            "unknown_columns": list(self.unknown_columns),
            "columns_in": len(before.columns),
        }
        mapping_report["log"] = self.logs[-1]
        return renamed, mapping_report

    def _filter_rows(
        self,
        frame: pd.DataFrame,
        column: str,
        rule: str,
        keep_mask: pd.Series,
        training: bool,
        action_text: str,
    ) -> pd.DataFrame:
        before = frame.copy()
        rejected = ~keep_mask.fillna(False)
        for index in frame.index[rejected]:
            self._add_reason(index, rule)
        if training:
            frame = frame.loc[keep_mask.fillna(False)].copy()
        self._log(
            rule,
            before,
            frame,
            f"{action_text}; rejected={int(rejected.sum())}; "
            f"{'dropped' if training else 'retained with reason in scoring mode'}.",
        )
        return frame

    def filter_consent(
        self, frame: pd.DataFrame, mode: str
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        before = frame.copy()
        column = "consent"
        if column not in frame.columns:
            if mode == "training":
                self.warnings.append(
                    "Consent column missing; no consent filter applied."
                )
            self._log(
                "consent_filter",
                before,
                frame,
                "Consent field absent; no rows filtered.",
            )
        else:
            accepted = {
                self.codec.normalize_text(value)
                for value in self.settings.get(
                    "consent_accepted", ["yes", "yes / i agree", "i agree", "true", "1"]
                )
            }
            values = frame[column].map(self.codec.normalize_text)
            frame = self._filter_rows(
                frame,
                column,
                "consent_filter",
                values.isin(accepted),
                mode == "training",
                "Applied configured consent acceptance values",
            )
        return frame, self.logs[-1]

    def filter_attention(
        self, frame: pd.DataFrame, mode: str
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        before = frame.copy()
        column = "attention_check"
        if column not in frame.columns:
            if mode == "training":
                self.warnings.append(
                    "Attention-check column missing; no filter applied."
                )
            self._log(
                "attention_filter",
                before,
                frame,
                "Attention-check field absent; no rows filtered.",
            )
        else:
            accepted = {
                self.codec.normalize_text(value)
                for value in self.settings.get("attention_accepted", ["agree"])
            }
            values = frame[column].map(self.codec.normalize_text)
            frame = self._filter_rows(
                frame,
                column,
                "attention_filter",
                values.isin(accepted),
                mode == "training",
                "Applied configured attention-check acceptance values",
            )
        return frame, self.logs[-1]

    def _encode_item_value(self, item: str, value: Any) -> int | None:
        if pd.isna(value):
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return int(value)
        normalized = self.codec.normalize_text(value)
        try:
            numeric_code = int(normalized)
        except ValueError:
            numeric_code = None
        if numeric_code is not None:
            return numeric_code
        table = self.codec._label_to_code[item]
        if normalized in table:
            return int(table[normalized])
        clean_value = re.sub(r"[^\w\s-]", "", normalized)
        if clean_value in table:
            return int(table[clean_value])
        candidates: list[tuple[str, int]] = []
        for level in self.codec.items[item].get("levels", []):
            candidates.append((str(level["label"]), int(level["code"])))
        for key in ("structural_na", "prefer_not_to_say"):
            special = self.codec.items[item].get(key)
            if special:
                candidates.append((str(special["label"]), int(special["code"])))
        scores: list[tuple[float, int, str]] = []
        try:
            from rapidfuzz import fuzz

            for label, code in candidates:
                scores.append(
                    (
                        float(
                            fuzz.token_sort_ratio(
                                normalized, self.codec.normalize_text(label)
                            )
                        ),
                        code,
                        label,
                    )
                )
        except ImportError:
            for label, code in candidates:
                scores.append(
                    (
                        difflib.SequenceMatcher(
                            None, normalized, self.codec.normalize_text(label)
                        ).ratio()
                        * 100,
                        code,
                        label,
                    )
                )
        scores.sort(reverse=True)
        if not scores:
            return None
        threshold = float(self.settings.get("value_fuzzy_threshold", 90))
        best_score, best_code, _ = scores[0]
        if best_score < threshold:
            return None
        if len(scores) > 1 and scores[1][0] == best_score and scores[1][1] != best_code:
            return None
        return best_code

    def encode_values(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        before = frame.copy()
        invalid: dict[str, int] = {}
        for item in self.items:
            if item not in frame.columns:
                continue
            source_values = frame[item]
            encoded = source_values.map(
                lambda value, item=item: self._encode_item_value(item, value)
            )
            invalid_mask = source_values.notna() & encoded.isna()
            invalid[item] = int(invalid_mask.sum())
            for index in frame.index[invalid_mask]:
                self._add_reason(index, f"unrecognized_value:{item}")
            frame[item] = encoded.astype("Int64")
        for column in ("consent", "attention_check"):
            if column in frame.columns:
                frame[column] = frame[column].map(self._clean_text)
        for outcome in ("attrition_intent_score",):
            if outcome in frame.columns:
                original = frame[outcome]
                frame[outcome] = pd.to_numeric(
                    original.astype(str).str.replace("%", "", regex=False),
                    errors="coerce",
                )
                invalid[outcome] = int((original.notna() & frame[outcome].isna()).sum())
                for index in frame.index[original.notna() & frame[outcome].isna()]:
                    self._add_reason(index, f"unrecognized_value:{outcome}")
        if "at_risk" in frame.columns:
            outcome_meta = self.codec.outcomes.get("at_risk", {})
            labels = {
                self.codec.normalize_text(level["label"]): int(level["code"])
                for level in outcome_meta.get("levels", [])
            }

            def encode_risk(value: Any) -> Any:
                if pd.isna(value):
                    return np.nan
                normalized = self.codec.normalize_text(value)
                if normalized in labels:
                    return labels[normalized]
                if normalized in {"1", "true", "yes", "at risk", "at-risk"}:
                    return 1
                if normalized in {"0", "false", "no", "low risk", "persisting"}:
                    return 0
                return np.nan

            source = frame["at_risk"]
            frame["at_risk"] = source.map(encode_risk).astype("Int64")
            invalid["at_risk"] = int((source.notna() & frame["at_risk"].isna()).sum())
        self._log(
            "encode_values",
            before,
            frame,
            "Encoded item labels to schema codes; values below fuzzy threshold "
            f"{self.settings.get('value_fuzzy_threshold', 90)} became missing. "
            f"Invalid counts: {invalid}.",
        )
        return frame, {"invalid_values": invalid, "log": self.logs[-1]}

    def check_ranges(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        before = frame.copy()
        invalid_counts: dict[str, int] = {}
        for item in self.items:
            if item not in frame.columns:
                continue
            meta = self.codec.items[item]
            valid_codes = {int(level["code"]) for level in meta.get("levels", [])}
            for special_key in ("structural_na", "prefer_not_to_say"):
                special = meta.get(special_key)
                if special:
                    valid_codes.add(int(special["code"]))
            numeric = pd.to_numeric(frame[item], errors="coerce")
            invalid = numeric.notna() & ~numeric.isin(valid_codes)
            invalid_counts[item] = int(invalid.sum())
            for index in frame.index[invalid]:
                self._add_reason(index, f"out_of_range:{item}")
            frame.loc[invalid, item] = pd.NA
        self._log(
            "item_range_checks",
            before,
            frame,
            f"Validated each item against its own schema codes; invalid counts: {invalid_counts}.",
        )
        return frame, {
            "invalid_range_values": invalid_counts,
            "log": self.logs[-1],
        }

    def special_answers(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        before = frame.copy()
        pns_counts: dict[str, int] = {}
        if "part_time_job" in frame.columns and "job_hours" in frame.columns:
            working_without_hours = (
                frame["part_time_job"].gt(0).fillna(False) & frame["job_hours"].isna()
            )
            frame["flag_working_hours_missing"] = working_without_hours
        else:
            frame["flag_working_hours_missing"] = False
        for item in self.items:
            meta = self.codec.items[item]
            pns = meta.get("prefer_not_to_say")
            if item not in frame.columns or not pns:
                continue
            flag = f"pns_{item}"
            mask = frame[item].eq(int(pns["code"])).fillna(False)
            frame[flag] = mask.astype(bool)
            pns_counts[item] = int(mask.sum())
            frame.loc[mask, item] = pd.NA
        commute = ["commute_time", "commute_spend", "transport_mode"]
        available = [
            item for item in commute if self.codec.items[item].get("structural_na")
        ]
        if all(item in frame.columns for item in available):
            sna_mask = pd.DataFrame(
                {
                    item: frame[item].eq(
                        int(self.codec.items[item]["structural_na"]["code"])
                    )
                    for item in available
                }
            )
            frame["is_online"] = sna_mask.all(axis=1)
        else:
            frame["is_online"] = False
        if "part_time_job" in frame.columns and "job_hours" in frame.columns:
            no_job_code = next(
                int(level["code"])
                for level in self.codec.items["part_time_job"]["levels"]
                if self.codec.normalize_text(level["label"]) == "no"
            )
            none_hours_code = next(
                int(level["code"])
                for level in self.codec.items["job_hours"]["levels"]
                if self.codec.normalize_text(level["label"]) == "none"
            )
            no_job = frame["part_time_job"].eq(no_job_code).fillna(False)
            frame["flag_unemployed_with_hours"] = no_job & frame["job_hours"].gt(
                none_hours_code
            ).fillna(False)
            frame.loc[no_job, "job_hours"] = none_hours_code
            working = frame["part_time_job"].notna() & ~no_job
            frame.loc[working & frame["job_hours"].isna(), "job_hours"] = pd.NA
        self._log(
            "special_answers",
            before,
            frame,
            "Converted PNS to missing with flags, retained structural N/A codes, "
            "derived is_online, and applied job-hours gate.",
        )
        return frame, {"prefer_not_to_say": pns_counts, "log": self.logs[-1]}

    def consistency_flags(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        before = frame.copy()
        commute = ["commute_time", "commute_spend", "transport_mode"]
        available = [item for item in commute if item in frame.columns]
        if len(available) == len(commute):
            sna_codes = {
                item: int(self.codec.items[item]["structural_na"]["code"])
                for item in commute
                if self.codec.items[item].get("structural_na")
            }
            sna = pd.DataFrame(
                {
                    item: (
                        frame[item].eq(sna_codes[item]).fillna(False)
                        if item in sna_codes
                        else pd.Series(False, index=frame.index)
                    )
                    for item in commute
                }
            )
            frame["flag_commute_physical_mode"] = (
                sna["commute_time"] & ~sna["transport_mode"]
            )
            frame["flag_commute_partial_na"] = sna.any(axis=1) & ~sna.all(axis=1)
        else:
            frame["flag_commute_physical_mode"] = False
            frame["flag_commute_partial_na"] = False
        if "part_time_job" in frame.columns and "job_hours" in frame.columns:
            no_job = frame["part_time_job"].eq(0).fillna(False)
            if "flag_unemployed_with_hours" not in frame.columns:
                frame["flag_unemployed_with_hours"] = no_job & frame["job_hours"].gt(
                    0
                ).fillna(False)
        elif "flag_unemployed_with_hours" not in frame.columns:
            frame["flag_unemployed_with_hours"] = False
        counts = {
            column: int(frame[column].sum())
            for column in (
                "flag_commute_physical_mode",
                "flag_commute_partial_na",
                "flag_unemployed_with_hours",
            )
        }
        for column, count in counts.items():
            if count:
                for index in frame.index[frame[column]]:
                    self._add_reason(index, column)
        self._log(
            "consistency_flags",
            before,
            frame,
            f"Kept inconsistent rows and added flags; counts: {counts}.",
        )
        counts["log"] = self.logs[-1]
        return frame, counts

    def deduplicate(
        self, frame: pd.DataFrame, mode: str
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        before = frame.copy()
        id_column = "student_id"
        if id_column not in frame.columns:
            log = self._log(
                "deduplicate",
                before,
                frame,
                "No student_id; duplicate-ID rule not applied.",
            )
            return frame, {"duplicates_removed": 0, "log": log}
        duplicates = frame[id_column].notna() & frame[id_column].duplicated(keep="last")
        for index in frame.index[duplicates]:
            self._add_reason(index, "duplicate_student_id")
        if mode == "training":
            frame = frame.loc[~duplicates].copy()
        count = int(duplicates.sum())
        self._log(
            "deduplicate",
            before,
            frame,
            f"Keep-last student_id duplicate rule; duplicates={count}; "
            f"{'removed' if mode == 'training' else 'retained with reason in scoring mode'}.",
        )
        return frame, {"duplicates_removed": count, "log": self.logs[-1]}

    def detect_careless(
        self, frame: pd.DataFrame, mode: str
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        before = frame.copy()
        ordinal = [
            item
            for item in self.items
            if self.codec.items[item].get("type") == "ordinal" and item in frame.columns
        ]
        threshold = float(self.settings.get("straight_liner_fraction", 0.8))
        minimum = math.ceil(
            len(
                [
                    item
                    for item in self.items
                    if self.codec.items[item].get("type") == "ordinal"
                ]
            )
            * threshold
        )
        flagged = pd.Series(False, index=frame.index)
        for index, row in frame[ordinal].iterrows():
            positions: list[float] = []
            for item in ordinal:
                if pd.isna(row[item]):
                    continue
                levels = self.codec.items[item]["levels"]
                codes = [int(level["code"]) for level in levels]
                code = int(row[item])
                if code not in codes:
                    continue
                positions.append((code - min(codes)) / max(1, max(codes) - min(codes)))
            if len(positions) >= minimum and positions:
                _, counts = np.unique(np.round(positions, 8), return_counts=True)
                flagged.at[index] = int(counts.max()) / len(positions) >= threshold
        frame["flag_straight_liner"] = flagged.astype(bool)
        for index in frame.index[flagged]:
            self._add_reason(index, "straight_liner")
        drop = bool(self.settings.get("drop_straight_liners", True))
        if mode == "training" and drop:
            frame = frame.loc[~flagged].copy()
        count = int(flagged.sum())
        self._log(
            "careless_response",
            before,
            frame,
            f"Flagged same relative response position on at least {threshold:.0%} "
            f"of ordinal items; count={count}; "
            f"{'dropped' if mode == 'training' and drop else 'retained'}.",
        )
        return frame, {
            "straight_liners": count,
            "dropped": count if mode == "training" and drop else 0,
            "log": self.logs[-1],
        }

    def missing_data(
        self, frame: pd.DataFrame, mode: str
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        before = frame.copy()
        features = [item for item in self.items if item in frame.columns]
        missingness = {
            item: {
                "missing": int(frame[item].isna().sum()),
                "fraction": float(frame[item].isna().mean()) if len(frame) else 0.0,
            }
            for item in features
        }
        if features:
            row_fraction = frame[features].isna().mean(axis=1)
            threshold = float(self.settings.get("max_missing_feature_fraction", 0.25))
            flagged = row_fraction > threshold
            for index in frame.index[flagged]:
                self._add_reason(index, "more_than_25_percent_features_missing")
            if mode == "training":
                frame = frame.loc[~flagged].copy()
        else:
            flagged = pd.Series(False, index=frame.index)
            threshold = float(self.settings.get("max_missing_feature_fraction", 0.25))
            self.warnings.append("No mapped questionnaire features are available.")
        self._log(
            "missing_data",
            before,
            frame,
            f"Did not impute; rows over {threshold:.0%} missing feature cells="
            f"{int(flagged.sum())}; training drops them and scoring retains with reason.",
        )
        return frame, {
            "missingness": missingness,
            "over_missing_threshold": int(flagged.sum()),
            "dropped": int(flagged.sum()) if mode == "training" else 0,
            "log": self.logs[-1],
        }

    def build_outcomes(
        self, frame: pd.DataFrame, mode: str
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        before = frame.copy()
        threshold = float(self.codec.outcomes.get("at_risk", {}).get("threshold", 60))
        score_present = "attrition_intent_score" in frame.columns
        label_present = "at_risk" in frame.columns
        if score_present:
            score = pd.to_numeric(frame["attrition_intent_score"], errors="coerce")
            invalid = score.notna() & ~score.between(0, 100)
            score.loc[invalid] = np.nan
            for index in frame.index[invalid]:
                self._add_reason(index, "outcome_out_of_range:attrition_intent_score")
            frame["attrition_intent_score"] = score
            derived = score.ge(threshold).astype("Int64")
            if label_present:
                derived = derived.fillna(frame["at_risk"])
            frame["at_risk"] = derived
        elif label_present:
            frame["at_risk"] = pd.to_numeric(frame["at_risk"], errors="coerce").astype(
                "Int64"
            )
        else:
            frame["at_risk"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        if "attrition_intent_score" not in frame.columns:
            frame["attrition_intent_score"] = np.nan
        if mode == "training" and not score_present and not label_present:
            self.warnings.append(
                "Training requires attrition_intent_score or at_risk, but no outcome was mapped."
            )
        outcome_source = (
            "attrition_intent_score"
            if score_present
            else "at_risk" if label_present else None
        )
        self._log(
            "outcome_handling",
            before,
            frame,
            f"Derived at_risk from score threshold {threshold:g} when available; "
            f"outcome source={outcome_source or 'none'}; scoring-only rows retained.",
        )
        return frame, {
            "outcome_source": outcome_source,
            "threshold": threshold,
            "log": self.logs[-1],
        }

    def _make_report(
        self,
        frame: pd.DataFrame,
        mode: str,
        input_rows: int,
        mapping: dict[str, Any],
        stage_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        features = [item for item in self.items if item in frame.columns]
        class_balance: dict[str, Any] = {}
        if "at_risk" in frame and frame["at_risk"].notna().any():
            counts = frame["at_risk"].dropna().astype(int).value_counts().sort_index()
            class_balance = {
                str(int(label)): int(count) for label, count in counts.items()
            }
            positives = int(counts.get(1, 0))
            total = int(counts.sum())
            positive_fraction = positives / total if total else 0.0
        else:
            positive_fraction = None
        distributions = {
            item: {
                str(label): int(count)
                for label, count in frame[item].value_counts(dropna=False).items()
            }
            for item in features
        }
        warnings = list(dict.fromkeys(self.warnings))
        if len(frame) < int(self.settings.get("small_sample_rows", 300)):
            warnings.append(
                f"Sample contains fewer than 300 retained rows ({len(frame)})."
            )
        if positive_fraction is not None and positive_fraction < float(
            self.settings.get("low_at_risk_fraction", 0.15)
        ):
            warnings.append(f"At-risk share is below 15% ({positive_fraction:.1%}).")
        dominant_limit = float(self.settings.get("dominant_item_level_fraction", 0.60))
        dominant = []
        for item in features:
            observed = frame[item].dropna()
            if (
                len(observed)
                and observed.value_counts(normalize=True).max() > dominant_limit
            ):
                dominant.append(item)
        if dominant:
            warnings.append(
                "More than 60% of responses occupy one level for: "
                + ", ".join(dominant)
            )
        traffic = "green" if not warnings else "yellow"
        messages = (
            ["Data quality checks passed with no configured warnings."]
            if not warnings
            else ["Review the warnings before using these data."]
        )
        if self.missing_required:
            traffic = "red"
            messages = [
                "Required questionnaire fields are missing; training or scoring may not be valid."
            ]
        if mode == "training" and not stage_metrics["outcomes"]["outcome_source"]:
            traffic = "red"
            messages = ["Training mode requires an available outcome."]
        if mode == "training" and not class_balance:
            traffic = "red"
            messages = [
                "Training mode requires at least one valid outcome value after cleaning."
            ]
        reasons = {
            reason: sum(
                reason in row_reasons for row_reasons in self.row_reasons.values()
            )
            for reason in sorted(
                {reason for values in self.row_reasons.values() for reason in values}
            )
        }
        return {
            "run_id": "",
            "mode": mode,
            "rows_in": int(input_rows),
            "rows_out": len(frame),
            "columns_in": mapping["columns_in"],
            "columns_out": len(frame.columns),
            "per_rule_counts": reasons,
            "stage_log": list(self.logs),
            "mapping": mapping,
            "missingness": stage_metrics["missing"]["missingness"],
            "class_balance": class_balance,
            "at_risk_fraction": positive_fraction,
            "item_distributions": distributions,
            "consistency_flags": stage_metrics["consistency"],
            "warnings": warnings,
            "traffic_light": {"status": traffic, "messages": messages},
            "row_reasons": {
                str(index): reasons for index, reasons in self.row_reasons.items()
            },
            "no_imputation": True,
        }

    def clean(
        self,
        source: str | Path | bytes | bytearray | pd.DataFrame,
        mode: str = "training",
        filename: str | None = None,
    ) -> CleaningResult:
        """Run all stages and return canonical data with a detailed quality report."""
        if mode not in {"training", "scoring"}:
            raise ValueError("mode must be either 'training' or 'scoring'")
        self.logs = []
        self.warnings = []
        self.row_reasons = {}
        self.mapping_suggestions = []
        self.missing_required = []
        self.unknown_columns = []
        raw, _ = self.read_input(source, filename)
        input_rows = len(raw)
        mapped, mapping = self.map_headers(raw)
        for item in mapping["missing_required"]:
            for index in mapped.index:
                self._add_reason(index, f"missing_required_column:{item}")
        frame, _ = self.filter_consent(mapped, mode)
        frame, _ = self.filter_attention(frame, mode)
        frame, value_metrics = self.encode_values(frame)
        frame, range_metrics = self.check_ranges(frame)
        frame, special_metrics = self.special_answers(frame)
        frame, consistency = self.consistency_flags(frame)
        frame, duplicate_metrics = self.deduplicate(frame, mode)
        frame, careless_metrics = self.detect_careless(frame, mode)
        frame, missing_metrics = self.missing_data(frame, mode)
        frame, outcome_metrics = self.build_outcomes(frame, mode)
        required_outcomes = {"attrition_intent_score", "at_risk"}.intersection(
            mapping["mapped_columns"].values()
        )
        if mode == "training" and not required_outcomes:
            self.warnings.append(
                "Training mode requires at least one configured outcome column."
            )
        if mode == "training" and not frame["at_risk"].notna().any():
            self.warnings.append(
                "Training mode has no valid outcome values after cleaning."
            )
        stage_metrics = {
            "values": value_metrics,
            "ranges": range_metrics,
            "special": special_metrics,
            "consistency": consistency,
            "duplicates": duplicate_metrics,
            "careless": careless_metrics,
            "missing": missing_metrics,
            "outcomes": outcome_metrics,
        }
        if mode == "scoring":
            frame["row_reasons"] = [
                "; ".join(self.row_reasons.get(index, [])) for index in frame.index
            ]
        report = self._make_report(frame, mode, input_rows, mapping, stage_metrics)
        run_id = (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        )
        report["run_id"] = run_id
        if mode == "training" and not required_outcomes:
            report["traffic_light"] = {
                "status": "red",
                "messages": ["Training mode requires at least one outcome column."],
            }
        elif mode == "training" and not frame["at_risk"].notna().any():
            report["traffic_light"] = {
                "status": "red",
                "messages": [
                    "Training mode requires at least one valid outcome value after cleaning."
                ],
            }
        return CleaningResult(data=frame.reset_index(drop=True), report=report)

    def write_result(
        self, result: CleaningResult, output_root: Path | None = None
    ) -> CleaningResult:
        """Persist parquet and JSON quality report beneath the selected project root."""
        root = get_project_root() if output_root is None else Path(output_root)
        run_id = result.report["run_id"]
        parquet_path = root / "data" / "interim" / f"clean_{run_id}.parquet"
        report_path = root / "data" / "interim" / f"clean_{run_id}.json"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        result.data.to_parquet(parquet_path, index=False)
        report_path.write_text(
            json.dumps(result.report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return CleaningResult(
            data=result.data,
            report=result.report,
            parquet_path=parquet_path,
            report_path=report_path,
        )
