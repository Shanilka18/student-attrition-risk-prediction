# Phase 3 Report — Reusable Survey Data Cleaning and Quality Assurance

## 1. What Was Done
- **10-Stage Reusable Cleaning Module**: Built `src/mei/cleaning/` implementing the complete, schema-driven survey data cleaning sequence required by §5 of `PROJECT_BRIEF.md`:
  1. `reader.py` (Stage 1): Ingestion of CSV, XLSX, and XLS formats with automatic encoding detection (`utf-8`, `utf-8-sig`, `cp1252`, `latin-1`), dropping fully empty rows/columns, whitespace stripping, and normalization of typographical curly quotes and en/em-dashes.
  2. `mapper.py` (Stage 2): Schema-driven header mapping from `config/mapping.yaml` (exact, case/punctuation normalized, and RapidFuzz suggestions with confidence $\ge 88.0\%$; suggestions are returned to the caller for UI auditing and never silently auto-accepted).
  3. `filters.py` (Stages 3, 6, 7, 8):
     - Stage 3: Screening of informed consent (`consent == 'Yes'`) and embedded trap attention check (`attention_check == 'Sometimes'`).
     - Stage 6: Duplicate submission resolution (sorting by timestamp and keeping the most recent submission per student ID).
     - Stage 7: Two-tiered straight-lining careless response detection (total response standard deviation $\sigma = 0$, or zero variance across $\ge 6$ of 7 indicators across all 4 periods).
     - Stage 8: Missing data policy (filtering rows with $> 25\%$ missing feature cells; strictly leaving remaining $\le 25\%$ missing cells as `NaN` without imputation to avoid target and validation leakage per Global Rule G3).
  4. `encoder.py` (Stages 4, 5): Typo-tolerant codec conversion using RapidFuzz ($\ge 90.0\%$ threshold) mapping natural language survey labels to ordinal integer codes $1..5$, with strict range validation.
  5. `outcomes.py` (Stage 9): Validation of outcome items $A_1..A_3$, derivation of composite `attrition_intent_score` ($0..100$) and binary `at_risk` indicator (threshold $\ge 60.0$) using `config/labels.yaml`, and class balance auditing.
  6. `pipeline.py` (Stage 10 & Orchestration): Orchestrates the 10 stages, serializes the cleaned dataset to `data/interim/clean_<run_id>.parquet` using `fastparquet`, and compiles a comprehensive JSON Data Quality Report (`data_quality_report.json`) with an automated traffic-light status system (**GREEN**, **AMBER**, **RED**).
- **Dual-Mode Contract Implemented**:
  - `training` mode: Requires outcome columns, enforces strict row drops for consent, attention check, straight-lining, and excessive missingness, returning a clean, leak-free cohort for model training.
  - `scoring` mode: Outcomes are optional; never silently drops uploaded student rows. Non-scorable rows are identified with explicit human-readable rejection reasons (e.g. `straight_lining`, `excessive_missing_features`, `attention_check_failed`) in report metadata so advisors receive complete transparency.
- **CLI Clean Command**: Extended `src/mei/cli.py` with `python -m mei.cli clean --input <path> --mode <training|scoring> --run-id <id>`, producing Rich formatted summaries, stage-by-stage rules logs, mapping suggestions, and non-scorable record audits.
- **Compatibility & Serialization**: Resolved NumPy 1.26.4 ABI incompatibility with PyArrow 18.0 by pinning `fastparquet==2024.11.0` and `cramjam==2.9.1` in `requirements.txt`.
- **Comprehensive Quality Assurance**: Implemented 12 unit and integration tests in `tests/test_cleaning.py` achieving **91% test coverage** on `src/mei/cleaning/`. Updated `tests/test_smoke.py`, achieving 35 passing tests across the entire workspace test suite. Verified 100% compliance with `ruff` and `ruff format`.
- **Documentation & Decisions**: Authored `docs/04_cleaning.md` adhering to Global Rule G8 (Mermaid flowchart, rules table, sample data counts, common mistakes, viva explanation, ELI5). Logged architectural decisions D-013 through D-016 in `docs/DECISIONS.md`.
- **Git Commit Withheld**: In strict compliance with user instructions, no git commits were executed.

---

## 2. Files Created and Modified
- `src/mei/cleaning/__init__.py` *(Created)*: Clean module exports.
- `src/mei/cleaning/models.py` *(Created)*: Pydantic models for data quality reports, rule logs, row details, and traffic-light status.
- `src/mei/cleaning/reader.py` *(Created)*: Multi-format file reader with encoding fallback and structural normalization.
- `src/mei/cleaning/mapper.py` *(Created)*: Schema-driven header mapper with exact, normalized, and fuzzy suggestion matching.
- `src/mei/cleaning/filters.py` *(Created)*: Consent, attention check, duplicate resolution, straight-lining, and missingness policy filters.
- `src/mei/cleaning/encoder.py` *(Created)*: Typo-tolerant codec encoder ($\ge 90\%$) and range checker ($[1, 5]$).
- `src/mei/cleaning/outcomes.py` *(Created)*: Outcome validation, score derivation, and class balance audit.
- `src/mei/cleaning/pipeline.py` *(Created)*: End-to-end 10-stage cleaning orchestrator and report generator.
- `src/mei/cli.py` *(Modified)*: Implemented `clean` command with Rich tables.
- `requirements.txt` *(Modified)*: Pinned `fastparquet==2024.11.0` and `cramjam==2.9.1`.
- `tests/test_cleaning.py` *(Created)*: 12 comprehensive unit and integration tests for cleaning rules.
- `tests/test_smoke.py` *(Modified)*: Removed `clean` stub and added end-to-end CLI tests for training and scoring modes.
- `docs/04_cleaning.md` *(Created)*: G8-compliant technical documentation for data cleaning and quality assurance.
- `docs/DECISIONS.md` *(Modified)*: Logged architectural decisions D-013 through D-016.
- `docs/phase_reports/phase_03.md` *(Created)*: Phase 3 completion report.

---

## 3. Key Numbers Read Directly from Output Files
*Source: `artifacts/runs/run_20261026_170216_8690f5/data_quality_report.json` and CLI run output on `data/raw/survey_raw_sample.csv` ($N = 2412$)*:
- **Raw Submissions In**: `2,412` rows
- **Clean Submissions Out**: `2,184` rows
- **Cohort Retention Rate**: `90.55%`
- **Traffic-Light Status**: **GREEN**
- **Non-Consent Submissions Dropped (R03A)**: `22` rows
- **Inattentive Submissions Dropped (R03B)**: `61` rows
- **Typographical Label Errors Corrected (R04)**: `41` cells (across 68,988 total cells)
- **Unrecognizable Text Entries**: `0` cells
- **Out-of-Range Codes Found (R05)**: `0` cells
- **Duplicate Submissions Resolved (R06)**: `11` earlier submissions dropped (retaining last by timestamp)
- **Straight-Lining Responses Dropped (R07)**: `89` rows
- **Excessive Missingness Dropped (R08)**: `45` rows ($> 25\%$ missing feature cells)
- **Remaining Missing Feature Cells (Unimputed)**: `3,053` cells out of 62,412 (`4.89%` overall feature missingness)
  - Period 1 Missingness: `4.26%` (664 cells / 15,603)
  - Period 2 Missingness: `4.22%` (659 cells / 15,603)
  - Period 3 Missingness: `5.51%` (860 cells / 15,603)
  - Period 4 Missingness: `5.58%` (870 cells / 15,603)
- **Labelled Students (R09)**: `2,184`
- **At-Risk Target Prevalence**: `25.32%` (553 students at-risk, 1,631 students not-at-risk, threshold $\ge 60.0$)
- **Scoring Mode Evaluation on Messy Upload** (`data/samples/sample_upload_messy.csv`):
  - Rows In/Out: `200` / `200` (100% retained, zero silent drops)
  - Unknown Extra Column Detected: `campus_branch` (correctly audited without crashing)
  - Non-Scorable Records Flagged: `5` rows (rejection reason: `straight_lining`)
  - Status: **AMBER** (sample size $< 300$, unknown extra column)
- **Cleaning Test Suite Coverage**: `91%` (525 / 576 statements covered, exceeding $\ge 85\%$ requirement)
- **Full Test Suite Status**: `35 passed in 43.51s`

---

## 4. Checks Run with PASS/FAIL Status
| Check | Command / Target | Status | Details |
|---|---|---|---|
| Ingestion & Encoding | `read_survey_file` | **PASS** | Auto-detects UTF-8, strips whitespace, normalizes curly quotes and dashes |
| Header Mapping | `map_headers` | **PASS** | Mapped 35/35 canonical columns; fuzzy suggestions $\ge 88\%$ returned without auto-accept |
| Consent Filtering | `filter_consent` | **PASS** | 22 unconsented rows dropped in training mode |
| Attention Check | `filter_attention_check` | **PASS** | 61 failed trap rows dropped in training mode |
| Codec Fuzzy Typo Tolerance | `encode_and_validate_indicators` | **PASS** | 41 typos corrected via $\ge 90\%$ similarity matching; unmapped $\to \text{NaN}$ |
| Range Validation | `encode_and_validate_indicators` | **PASS** | Enforces ordinal $[1, 5]$ scale across all items |
| Duplicate Resolution | `resolve_duplicates` | **PASS** | Keeps last valid submission by parsed datetime timestamp |
| Straight-Lining Screening | `detect_careless_responding` | **PASS** | 89 straight-liners flagged and dropped in training mode |
| Zero Imputation Policy (G3) | `apply_missingness_policy` | **PASS** | $>25\%$ missing dropped; remaining 4.89% NaNs strictly preserved |
| Outcome Derivation | `process_outcomes` | **PASS** | Derives `attrition_intent_score` ($0..100$) and `at_risk` ($\ge 60$); class balance 25.32% |
| CLI Clean Command (Training) | `python -m mei.cli clean ... --mode training` | **PASS** | Returns clean dataframe, saves parquet + report, status GREEN |
| CLI Clean Command (Scoring) | `python -m mei.cli clean ... --mode scoring` | **PASS** | Retains all 200 rows, audits 5 non-scorable rows, status AMBER |
| Test Coverage Requirement | `pytest --cov=src/mei/cleaning` | **PASS** | **91% test coverage** on `src/mei/cleaning/` (exceeds 85% bar) |
| Full Workspace Pytest Suite | `pytest tests/` | **PASS** | 35 passed in 43.51s |
| Linter Verification | `ruff check src tests` | **PASS** | All checks passed, 0 errors |
| Formatter Verification | `ruff format --check src tests` | **PASS** | 29 files already formatted |

---

## 5. Risks and Architectural Decisions Logged
- **Decision D-013**: Fastparquet Backend Selection for NumPy 1.26.4 Compatibility. Adopted `fastparquet` pinned at `2024.11.0` with `cramjam` `2.9.1` to eliminate PyArrow ABI incompatibilities on NumPy 1.26.4.
- **Decision D-014**: Non-Imputation Policy in Cleaning Pipeline (Strict Rule G3 Compliance). Retained remaining 4.89% missing feature cells as `NaN` in `clean_<run_id>.parquet`. Imputation transformers must be fitted strictly inside downstream cross-validation pipelines to avoid validation leakage.
- **Decision D-015**: Dual-Mode Contract: Rejection Reasons for Scoring vs. Row Removal for Training. Training mode drops invalid rows; scoring mode audits and retains all uploaded records while flagging non-scorable entries with explicit reasons.
- **Decision D-016**: Two-Tiered Careless Responding Detection (Straight-Lining Heuristics). Flagged straight-liners using global zero variance ($\sigma = 0$) and indicator-specific invariance ($\ge 6/7$ indicators invariant across all 4 periods).
- **Git Commit Status**: Per explicit user instruction, no git commits have been executed.
