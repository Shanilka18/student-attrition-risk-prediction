# Phase 1 Report — Machine-Readable Questionnaire Schema, Codec, Mapping, and Templates

## 1. What Was Done
- **Canonical Schema Engineering**: Created `config/schema.yaml` defining all 35 survey columns with canonical names, data types, permitted ordinal codes (1–5), exact human-readable text labels, functional roles (`id`, `meta`, `consent`, `feature`, `attention_check`, `outcome`), and temporal period markers (P1–P4).
- **Google Forms Header Mapping & Fuzzy Fallback**: Created `config/mapping.yaml` listing accepted header variations (realistic Google Forms long prompts with grid row brackets) for every canonical column. Implemented RapidFuzz matching configuration with an 88.0% similarity threshold restricted strictly to suggestion mode (`auto_accept: false`), ensuring human oversight for unmapped headers.
- **Attrition Target and Band Definitions**: Created `config/labels.yaml` formalizing the attrition intent score formula, binary `at_risk` threshold (60.0), and 3 MEI risk bands (0–34 Low, 35–64 Moderate, 65–100 High).
- **Schema Validation & Codec Implementation**:
  - Implemented `src/mei/io/schema.py` with Pydantic validation models (`ColumnSchema`, `SurveySchema`, `MappingConfig`, `LabelsConfig`, and `validate_all_configs()`).
  - Implemented `src/mei/io/codec.py` with `SurveyCodec`, providing bidirectional translation between ordinal integer codes and survey labels with normalization tolerant of case differences, extra whitespace, and unicode dashes (en-dash, em-dash, minus).
- **Survey Documentation & Academic Grounding**:
  - Authored `docs/questionnaire_engagement.md` containing the form-ready questionnaire layout: informed consent, period definitions, 7 indicator grids, attention check, 3 outcome statements, estimated completion time (4–6 min), and conceptual grounding paragraphs per indicator using standard "citation to be added by author" placeholders.
  - Authored `docs/02_data_design.md` detailing the complete data dictionary table, worked label calculation examples, and research limitations (retrospective recall bias, self-report proxy validity).
- **Survey Templates Generation**: Generated `data/templates/survey_template.csv` and `data/templates/survey_template.xlsx` (with "Survey Data" headers and "Data Dictionary" sheet) using `openpyxl`.
- **CLI Setup Integration**: Updated `python -m mei.cli setup` to execute comprehensive schema/mapping/labels validation and output an ASCII-safe Rich table summarizing column distribution across roles.
- **Comprehensive Test Suite**: Implemented 9 unit tests in `tests/test_schema.py` and updated `tests/test_smoke.py`, testing configuration integrity, codec round-trips, tolerance to unicode punctuation, rapidfuzz suggestion behavior, template structure, and strict target leakage prevention (Global Rule G3).
- **Git Commit Restraint**: Adhered strictly to user instruction to withhold git commits until explicit authorization is provided.

---

## 2. Files Created and Modified
- `config/schema.yaml`: Canonical machine-readable 35-column questionnaire schema.
- `config/mapping.yaml`: Google Forms header mapping table and RapidFuzz suggestion config.
- `config/labels.yaml`: Regression formula, classification threshold (60.0), and MEI risk bands.
- `src/mei/io/schema.py`: Pydantic validation models and cross-configuration validator.
- `src/mei/io/codec.py`: Bidirectional survey codec with resilient normalization.
- `src/mei/io/__init__.py`: Package exports for schema and codec interfaces.
- `src/mei/cli.py`: Integrated `setup` command with Rich summary table output.
- `docs/questionnaire_engagement.md`: Complete form-ready Google Forms questionnaire markdown.
- `docs/02_data_design.md`: Data dictionary, worked label calculations, and limitations.
- `docs/DECISIONS.md`: Logged decisions D-005, D-006, D-007, and D-008.
- `data/templates/survey_template.csv`: Canonical 35-column CSV template.
- `data/templates/survey_template.xlsx`: Two-sheet Excel workbook ("Survey Data" + "Data Dictionary").
- `tests/test_schema.py`: 9 unit tests for schema, codec, mapping, and templates.
- `tests/test_smoke.py`: Updated CLI smoke tests for setup and stub commands.
- `docs/phase_reports/phase_01.md`: This Phase 1 completion report.

---

## 3. Key Numbers Read Directly from Output Files
- **Total canonical columns in schema**: `35` (read from `config/schema.yaml`)
- **Temporal behavioral features**: `28` ($7 \text{ indicators} \times 4 \text{ periods}$)
- **Attrition intent outcome items**: `3` (`intent_leave`, `intent_defer`, `intent_not_continue`)
- **Identifier, screener, and consent items**: `4` (`student_id`, `timestamp`, `consent`, `attention_check`)
- **Mapping coverage**: `35 / 35` canonical columns covered in `config/mapping.yaml`
- **RapidFuzz similarity threshold**: `88.0%` (suggestion only, `auto_accept: false`)
- **Classification threshold (`at_risk = 1`)**: `60.0` points (read from `config/labels.yaml`)
- **MEI risk bands**: `3` bands (`Low: [0, 34]`, `Moderate: [35, 64]`, `High: [65, 100]`)
- **Estimated survey duration**: `4 to 6 minutes` (read from `docs/questionnaire_engagement.md`)
- **CSV template header count**: `35` columns (`data/templates/survey_template.csv`)
- **Excel template sheets and dimensions**:
  - Sheet `Survey Data`: `0` rows, `35` columns
  - Sheet `Data Dictionary`: `35` rows, `7` metadata columns
- **Test suite results**: `14 passed in 4.31s` (9 schema tests + 5 smoke/CLI tests)
- **Linter status**: `0 errors` (`ruff check src tests` returned `All checks passed!`)

---

## 4. Checks Run with PASS/FAIL Status
| Check | Command / Target | Status | Details |
|---|---|---|---|
| Schema Validation | `python -m mei.cli setup` | **PASS** | Validated schema, mapping, labels; printed 35-col summary table |
| Test Suite | `pytest tests/` | **PASS** | 14 passed in 4.31s |
| Linter | `ruff check src tests` | **PASS** | Clean pass (0 warnings, 0 errors) |
| Formatter | `black --check ...` | **PASS** | All source and test files adhere to Black standard |
| Target Leakage Guard (G3) | `test_no_target_leakage_roles` | **PASS** | Outcome items $A_1\text{–}A_3$ strictly excluded from features |
| Codec Round-Trip | `test_codec_round_trip` | **PASS** | 100% round-trip fidelity between codes and text labels |
| Codec Normalization | `test_codec_tolerance_variations` | **PASS** | Correctly resolved case variations, spacing, and en-dashes |
| Fuzzy Mapping Suggestions | `test_fuzzy_mapping_suggestion_behavior` | **PASS** | Generated suggestions for noisy headers; `auto_accept` is `False` |
| Template Integrity | Programmatic read via `pandas`/`openpyxl` | **PASS** | 35 headers in CSV; 2 sheets in Excel with 35 data dictionary rows |
| Questionnaire Completeness | `docs/questionnaire_engagement.md` | **PASS** | Contains all 7 grids, 4 periods, screener, and outcomes |

---

## 5. Risks and Architectural Decisions Logged
- **Decision D-005**: Established canonical 35-column questionnaire schema and strict role assignments.
- **Decision D-006**: Implemented RapidFuzz header matching in suggestion-only mode (threshold 88.0%) with mandatory human review.
- **Decision D-007**: Introduced resilient label normalization in `SurveyCodec` to prevent dropouts from unicode dash and whitespace mismatches.
- **Decision D-008**: Adopted ASCII-safe symbols in console tables for Windows `cp1252` compatibility.
- **Risk Noted**: Questionnaire reliance on retrospective recall bias and self-reported intention as proxy for attrition (documented in `docs/02_data_design.md` per Global Rule G8).
