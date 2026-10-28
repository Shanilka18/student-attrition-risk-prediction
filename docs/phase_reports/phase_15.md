# Phase 15 Report: Student Scoring Workflow, Column Auto-Validation, and Trust Drawer

**Date**: 2026-10-28  
**Role**: Senior full-stack engineer focused on file-upload workflows and trust in results  
**Active Run ID**: `run_20261026_124514_1407b3`  
**Component ID**: `KND_CDAP_IT_05`  
**Status**: COMPLETED  

---

## 1. Executive Summary

Phase 15 delivered the complete end-to-end **Student Scoring Workflow** (`/score`) enabling university academic advisors and administrators to upload student survey files (`.csv`, `.xlsx`, `.xls` $\le 20$ MB) or enter answers for an individual student, validate column schemas, and execute calibrated Engagement Index (0–100) scoring using the frozen champion model (`knn` / k-Nearest Neighbours, category `classical`, Test F1: `0.8842`, Optimal Threshold: `0.26`).

The workflow is structured as a clear 3-step stepper:
1. **Upload Survey (`/score`)**: Drag-and-drop or file picker with format validation ($\le 20$ MB), one-click download links for the canonical Excel template, CSV template, and 28-indicator Data Dictionary, an interactive "Score an Individual Student" modal (7 questions $\times$ 4 periods with exact questionnaire phrasing and response scales), and a prominent data privacy guarantee backed by automated 24-hour local file purging.
2. **Check Columns & Quality Audit (`POST /api/score/validate`)**: Automatic multi-tier header matching with confidence tiers (`High Confidence` $\ge 85\%$, `Check this` $50\%\text{–}84\%$, `Not Found` $< 50\%$), dropdown overrides, live sample values preview, a data quality traffic light (`GREEN`, `AMBER`, `RED`) tracking scorable counts, missing data, duplicates, and out-of-range values, with one-click export of problem rows (`/api/score/{upload_id}/problems`).
3. **Cohort Scoring Results (`POST /api/score/run`)**: High-level summary tiles, risk band distribution chart (Apache ECharts), an interactive scored students table (ID search, band/flag filters, MEI bar, pattern, level, primary reasons with `+2` badge), an on-upload model evaluation card when outcome labels exist, and 2-sheet formatted Excel/CSV export (`mei_scored_YYYY-MM-DD_<upload_id>.xlsx`).
- **Student Trust Drawer (`ScoreDetailDrawer`)**: Slide-out drawer displaying 4-period longitudinal trajectories across composite engagement and all 7 individual indicators, a point-based waterfall attribution ($E[MEI] + \sum \Delta_{driver} = MEI$) anchored to the cohort baseline average ($24.6$ pts), behavioral pattern/level meanings, and an ethical non-punitive institutional framing notice.

All calculations use real artifacts and real model inference (Global Rule G2). Automated 24-hour retention purging was implemented and verified. All 12 backend workflow tests and 36 frontend Vitest tests pass with zero forbidden word/color/emoji violations.

---

## 2. Key Numbers Read Directly from Output Files

### 2.1 Active Champion Inference Baseline (`champion/model_card.json`, `champion/pipeline.joblib`)
- **Active Champion Model**: `knn` (k-Nearest Neighbours Classifier, `classical` category).
- **Test Set F1-Score**: **0.8842** (Cross-validation F1: **0.7949**).
- **Optimal Decision Threshold ($\tau^*$ )**: **0.2600** (Sensitive screening threshold: **0.2100**).
- **Cohort Baseline Average Engagement Score**: **24.6 points** ($0\text{–}100$ scale).
- **Total Pipeline Input Features**: **100** engineered longitudinal features (28 raw questions, 56 summary statistics/slopes, 16 temporal composite dynamics).

### 2.2 Standard Cohort Scoring (`tests/fixtures/sample_upload_clean.xlsx`)
- **Uploaded Rows**: **200** students.
- **Scorable Rows**: **200** students (**100.0%** completion rate).
- **Column Auto-Mapping**: **29 of 29** columns mapped at **100% confidence** (student ID + 28 longitudinal indicators).
- **Data Quality Audit Traffic Light**: **GREEN** (0 missing responses, 0 out-of-range values, 0 duplicate IDs).
- **Calibrated Risk Distribution**:
  - **Low Risk (0–29)**: **142** students (**71.0%** of cohort).
  - **Moderate Risk (30–69)**: **31** students (**15.5%** of cohort).
  - **High Risk (70–100)**: **27** students (**13.5%** of cohort).
- **Top Scored Student (`TEST_STU_0037`)**:
  - Calculated MEI Score: **85 / 100** (`High Risk`, `Flagged`).
  - Longitudinal Pattern: **Declining** (High initial pace tapering to low engagement in P3–P4).
  - Engagement Intensity: **Medium**.
  - Waterfall Attribution Breakdown:
    - Cohort Baseline Average: **24.6 pts**
    - Low overall Practice & Quizzes: **+27 pts**
    - Reduced Study Consistency in late periods: **+21 pts**
    - Declining participation in Class Activity: **+12 pts**
    - Net Total MEI: **85 pts**.

### 2.3 Large-Scale Cohort Stress Benchmark (`tests/fixtures/large_20k_rows.csv`)
- **Cohort Size**: **20,000** student records.
- **Execution Processing Time**: **25.10 seconds** (Strictly within the $< 60$ second project requirement).
- **Vectorized Cleaning Throughput**: Chunked inference at **5,000 rows/batch** with cached string-to-numeric mapping.
- **Scored Count**: **20,000** students scored successfully without memory exhaustion or process failure.

### 2.4 Multi-Language & Messy Data Ingestion Benchmarks
- **Messy Sample (`sample_upload_messy.csv`)**: **150** students scored; text-to-numeric Likert mapping parsed successfully; out-of-range values clipped to $[1, 5]$; minor missing values median-imputed.
- **Multilingual / Unicode Sample (`sample_upload_sinhala.csv`)**: **35** students scored; Sinhala and Unicode column headers resolved cleanly with $100\%$ schema mapping accuracy.
- **Google Forms Export (`sample_upload_google_forms.csv`)**: **25** students scored; verbose question strings (e.g. *"How many hours per week did you dedicate to studying..."*) resolved via token cosine similarity.
- **Empty / Corrupt File Rejections**: 0-byte file (`empty_file.csv`) and PDF file (`invalid_file.pdf`) immediately rejected with structured HTTP 400 client errors ($< 0.03$s).

### 2.5 Automated 24-Hour Privacy Purge Benchmark
- **Cleanup Verification (`cleanup_old_uploads`)**: Purged **1** expired upload session directory ($> 24$h old); preserved **1** active upload session directory ($< 24$h old).

---

## 3. Files Created or Modified

### 3.1 Backend API & Ingestion Engine
- **`src/mei/api/routes/scoring.py`** (MODIFIED):
  - Implemented `validate_upload` endpoint returning `column_matches`, `quality_summary`, `fixable_problems`, and saving `problem_rows.csv`/`json`.
  - Added `download_problem_rows` (`GET /api/score/{upload_id}/problems`) supporting `.csv` and `.xlsx` downloads.
  - Optimized `run_scoring` (`POST /api/score/run`) with vectorized feature processing, chunking ($5,000$ rows), risk distributions, and optional outcome verification metrics (`ScoringOutcomeCheck`).
  - Added `get_scoring_session` (`GET /api/score/{upload_id}/session`) for deep-linking and state restoration.
  - Implemented `get_student_detail` (`GET /api/score/{upload_id}/student/{student_id}`) returning 4-period trajectories (composite and 7 indicators), waterfall points, behavioral patterns, and ethical guardrail notice.
  - Updated `download_scoring_results` (`GET /api/score/{upload_id}/download`) to produce 2-sheet formatted Excel workbooks (`Scored Students` + `Notes`) and date-stamped filenames `mei_scored_YYYY-MM-DD_<upload_id>.xlsx`.
  - Implemented `score_single_student` (`POST /api/score/single`) for direct web-form scoring.
  - Implemented `cleanup_old_uploads` for automated 24-hour file retention purging.
- **`src/mei/api/models.py`** (MODIFIED): Added Pydantic schema models for `ValidateUploadResponse`, `ColumnMatchDetail`, `DataQualitySummary`, `FixableProblemItem`, `ScoringOutcomeCheck`, `StudentDetailResponse`, `SingleScoreRequest`, and `SingleScoreResponse`.
- **`src/mei/cleaning/filters.py`** (MODIFIED): Vectorized straight-lining detection using NumPy array operations.
- **`src/mei/cleaning/encoder.py`** (MODIFIED): Vectorized text-to-numeric Likert mapping with cached dictionary lookup.

### 3.2 Frontend Architecture & UI Components
- **`frontend/src/api/types.ts`** (MODIFIED): Added TypeScript interfaces for `ValidateUploadResponse`, `ColumnMatchDetail`, `DataQualitySummary`, `FixableProblemItem`, `ScoringOutcomeCheck`, `StudentDetailResponse`, and single student payloads.
- **`frontend/src/api/queries.ts`** (MODIFIED): Added React Query hooks: `useValidateUpload`, `useRunScoring`, `useScoringResults`, `useStudentDetail`, `useScoreSingleStudent`, and file download helper functions.
- **`frontend/src/components/ui/Badge.tsx`** (NEW): Reusable badge primitive supporting `forest`, `amber`, `crimson`, `paper`, and `outline` variants.
- **`frontend/src/components/score/ScoreUploader.tsx`** (NEW): Step 1 drag-and-drop file uploader with size validation, resource download links, privacy card, and "Score an Individual Student" launch button.
- **`frontend/src/components/score/SingleScoreModal.tsx`** (NEW): Full interactive form for scoring a single student across 4 semester intervals with exact Likert radio buttons, quick profile presets, and live waterfall explanation.
- **`frontend/src/components/score/ColumnCheckStep.tsx`** (NEW): Step 2 column matching table with confidence indicators, select dropdown overrides, live sample values, data quality traffic light, and problem row downloads.
- **`frontend/src/components/score/ScoreResultsStep.tsx`** (NEW): Step 3 cohort results page with summary tiles, ECharts risk band distribution, filterable scored students table, outcome check card, and export controls.
- **`frontend/src/components/score/ScoreDetailDrawer.tsx`** (NEW): Slide-out drawer with 4-period trajectory chart (composite and 7 individual indicators), point-based waterfall attribution breakdown, pattern description, and non-punitive institutional framing.
- **`frontend/src/pages/ScorePage.tsx`** (MODIFIED): Stepper container orchestrating transitions between Steps 1, 2, and 3, URL query parameter state restoration (`upload_id`, `step`, `single`), and session recovery.
- **`frontend/src/__tests__/column-matching.test.ts`** (NEW): Vitest test suite for column matching algorithms, confidence categorization, and dirty column name resolution.

### 3.3 Test Suites & Verification Scripts
- **`tests/test_scoring_workflow.py`** (NEW): 12-test comprehensive backend test suite covering all 8 fixtures, chunking, Excel workbook integrity, student detail trajectories, single student scoring, and automated 24-hour cleanup.
- **`scripts/generate_fixtures.py`** (NEW): Generator for the 8 test fixtures in `tests/fixtures/`.
- **`scripts/capture_phase15_screenshots.py`** (NEW): Automated screenshot capture script using headless Chrome against live FastAPI uvicorn server.
- **`docs/figures/ui_score_upload.png`** (NEW): Verified 1440px screenshot of Step 1.
- **`docs/figures/ui_score_single_student.png`** (NEW): Verified 1440px screenshot of Single Student Scoring Modal.
- **`docs/figures/ui_score_check_columns.png`** (NEW): Verified 1440px screenshot of Step 2 Column Check & Quality Audit.
- **`docs/figures/ui_score_results.png`** (NEW): Verified 1440px screenshot of Step 3 Cohort Results.
- **`docs/figures/ui_score_drawer.png`** (NEW): Verified 1440px screenshot of Student Trust Drawer with Trajectory & Waterfall.
- **`docs/16_scoring_workflow.md`** (NEW): Comprehensive user guide and technical architecture specification.
- **`docs/DECISIONS.md`** (MODIFIED): Appended architectural decisions D-059, D-060, and D-061.

---

## 4. Checks Executed with PASS/FAIL Status

| Check Description | Target / Requirement | Result | Status |
| :--- | :--- | :--- | :--- |
| **Python Scoring Workflow Tests** | `pytest tests/test_scoring_workflow.py` | 12 passed in 48.75s | **PASS** |
| **Python Smoke Test Suite** | `pytest tests/test_smoke.py` | 35 passed in 26.11s | **PASS** |
| **Frontend Unit Tests** | `vitest run` in `frontend/` | 8 test files, 36 tests passed in 13.03s | **PASS** |
| **Frontend Production Build** | `npm run build` (`tsc && vite build`) | Built cleanly in 24.66s, 0 errors | **PASS** |
| **Forbidden Words & Colors Audit** | `node frontend/scripts/check-forbidden.cjs` | 63 files scanned, 0 violations | **PASS** |
| **Clean Sample Ingestion** | Fixture 1 (200 rows) | 29/29 mapped, 200 scored | **PASS** |
| **Messy Sample Ingestion** | Fixture 2 (150 rows) | Text-to-numeric resolved, 150 scored | **PASS** |
| **Extra & Missing Columns** | Fixture 3 (50 rows) | Extra ignored, missing handled | **PASS** |
| **Large Cohort SLA (20k rows)** | Fixture 4 (20,000 rows scored in $< 60$s) | Completed in 25.10s | **PASS** |
| **Unicode / Sinhala Ingestion** | Fixture 5 (35 rows with Sinhala headers) | 28/28 mapped, 35 scored | **PASS** |
| **Google Forms Ingestion** | Fixture 6 (25 rows with question headers) | 28/28 mapped, 25 scored | **PASS** |
| **Empty File Rejection** | Fixture 7 (0-byte file) | Structured 400 error | **PASS** |
| **Invalid Format Rejection** | Fixture 8 (non-spreadsheet `.pdf`) | Structured 400 error | **PASS** |
| **2-Sheet Excel Export Integrity** | OpenPyXL validation of `Scored Students` & `Notes` | 2 sheets verified, metadata intact | **PASS** |
| **Student Detail & Waterfall Sum** | `GET /api/score/.../student/...` | Point delta sum matches final MEI | **PASS** |
| **Single Student Direct Scoring** | `POST /api/score/single` | 28 questions evaluated in 0.12s | **PASS** |
| **Automated 24h Retention Purge** | `cleanup_old_uploads()` | Purged $> 24$h, retained $< 24$h | **PASS** |
| **UI Screenshot Inspection** | 5 high-resolution screenshots visually checked | All components rendered cleanly | **PASS** |

---

## 5. Risks or Decisions Logged

1. **D-059: Multi-Tier Column Auto-Mapping and Data Quality Traffic Light Auditing**:
   - Implemented exact, alias, and token cosine similarity matching with 3 confidence tiers. High confidence ($\ge 85\%$) is accepted automatically; lower confidence requires explicit user confirmation via dropdowns.
   - Enforced quality traffic light auditing (`GREEN`, `AMBER`, `RED`) with automatic bounding of out-of-range ratings and duplicate deduplication.
2. **D-060: Vectorized High-Throughput Cohort Scoring with Longitudinal Trajectories and Waterfall Point Attribution**:
   - Optimized Pandas feature engineering with vectorized operations and chunked inference ($5,000$ rows), achieving $25.1$s processing time on $20,000$ rows.
   - Grounded all student explanations in additive point contributions anchored to the active run's cohort baseline ($24.6$ pts).
3. **D-061: Trust & Privacy Enforcement with Automated 24-Hour File Purging and Non-Punitive Educational Guardrails**:
   - Guaranteed local ephemeral data processing with automated 24-hour file cleanup (`cleanup_old_uploads`).
   - Integrated prominent ethical guardrail messaging into the student detail drawer to prevent punitive misinterpretation of study engagement scores.
