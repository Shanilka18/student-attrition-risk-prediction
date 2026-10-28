# 16. Student Scoring Workflow (`/score`)

This document outlines the architecture, data processing pipeline, and user experience of the **Student Scoring Workflow** (`/score`) within the Multimodal Student Engagement Intelligence (MEI) platform (Component ID: `KND_CDAP_IT_05`).

---

## 1. Overview and Design Philosophy

The scoring workflow enables university staff (academic advisors, course coordinators, and student support personnel) to upload student survey responses, validate and audit the data structure, and evaluate engagement risks using the frozen champion model (`knn` / k-Nearest Neighbours, Test F1: `0.8842`, Optimal Threshold: `0.26`).

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   Step 1:       │  ───> │   Step 2:       │  ───> │   Step 3:       │
│   Upload Survey │       │   Check Columns │       │   Cohort Results│
│   or 1 Student  │       │   & Data Audit  │       │   & Trust Drawer│
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

### Core Design Requirements
1. **Zero Data Leakage & Real Artifacts**: All calculations strictly utilize the frozen champion model, trained preprocessing pipeline, and longitudinal feature extractors from the active run (`artifacts/runs/<active>/champion/`).
2. **Flexible Ingestion**: Robust handling of varying header nomenclature, arbitrary column orderings, missing questions, and text response scales.
3. **Institutional Trust & Transparency**: Every calculated score is explained with a point-based waterfall attribution and 4-period longitudinal trajectories.
4. **Data Privacy by Design**: All file processing occurs locally within the institutional deployment; uploaded files and temporary session artifacts are automatically purged after 24 hours.

---

## 2. Stepper Workflow

### Step 1: Upload Survey & Privacy Guarantee
- **File Ingestion**: Accepts `.csv`, `.xlsx`, and `.xls` files up to 20 MB.
- **Resource Downloads**: One-click download links for the canonical Excel template (`/api/template?format=xlsx`), CSV template (`/api/template?format=csv`), and the complete 28-indicator Data Dictionary (`/api/dictionary`).
- **Score an Individual Student**: A dedicated modal enabling staff to enter survey answers directly for an individual student across the 4 semester intervals (7 indicators $\times$ 4 periods = 28 questions) using the exact questionnaire wording and response scales.
- **Privacy Notice**: Clearly states that all uploaded files are processed strictly within the local environment, never shared with third parties, and automatically purged after 24 hours.

### Step 2: Check Columns & Quality Audit
Upon upload, the backend validates the file schema and executes an automated data quality audit (`POST /api/score/validate`):
- **Column Matching Table**: Displays required canonical indicators against detected columns in the uploaded file, accompanied by a match confidence tier:
  - `High Confidence` ($\ge 85\%$ match score): Auto-accepted.
  - `Check this` ($50\%\text{–}84\%$ match score): Requires user confirmation via interactive select dropdowns.
  - `Not Found` ($< 50\%$ match score): User can manually map or leave unmapped if optional.
- **Quality Traffic Light Summary**:
  - `GREEN`: 100% of required survey items mapped cleanly; zero blocking data issues.
  - `AMBER`: High-confidence mapping with non-fatal data issues (e.g. minor missing answers $< 50\%$ imputed with cohort medians).
  - `RED`: Missing required longitudinal survey items or $> 50\%$ incomplete responses.
- **Issue Breakdown & Remediation Actions**:
  - Out-of-range numerical values clipped to valid Likert bounds ($1\text{–}5$).
  - Duplicate student IDs deduplicated by keeping the latest record.
  - Straight-lining responses (zero variance across all 28 items) flagged for academic advisor review.
- **Actions**:
  - *Score Validated Cohort*: Proceeds to Step 3 with rows that can be scored.
  - *Download Problem Rows*: Exports CSV/Excel of invalid rows with specific error reasons (`/api/score/{upload_id}/problems`).
  - *Upload Different File*: Resets workflow.

### Step 3: Cohort Scoring Results & Explainability
Executes inference on validated rows (`POST /api/score/run`) and renders interactive results:
- **Summary Metric Tiles**:
  - Total Students Scored (with scorable completion rate).
  - Low Risk Count & Share (MEI $0\text{–}29$).
  - Moderate Risk Count & Share (MEI $30\text{–}69$).
  - High Risk Count & Share (MEI $70\text{–}100$).
- **Risk Band Distribution Chart**: Apache ECharts bar chart depicting student counts across calibrated risk boundaries with matching design system color tokens (`forest`, `amber`, `crimson`).
- **Scored Students Interactive Table**:
  - Student Identifier (with instant client-side search).
  - Engagement Index (MEI) formatted with bold tabular numeral and proportional progress bar.
  - Risk Band badge (`Low`, `Moderate`, `High`).
  - At-Risk Flag (`Flagged` vs `Not Flagged`).
  - Longitudinal Behavioral Pattern (`Consistent`, `Declining`, `Sporadic`, `Improving`).
  - Engagement Intensity Level (`High`, `Medium`, `Low`).
  - Plain-English Primary Drivers (first driver visible, remaining accessible via `+2` badge).
  - Pagination, sorting, and band/flag filtering.
- **Model Evaluation on Uploaded File (Outcome Check Card)**: If ground-truth attrition labels (`at_risk`) are included in the uploaded file, an evaluation card renders the test confusion matrix, Recall, Precision, F1-score, and AUC-ROC, explicitly labeled as computed solely on the uploaded cohort.
- **Export Options**: Download full results as CSV or formatted Excel (`.xlsx`) containing:
  - *Sheet 1*: `Scored Students` (all IDs, MEI scores, risk bands, flags, patterns, and driver explanations).
  - *Sheet 2*: `Notes` (active champion model name, model ID, category, benchmark score, optimal threshold, generation timestamp, and interpretation guidelines).

---

## 3. Side Drawer: Individual Student Deep-Dive (`ScoreDetailDrawer`)

Clicking any student row opens a detailed slide-out inspection drawer (`GET /api/score/{upload_id}/student/{student_id}`):

```
┌────────────────────────────────────────────────────────┐
│ TEST_STU_0037                       [X] Close          │
│ MULTIMODAL ENGAGEMENT INDEX: 85 / 100   [High Risk]    │
├────────────────────────────────────────────────────────┤
│ 4-Period Trajectory (Weeks 1-4 -> Weeks 13-16)        │
│ [Composite] [LMS] [Study] [Review] [Consistency] ...   │
│   5 ───*                                               │
│   4     \                                              │
│   3      \──*                                          │
│   2          \──*                                      │
│   1              *───*                                 │
├────────────────────────────────────────────────────────┤
│ Waterfall Point Attribution                            │
│ • Cohort Baseline Average:                    24.6 pts │
│ • Low overall Practice & Quizzes:              +27 pts │
│ • Reduced Study Consistency in late periods:   +21 pts │
│ • Declining participation in Class Activity:   +12 pts │
│ = Calculated MEI Score:                         85 pts │
├────────────────────────────────────────────────────────┤
│ Trajectory Pattern: Declining                          │
│ Engagement Intensity: Medium                           │
├────────────────────────────────────────────────────────┤
│ What This Score Does Not Mean:                         │
│ "The score describes study engagement answers. It is   │
│ not a prediction about a person's character and should │
│ be used to start a supportive conversation."           │
└────────────────────────────────────────────────────────┘
```

### Key Drawer Components:
1. **Score & Band Header**: High-visibility MEI score and risk badge.
2. **4-Period Engagement Trajectory**: Interactive ECharts line chart with period markers (Weeks 1–4, Weeks 5–8, Weeks 9–12, Weeks 13–16). Users can toggle between the composite engagement trajectory and individual indicators (LMS access, study hours, review time, consistency, class participation, assignment completion, quiz practice).
3. **Waterfall Point Attribution**: Exact breakdown showing how individual behavioral drivers shift the score relative to the cohort baseline average ($24.6$ pts).
4. **Behavioral Pattern & Intensity**: Plain-language description of the student's study trajectory across the term.
5. **Non-Punitive Institutional Notice**: Explicit framing stating that the index reflects study habits and must only be used to initiate supportive academic conversations, never for disciplinary or punitive actions.

---

## 4. Column Matching and Ingestion Engine

The schema mapper (`src/mei/cleaning/mapper.py`) matches user headers to canonical schema keys using a multi-stage deterministic matching pipeline:

1. **Exact & Normalized String Match**: Strips whitespace, punctuation, accents, and converts to lowercase.
2. **Alias Dictionary Lookup**: Evaluates against predefined domain aliases (e.g. `lms_p1`, `portal_w1`, `study_hours_q1`, `quiz_practice_final`).
3. **Token Cosine Similarity**: Uses TF-IDF character n-gram tokenization to calculate similarity scores against canonical indicator definitions.
4. **Confidence Thresholding**:
   - $\ge 0.85$: High confidence (auto-mapped).
   - $0.50\text{–}0.84$: Medium confidence (flagged for review).
   - $< 0.50$: Low confidence (unmapped).

---

## 5. Performance and Robustness

The scoring engine was subjected to rigorous stress testing across 8 distinct fixtures (`tests/test_scoring_workflow.py`):

| Test Fixture | File Type / Description | Validation Result | Scoring Status | Processing Time |
| :--- | :--- | :--- | :--- | :--- |
| **Fixture 1** | Clean standard sample (200 rows) | 29/29 Mapped | 200 Scored | 0.85s |
| **Fixture 2** | Messy sample with text options & headers | 28/28 Mapped | 150 Scored | 0.92s |
| **Fixture 3** | Extra columns & missing optional items | 28/28 Mapped | 50 Scored | 0.41s |
| **Fixture 4** | Large institutional cohort (20,000 rows) | 29/29 Mapped | 20,000 Scored | 25.10s ($< 60$s SLA) |
| **Fixture 5** | Unicode & Sinhala multilingual headers | 28/28 Mapped | 35 Scored | 0.38s |
| **Fixture 6** | Google Forms export with full questions | 28/28 Mapped | 25 Scored | 0.32s |
| **Fixture 7** | Empty file (0 bytes) | Rejected (400) | Blocked Cleanly | 0.02s |
| **Fixture 8** | Non-spreadsheet file (`.pdf`) | Rejected (400) | Blocked Cleanly | 0.02s |

### High-Throughput Optimizations
- **Chunked Processing**: Cohorts $> 5,000$ rows are scored in memory-efficient vectorized chunks.
- **Vectorized Cleaning**: Replaced iterative row loops with vectorized Pandas `.isna().sum(axis=1)` checks and dictionary mapping caches.
- **Sub-Second Response for Typical Cohorts**: Standard cohorts ($N \le 1,000$) complete validation and full SHAP explanation scoring in $< 1.5$ seconds.

---

## 6. Automated 24-Hour File Lifecycle

To comply with educational data privacy regulations (FERPA, GDPR):
- Uploaded raw files, intermediate clean tables, and scored parquet files are stored under `artifacts/uploads/up_<uuid>/`.
- The `cleanup_old_uploads()` routine automatically purges any upload session directory with modification timestamps older than 24 hours ($86,400$ seconds).
- The cleanup routine runs automatically on every upload request and can be triggered via scheduled maintenance cron.
- The unit test `test_automated_24h_cleanup()` verifies that expired directories are purged while active sessions are preserved.
