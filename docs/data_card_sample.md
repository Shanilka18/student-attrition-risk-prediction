# Dataset Card: MEI Synthetic Survey Sample Data

## 1. Dataset Description
- **Dataset Name**: MEI Synthetic Survey Sample Data (`data_source: sample`)
- **Version**: 1.0.0
- **Creation Date**: 2026-10-26
- **Project Component**: Multimodal Student Engagement Intelligence (MEI), CDAP Component 2
- **Data Modality**: Tabular questionnaire responses (Google Forms raw export simulation)
- **Primary Files**:
  - `data/raw/survey_raw_sample.csv` & `data/raw/survey_raw_sample.xlsx` (2412 rows, 35 columns)
  - `data/raw/ground_truth_hidden.csv` (2400 rows, 9 columns)
  - `data/samples/sample_upload_clean.xlsx` (200 rows, 29 columns)
  - `data/samples/sample_upload_messy.csv` (200 rows, 30 columns)

---

## 2. Intended Use and Scope
- **Primary Use**: End-to-end benchmarking, verification, and automated testing of the MEI data cleaning, feature engineering, engagement state modeling, supervised classification, and explainability pipelines.
- **Out of Scope**: Clinical psychological assessment, real-world punitive student attrition decisions, or commercial deployment.

---

## 3. Dataset Composition and Variables
The main dataset simulates an anonymous Google Forms export containing 35 canonical columns:
- **Identifier**: `student_id` (e.g. `STU_0001` .. `STU_2400`).
- **Metadata**: `timestamp` (spread over 14 calendar days).
- **Informed Consent**: `consent` (`"I consent to participate"` [99%] / `"I do not consent"` [1%]).
- **Behavioral Indicators (28 columns)**: 7 engagement indicators asked across 4 academic periods (P1–P4), coded 1–5:
  1. `resource_access`: Days/week accessing online materials (0 / 1–2 / 3–4 / 5–6 / Every day).
  2. `study_time`: Hours/week studying (<2 / 2–5 / 6–10 / 11–15 / >15).
  3. `review_time`: Hours/week reviewing past material (None / <1 / 1–2 / 3–4 / >4).
  4. `quiz_practice`: Attempting quizzes/practice (Never / Rarely / Sometimes / Often / Always).
  5. `assignment_engagement`: Timely assignment submission (Never .. Always).
  6. `study_consistency`: Regular study schedule agreement (Strongly disagree .. Strongly agree).
  7. `participation`: Active participation in lectures and discussions (Never .. Always).
- **Attention Check (1 column)**: Screening row with instruction to select `"Sometimes"`.
- **Outcome Items (3 columns)**: Self-reported attrition intention items ($A_1..A_3$, 5-point Likert scale: `intent_leave`, `intent_defer`, `intent_not_continue`).
  - Target: `at_risk = 1` if `attrition_intent_score >= 60.0`.

---

## 4. Latent Data Generation Mechanics
- **Cohort Size**: $N = 2400$ students generated with random seed 42.
- **Longitudinal Archetypes**:
  - `consistent_high`: 28%
  - `consistent_moderate`: 24%
  - `consistently_low`: 10%
  - `declining`: 17%
  - `sporadic`: 13%
  - `improving`: 8%
- **Factor Loadings**: Range from 0.70 (`participation`) to 0.90 (`resource_access`).
- **Diligence & Measurement Noise**: $\sigma_u = 0.08$ (shared student random effect), $\sigma_\nu = 0.10$ (item-specific noise).

---

## 5. Noise and Administrative Anomalies Injected
| Anomaly Type | Mechanism | Target Rate | Injected Count |
|---|---|---|---|
| Duplicate Submissions | Repeated rows with later timestamp & slight drift | 0.50% | 12 duplicates (total 2412 rows) |
| Missing Items (MCAR) | Completely random missing cells | 2.00% | 2.79% of all indicator cells |
| Period Dropout (MAR) | Missing all 7 indicators in P3 or P4 | 3.00% | 73 students (3.04%) |
| Acquiescent Responding | Systematic upward shift on agreement items | 10.00% | 232 students (9.67%) |
| Extreme Responding | Tendency toward categories 1 and 5 | 5.00% | 113 students (4.71%) |
| Midpoint Responding | Tendency toward category 3 | 5.00% | 139 students (5.79%) |
| Straight-Lining | Same answer across all 28 items | 2.00% | 50 students (2.08%) |
| Screener Failure | Invalid answer on attention check row | 2.00% | 62 students (2.58%) |
| Typographical Errors | Label variations ("Alwys", extra spaces, hyphens) | 0.30% | 205 cells (0.30%) |
| Non-Consent | Submission with consent refusal | 1.00% | 22 students (0.92%) |

---

## 6. Performance Calibration
- **5-Fold Cross-Validation LightGBM AUC**: 0.8906 (Target: 0.86–0.92).
- **5-Fold Cross-Validation Logistic Regression AUC**: 0.8795.
- **Prevalence of `at_risk == 1`**: 25.71% (Target: 24%–32%).

---

## 7. Ethical and Governance Statement
This dataset is **entirely synthetic** and does not contain personal identifying information or telemetry from real university students. It is generated strictly to ensure reproducible, leak-free pipeline engineering prior to IRB-approved real survey administration.
