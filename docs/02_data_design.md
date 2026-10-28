# 02 — Data Design, Questionnaire Schema, and Labeling Rules

## What it is
The formal, machine-readable data design and schema definition for the **Multimodal Student Engagement Intelligence (MEI)** component. It provides the single source of truth across the entire research system—governing synthetic sample data generation, real questionnaire export ingestion, data cleaning, feature extraction, model training, and the interactive web prediction dashboard.

## Why we do it
In real-world educational research, data collection instruments frequently evolve or produce noisy exports with mismatched column headers, mixed character encodings, and unstructured text options. By establishing an immutable canonical schema (`config/schema.yaml`), a variant header map (`config/mapping.yaml`), and strict target formulas (`config/labels.yaml`), we guarantee:
1. **Zero Target Leakage (Rule G3)**: The outcome items $A_1\text{–}A_3$ defining attrition intent are strictly quarantined from all feature pipelines.
2. **Platform Invariance**: Google Forms, Qualtrics, Excel uploads, or CSV exports map deterministically into identical internal data structures.
3. **Auditability**: Every survey response translates into numeric codes and back to canonical English labels without information loss.

## How it works step by step
1. **Canonical Schema Definition**: Every column is specified in `config/schema.yaml` with an assigned `role`, `type`, `allowed_codes`, and bidirectional `text_labels`.
2. **Header Normalization & Mapping**: When a survey file is loaded, its headers are matched against variant aliases in `config/mapping.yaml`. If an unknown header appears, RapidFuzz suggests candidates at an 88% similarity threshold for explicit confirmation (never auto-accepted).
3. **Bidirectional Codec**: `SurveyCodec` converts raw response strings into ordinal numbers ($1\text{–}5$) and vice versa, safely handling casing, irregular spaces, and unicode dashes.
4. **Target Calculation**: Target generation calculates the continuous `attrition_intent_score` ($0\text{–}100$) and binary `at_risk` flag ($1/0$) from outcome items $A_1\text{–}A_3$.
5. **Feature Matrix Isolation**: Only columns with `role: feature` enter downstream feature engineering and state modeling pipelines.

---

## Complete Data Dictionary
The survey defines **35 canonical columns**: 1 student identifier, 1 submission timestamp, 1 consent item, 28 temporal engagement features (7 indicators $\times$ 4 periods), 1 attention check, and 3 outcome items.

| Column Name | Role | Period | Data Type | Permitted Codes | Text Option Labels / Meaning |
|---|---|:---:|:---:|:---:|---|
| `student_id` | `id` | — | string | — | Unique anonymous student token (e.g., `STU-1001`) |
| `timestamp` | `meta` | — | string | — | Submission timestamp string (ISO 8601 or platform export) |
| `consent` | `consent` | — | integer | 0, 1 | `1` = I consent to participate; `0` = I do not consent |
| `resource_access_p1` | `feature` | 1 | integer | 1–5 | `1`=0 days; `2`=1–2 days; `3`=3–4 days; `4`=5–6 days; `5`=Every day |
| `resource_access_p2` | `feature` | 2 | integer | 1–5 | `1`=0 days; `2`=1–2 days; `3`=3–4 days; `4`=5–6 days; `5`=Every day |
| `resource_access_p3` | `feature` | 3 | integer | 1–5 | `1`=0 days; `2`=1–2 days; `3`=3–4 days; `4`=5–6 days; `5`=Every day |
| `resource_access_p4` | `feature` | 4 | integer | 1–5 | `1`=0 days; `2`=1–2 days; `3`=3–4 days; `4`=5–6 days; `5`=Every day |
| `study_time_p1` | `feature` | 1 | integer | 1–5 | `1`=<2 hrs; `2`=2–5 hrs; `3`=6–10 hrs; `4`=11–15 hrs; `5`=>15 hrs |
| `study_time_p2` | `feature` | 2 | integer | 1–5 | `1`=<2 hrs; `2`=2–5 hrs; `3`=6–10 hrs; `4`=11–15 hrs; `5`=>15 hrs |
| `study_time_p3` | `feature` | 3 | integer | 1–5 | `1`=<2 hrs; `2`=2–5 hrs; `3`=6–10 hrs; `4`=11–15 hrs; `5`=>15 hrs |
| `study_time_p4` | `feature` | 4 | integer | 1–5 | `1`=<2 hrs; `2`=2–5 hrs; `3`=6–10 hrs; `4`=11–15 hrs; `5`=>15 hrs |
| `review_time_p1` | `feature` | 1 | integer | 1–5 | `1`=None; `2`=<1 hr; `3`=1–2 hrs; `4`=3–4 hrs; `5`=>4 hrs |
| `review_time_p2` | `feature` | 2 | integer | 1–5 | `1`=None; `2`=<1 hr; `3`=1–2 hrs; `4`=3–4 hrs; `5`=>4 hrs |
| `review_time_p3` | `feature` | 3 | integer | 1–5 | `1`=None; `2`=<1 hr; `3`=1–2 hrs; `4`=3–4 hrs; `5`=>4 hrs |
| `review_time_p4` | `feature` | 4 | integer | 1–5 | `1`=None; `2`=<1 hr; `3`=1–2 hrs; `4`=3–4 hrs; `5`=>4 hrs |
| `quiz_practice_p1` | `feature` | 1 | integer | 1–5 | `1`=Never; `2`=Rarely; `3`=Sometimes; `4`=Often; `5`=Always |
| `quiz_practice_p2` | `feature` | 2 | integer | 1–5 | `1`=Never; `2`=Rarely; `3`=Sometimes; `4`=Often; `5`=Always |
| `quiz_practice_p3` | `feature` | 3 | integer | 1–5 | `1`=Never; `2`=Rarely; `3`=Sometimes; `4`=Often; `5`=Always |
| `quiz_practice_p4` | `feature` | 4 | integer | 1–5 | `1`=Never; `2`=Rarely; `3`=Sometimes; `4`=Often; `5`=Always |
| `assignment_engagement_p1` | `feature` | 1 | integer | 1–5 | `1`=Never; `2`=Rarely; `3`=Sometimes; `4`=Often; `5`=Always |
| `assignment_engagement_p2` | `feature` | 2 | integer | 1–5 | `1`=Never; `2`=Rarely; `3`=Sometimes; `4`=Often; `5`=Always |
| `assignment_engagement_p3` | `feature` | 3 | integer | 1–5 | `1`=Never; `2`=Rarely; `3`=Sometimes; `4`=Often; `5`=Always |
| `assignment_engagement_p4` | `feature` | 4 | integer | 1–5 | `1`=Never; `2`=Rarely; `3`=Sometimes; `4`=Often; `5`=Always |
| `study_consistency_p1` | `feature` | 1 | integer | 1–5 | `1`=Strongly disagree; `2`=Disagree; `3`=Neutral; `4`=Agree; `5`=Strongly agree |
| `study_consistency_p2` | `feature` | 2 | integer | 1–5 | `1`=Strongly disagree; `2`=Disagree; `3`=Neutral; `4`=Agree; `5`=Strongly agree |
| `study_consistency_p3` | `feature` | 3 | integer | 1–5 | `1`=Strongly disagree; `2`=Disagree; `3`=Neutral; `4`=Agree; `5`=Strongly agree |
| `study_consistency_p4` | `feature` | 4 | integer | 1–5 | `1`=Strongly disagree; `2`=Disagree; `3`=Neutral; `4`=Agree; `5`=Strongly agree |
| `participation_p1` | `feature` | 1 | integer | 1–5 | `1`=Never; `2`=Rarely; `3`=Sometimes; `4`=Often; `5`=Always |
| `participation_p2` | `feature` | 2 | integer | 1–5 | `1`=Never; `2`=Rarely; `3`=Sometimes; `4`=Often; `5`=Always |
| `participation_p3` | `feature` | 3 | integer | 1–5 | `1`=Never; `2`=Rarely; `3`=Sometimes; `4`=Often; `5`=Always |
| `participation_p4` | `feature` | 4 | integer | 1–5 | `1`=Never; `2`=Rarely; `3`=Sometimes; `4`=Often; `5`=Always |
| `attention_check` | `attention_check` | — | integer | 1–5 | Expected code: `3` ("Sometimes") |
| `intent_leave` | `outcome` | — | integer | 1–5 | A1: "I have seriously thought about leaving my degree" (1–5) |
| `intent_defer` | `outcome` | — | integer | 1–5 | A2: "I have considered deferring or taking a break" (1–5) |
| `intent_not_continue` | `outcome` | — | integer | 1–5 | A3: "I do not plan to continue next semester" (1–5) |

---

## Formulas and Label Rules

### 1. Continuous Attrition Intent Score (Track B)
The outcome items $A_1, A_2, A_3$ are measured on a 1–5 Likert scale. Their arithmetic mean ranges from $1.0$ (strong retention intention) to $5.0$ (strong attrition intention). The target score normalizes this range to $0\text{–}100$:

$$\bar{A} = \frac{A_1 + A_2 + A_3}{3}$$

$$\text{attrition\_intent\_score} = \left(\frac{\bar{A} - 1}{4}\right) \times 100$$

### 2. Binary At-Risk Classification (Track A)
A student is flagged as high engagement attrition risk if their score reaches or exceeds the threshold of $60.0$ (configured in `config/labels.yaml`):

$$\text{at\_risk} = \begin{cases} 1, & \text{if } \text{attrition\_intent\_score} \ge 60.0 \\ 0, & \text{if } \text{attrition\_intent\_score} < 60.0 \end{cases}$$

### Worked Examples
- **Student S-101 (Severe Risk Intent)**:
  - Answers: $A_1 = 5$ (Strongly agree), $A_2 = 4$ (Agree), $A_3 = 5$ (Strongly agree).
  - $\bar{A} = (5 + 4 + 5) / 3 = 4.667$
  - $\text{Score} = ((4.667 - 1) / 4) \times 100 = 91.67$
  - Classification: $\text{at\_risk} = 1$ (High Risk Band)

- **Student S-204 (Low Risk / Committed)**:
  - Answers: $A_1 = 1$ (Strongly disagree), $A_2 = 2$ (Disagree), $A_3 = 1$ (Strongly disagree).
  - $\bar{A} = (1 + 2 + 1) / 3 = 1.333$
  - $\text{Score} = ((1.333 - 1) / 4) \times 100 = 8.33$
  - Classification: $\text{at\_risk} = 0$ (Low Risk Band)

- **Student S-305 (Threshold Boundary Case)**:
  - Answers: $A_1 = 3$ (Neutral), $A_2 = 4$ (Agree), $A_3 = 3$ (Neutral).
  - $\bar{A} = (3 + 4 + 3) / 3 = 3.333$
  - $\text{Score} = ((3.333 - 1) / 4) \times 100 = 58.33$
  - Classification: $\text{at\_risk} = 0$ (Moderate Risk Band, below 60.0 threshold)

- **Student S-402 (Exact Threshold Match)**:
  - Answers: $A_1 = 4$ (Agree), $A_2 = 3$ (Neutral), $A_3 = 3.2$? For integer codes: $A_1 = 4, A_2 = 4, A_3 = 2 \implies \bar{A} = 3.333 \implies 58.33$.
  - Answers: $A_1 = 4, A_2 = 4, A_3 = 3 \implies \bar{A} = 3.667 \implies \text{Score} = 66.67 \implies \text{at\_risk} = 1$.

---

## Known Limitations and Threats to Validity

1. **Retrospective Self-Report (Recall Bias)**:
   - Students recall their weekly study behaviors for Period 1 (up to 12–14 weeks prior). While grid anchoring with semester milestones (midterms, coursework) aids memory retrieval, retrospective self-reports inherently exhibit smoothing and cognitive recall bias compared to real-time event logs.
2. **Proxy Label vs. True Behavioral Attrition**:
   - The ground truth target is self-reported *attrition intention* ($A_1\text{–}A_3$), which serves as an established theoretical proxy for actual academic withdrawal. Some students expressing departure intent persist due to financial or parental constraints, while others withdraw unexpectedly due to non-academic emergencies.
3. **Careless Responding and Common-Method Variance**:
   - Closed-ended surveys are subject to straight-lining (selecting the same rating across all rows) or hasty completion. This risk is actively countered in Phase 2 via attention-check verification, response duration screening, and intra-individual variance gating.
4. **Social Desirability Bias**:
   - Students may slightly over-report study hours or participation. By guaranteeing full anonymity, offering non-judgmental response intervals (e.g. `2–5 hours`), and emphasizing that responses have no academic bearing, social desirability pressure is minimized.

---

## Decisions and Rejected Alternatives
- **Decision**: Encoded all ordinal indicators on a 1–5 integer scale.
  - *Reason*: Standardizes input arrays for tree boosting, neural networks, and Gaussian Hidden Markov Models while enabling seamless round-trip decoding into human-readable text.
- **Decision**: Reject automatic acceptance of fuzzy matched headers.
  - *Reason*: Column misalignments in machine learning cause silent model failure. RapidFuzz (threshold 88%) generates suggestions for human verification, ensuring data integrity.

## Common Mistakes
- Treating outcome columns ($A_1\text{–}A_3$) as model inputs (instant target leakage).
- Assuming survey responses are already clean numbers without running through the bidirectional codec.
- Overriding the 60.0 threshold without logging changes in `config/labels.yaml` and `docs/DECISIONS.md`.

## How to Explain This in a Viva
"Our data design establishes an immutable 35-column canonical contract that guarantees zero target leakage. By formalizing 28 ordinal engagement indicators across 4 discrete temporal periods alongside 3 isolated attrition intent outcome items, we ensure that every downstream module—from HMM state decoding to tree boosting—operates on consistent, leak-free representations whether ingested from Google Forms, CSV, or live Excel spreadsheets."

## Explain Like I Am 5
"When you play a board game, everyone has to agree on the rules and what each card means before you roll the dice. This file is our rulebook: it explains what all 35 questions mean, makes sure nobody cheats by looking at the answers early, and tells us how to turn checkmarks into points."
