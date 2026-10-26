# Phase 1 Documentation: Data Contract & Ingestion Specifications

> **RESEARCH NOTICE & ETHICAL DISCLAIMER**  
> The Psychosocial Risk Index (PRI) is an academic decision-support indicator developed for educational research at SLIIT Kandy. It is **not** a clinical psychological assessment or psychiatric diagnosis. All student records are strictly de-identified using anonymous Study IDs (`study_id`).

---

## 1. Phase Overview & Objectives

In machine learning and psychometric research, models are only as reliable as the integrity of the data ingested. Upstream data inconsistencies—such as unstandardized column headers across surveys, out-of-bounds Likert responses, missing values, duplicate participant entries, or inattentive "straight-lining" respondents—can severely corrupt factor analysis, bias model calibration, and compromise experimental reproducibility.

Phase 1 establishes a **formal Data Contract** for the Psychosocial Intelligence Framework. It implements:
1. Strict schema specifications (`/config/schema.yaml` and `/src/data/schema.py`).
2. A comprehensive validation engine (`validate_dataframe`) returning an auditable diagnostic report.
3. An intelligent fuzzy column-mapping engine (`suggest_column_mappings`) to resolve colloquial or differing questionnaire headers with confidence scoring.
4. Robust tabular file readers supporting `.csv`, `.xlsx`, and `.xls` formats.

---

## 2. Simple Explanation (Explain Like I am 5 Years Old)

### The Everyday Analogy: The "Airport Security & Passport Gatekeeper"

Imagine you are about to board an airplane to go on holiday. Before anyone can walk onto the plane, they must pass through the **Airport Security & Passport Gate**:

1. **The Official Passport Check (`study_id` & Required Items):**  
   The officer checks that you have your passport and ticket. If you left your passport at home (missing `study_id`) or forgot one of your mandatory boarding passes (missing one of the 15 questions), the gate remains closed. You cannot board until your documents are complete.

2. **The Luggage Weight Scale (Likert Scale Boundaries $[1, 5]$):**  
   At the check-in desk, your bag is placed on a scale. It must weigh between 1 and 5 kilograms. If someone brings a bag weighing 6 kilograms (or -1 kg!), the alarm buzzes: *"Invalid measurement! We only allow between 1 and 5."*

3. **Spotting Duplicate Passengers (Duplicate `study_id`):**  
   If the computer sees that two people are trying to board using the exact same seat number and name, security pauses everything to investigate the duplicate identity.

4. **Catching the "Sleepy Passenger" (Straight-Lining Check):**  
   Imagine a passenger is so exhausted that whenever the immigration officer asks a question—*"What is your name?", "Where are you going?", "Did you pack your own bag?"*—the passenger mindlessly repeats *"Yes, Yes, Yes, Yes"* without even listening. Our system catches these "straight-lining" respondents who circled the exact same number (e.g. 3, 3, 3...) for all 15 questions, flagging them as unengaged answers so they don't spoil our scientific study.

5. **The Friendly International Translator (Fuzzy Column Mapping):**  
   If an international traveler hands over a ticket that says *"Q1_workload"* instead of the official code *"S1"*, the gatekeeper has a smart translator dictionary that says: *"Aha! You mean Question S1 (Academic Workload Stress)! I am 95% confident."*

---

## 3. Technical Explanation (For Viva Panel & ML Specialists)

### 3.1 Data Contract Architecture & Invariants

The data contract is formalized through declarative configuration in `config/schema.yaml` and programmatic enforcement in `src/data/schema.py`. It guarantees the following schema invariants before any feature transformation or model ingestion occurs:

$$\mathcal{D} = \left\{ \mathbf{x}_i \in \mathbb{N}^{15}, \, \text{id}_i \in \mathcal{S} \mid \forall j \in \{1, \dots, 15\}, \, x_{i,j} \in \{1, 2, 3, 4, 5\} \right\}_{i=1}^N$$

Where:
- $\text{id}_i$ satisfies uniqueness: $|\{\text{id}_1, \dots, \text{id}_N\}| = N$.
- All Likert items $x_{i,j}$ are non-null integers bounded on the closed interval $[1, 5]$.
- Optional supervised labels satisfy $\text{attrition\_label}_i \in \{0, 1\}$ and $\text{dropout\_intention}_i \in [1, 5]$.

### 3.2 Canonical Schema Definition

| Variable Name | Construct / Role | Type | Constraints | Sample Aliases Handled |
| :--- | :--- | :--- | :--- | :--- |
| `study_id` | Participant Identifier | String | Non-null, Unique | `id`, `student_id`, `participant_id` |
| `S1`, `S2`, `S3` | Stress (Risk Factor) | Integer | $\in \{1, 2, 3, 4, 5\}$ | `q1_stress`, `exam_anxiety`, `personal_strain` |
| `M1`, `M2`, `M3` | Motivation (Protective) | Integer | $\in \{1, 2, 3, 4, 5\}$ | `degree_interest`, `determination`, `career_goal` |
| `B1`, `B2`, `B3` | Belongingness (Protective) | Integer | $\in \{1, 2, 3, 4, 5\}$ | `q7_belonging`, `faculty_engagement`, `peer_acceptance` |
| `SE1`, `SE2`, `SE3` | Self-Efficacy (Protective) | Integer | $\in \{1, 2, 3, 4, 5\}$ | `academic_confidence`, `resilience`, `time_management` |
| `PS1`, `PS2`, `PS3` | Peer Support (Protective) | Integer | $\in \{1, 2, 3, 4, 5\}$ | `study_group`, `emotional_support`, `collaboration` |
| `age_group` | Demographic Metadata | String | Allowed categories | `age`, `age_bracket` |
| `gender` | Demographic Metadata | String | Allowed categories | `sex`, `gender_identity` |
| `year_of_study` | Academic Metadata | String | `1`, `2`, `3`, `4` | `academic_year`, `level`, `yos` |
| `programme` | Academic Metadata | String | Valid degree codes | `program`, `major`, `degree` |
| `attrition_label` | Downstream Validation | Integer | $\in \{0, 1\}$ | `dropout`, `withdrawn`, `label` |
| `dropout_intention` | Validation Construct | Integer | $\in \{1, 2, 3, 4, 5\}$ | `turnover_intent`, `withdrawal_intention` |

### 3.3 Psychometric Quality Assurance: Straight-Lining Detection

In survey methodology, **straight-lining** (satisficing / non-differentiation) occurs when a respondent selects the identical Likert response across all consecutive battery items to minimize cognitive effort. This introduces systematic non-random measurement error.

The validation engine computes the intra-individual item variance for respondent $i$:

$$s_i^2 = \frac{1}{K - 1} \sum_{j=1}^{K} (x_{i,j} - \bar{x}_i)^2, \quad \text{where } K = 15$$

If $s_i^2 = 0.0$ and all $K$ items are populated, the respondent is flagged as a straight-liner. The validation report isolates these participant IDs and marks them for exclusion from psychometric factor analysis and model training to preserve true variance.

### 3.4 Fuzzy Column Mapping Algorithm

To accommodate surveys collected across different academic terms or external software (Qualtrics, Google Forms, LMS exports), the framework utilizes a two-tier resolution strategy:
1. **Tier 1 (Exact & Alias Lookup):** Normalizes column names (case-folding, stripping whitespace, removing underscores/hyphens) against registered aliases.
2. **Tier 2 (Fuzzy String Similarity):** Employs Ratcliff-Obershelp sequence pattern matching via `difflib.SequenceMatcher`:

$$\text{similarity}(s_1, s_2) = \frac{2 \cdot M}{|s_1| + |s_2|}$$

where $M$ is the number of matching characters in matching substrings. Substring inclusion bonuses guarantee robust mappings (e.g. `academic_resilience` $\to$ `SE2` with confidence $\ge 0.75$).

### 3.5 Format Agnosticism

The ingestion module `read_survey_file` dynamically selects the appropriate engine based on file headers and extensions:
- `.csv`: Read via `pandas.read_csv` with fallback to `latin1` for non-UTF8 survey exports.
- `.xlsx`: Read via `openpyxl`.
- `.xls`: Read via `xlrd`.
- In-memory `BytesIO` and `StringIO` buffers: Supported for web API multi-part file uploads without requiring temporary disk persistence.
