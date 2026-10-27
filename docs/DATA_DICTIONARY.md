# Academic Intelligence Framework: Data Dictionary & Specifications

**Component**: Academic Intelligence Framework  
**Project**: Multi-Factor Student Attrition Risk Prediction (Project ID: `KND_CDAP_IT_05`)  
**Student**: V. S. H. Gunasekara (`IT23164680`)  
**Supervisors**: Ms. Chathurika Koswatte, Ms. Anuththara Ekanayake  
**Cluster**: Centre of Excellence for AI (CoEAI), SLIIT  

---

## 1. Observable Longitudinal Signals (`academic_long.csv`)

Each record represents one semester of observation for a specific student. A full sequence spans up to 8 consecutive semesters (Years 1 through 4). Students who attrite cease producing records at their attrition semester.

| Column Name | Type | Range / Domain | Nullable? | Description & Academic Rationale |
| :--- | :--- | :--- | :---: | :--- |
| `student_id` | String | `STU_00001` - `STU_99999` | No | Anonymized unique student ID. Primary key for student grouping to prevent cross-semester data leakage. |
| `semester` | Integer | `1` to `8` | No | Chronological semester index (1 = Y1S1, 2 = Y1S2, ..., 8 = Y4S2). |
| `gpa` | Float | `0.00` to `4.00` | Yes (MCAR/MAR) | Semester Grade Point Average on SLIIT 4.0 scale. Core lagging academic performance indicator. |
| `exam_marks` | Float | `0.0` to `100.0` | Yes (MCAR/MAR) | Mean percentage mark across all formal modular final examinations in the semester. Strong summative proxy ($r \approx 0.88$ with GPA). |
| `attendance_pct` | Float | `0.0` to `100.0` | Yes (MCAR/MAR) | Percentage of mandatory lectures, labs, and tutorial sessions attended. Leading behavioral persistence signal. |
| `assignment_delay_days`| Float | `0.0` to `30.0` | Yes (MCAR/MAR) | Mean days late for coursework submissions past deadline. Proxy for academic strain and executive dysfunction. |
| `late_assignments_count`| Integer | `0` to `8` | Yes (MCAR/MAR) | Total count of assignments submitted late during the semester. |

*Note: In accordance with Non-Negotiable Rule 2, `academic_long.csv` contains zero ground truth labels and zero latent traits.*

---

## 2. Outcome Labels & Censoring Table (`outcomes.csv`)

One row per enrolled student, maintained strictly separate from features:

| Column Name | Type | Range / Domain | Description |
| :--- | :--- | :--- | :--- |
| `student_id` | String | `STU_00001` - `STU_99999` | Foreign key matching `academic_long.csv`. |
| `attrited` | Integer | `0` or `1` | Cumulative attrition status (`1` = Withdrawn or academically terminated, `0` = Persisted/Graduated). |
| `last_semester_observed` | Integer | `1` to `8` | Final semester in which student produced data before dropping out or completing degree. |
| `attrition_semester` | Integer / Null | `1` to `7` (or `null`) | Specific semester where the probabilistic dropout event occurred (`null` for persisting students). |

---

## 3. Prefix Task Samples (`task_samples.csv`)

Structured for early-warning evaluation across sliding horizons $t \in [2, 3, 4, 5, 6]$:

| Column Name | Type | Range / Domain | Description |
| :--- | :--- | :--- | :--- |
| `sample_id` | String | `{student_id}_t{t}` | Unique prefix window key (e.g., `STU_00012_t3`). |
| `student_id` | String | `STU_00001` - `STU_99999` | Mandatory grouping key for zero-leakage cross-validation. |
| `window_t` | Integer | `2` to `6` | Observation horizon (features available strictly up to semester $t$). |
| `target_attrite_next_2_sems` | Integer | `0` or `1` | Binary prediction target: will the student attrite at semester $t+1$ or $t+2$? |
| `overall_attrited` | Integer | `0` or `1` | Auxiliary cohort metadata (never used as feature input). |
| `attrition_semester` | Integer / Null | `1` to `7` (or `null`) | Exact semester of attrition event. |

---

## 4. Latent-Process Generator Parameters & Literature Benchmarks

The data generation process reflects documented empirical benchmarks in Sri Lankan higher education and international literature:

### 4.1 Benchmark Literature Basis
- **Baseline Tertiary Attrition**: Default calibrated to **$20\%$** (with empirical cohort yield of $23.6\%$). Validated against standard higher education attrition figures (Proposal Ref [17]: Valencia-Arias et al., 2026; Ref [3]: Yang, 2021).
- **High-Attrition Regional Extreme**: Configurable preset producing **$\sim 40\%$** attrition, reflecting SLIATE published attrition rates where specific regional centers reach 48% to 65% (Proposal Section 1.1, p. 11, citing [17]).
- **Curriculum Load Shock**: Semesters 3 and 4 introduce a programmatic difficulty drop of $-0.08$ GPA representing second-year technical core courses (Data Structures, Algorithms, Computer Systems Architecture).

### 4.2 Latent Trait Distributions
- $\text{Ability} \sim \mathcal{N}(0, 1)$
- $\text{Conscientiousness} \sim \mathcal{N}(0, 1)$
- $\text{Resilience} \sim \mathcal{N}(0, 1)$

### 4.3 Trajectory Archetype Distribution
1. `stable-high`: 24%
2. `stable-average`: 28%
3. `slow-decline`: 14%
4. `sudden-drop`: 9%
5. `recovering`: 8%
6. `volatile`: 8%
7. `struggling-from-start`: 9%

### 4.4 Missingness Specifications
- **MCAR (Missing Completely At Random)**: Random attendance sensor failures at $3.0\%$ base rate.
- **MAR (Missing At Random)**: Students with $\text{GPA} < 2.0$ have $2.2\times$ higher probability of missing submission delay and late count logs (captures disengagement and avoidance behaviors).
- **Overall Missingness**: Bounded within $[2.0\%, 4.5\%]$ across features.

---

## 5. Academic Risk Index (0–100) Output Contract

The model output consumed by the shared Fusion Framework:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "AcademicRiskIndexPayload",
  "type": "object",
  "required": [
    "participant_id",
    "observation_window_t",
    "academic_risk_index",
    "risk_band",
    "data_provenance",
    "model_provenance",
    "top_contributing_factors",
    "plain_language_explanation"
  ],
  "properties": {
    "participant_id": {"type": "string"},
    "observation_window_t": {"type": "integer"},
    "academic_risk_index": {"type": "number", "minimum": 0.0, "maximum": 100.0},
    "risk_band": {"type": "string", "enum": ["LOW", "MODERATE", "HIGH"]},
    "data_provenance": {"type": "string", "enum": ["SYNTHETIC", "INSTITUTIONAL_VERIFIED"]},
    "model_provenance": {"type": "string"},
    "top_contributing_factors": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "signal": {"type": "string"},
          "attribution_score": {"type": "number"},
          "direction": {"type": "string", "enum": ["INCREASES_RISK", "DECREASES_RISK"]}
        }
      }
    },
    "plain_language_explanation": {"type": "string"}
  }
}
```
