# HANDOVER GUIDE

**Project**: KND_CDAP_IT_05 — Academic Intelligence Framework  
**Student**: V. S. H. Gunasekara · IT23164680  
**Purpose**: How to maintain, retrain, and plug in real university data.

---

## 1. How to Retrain the Models

### Quick retrain (all models, quick Optuna budget)
```bash
python run.py train --budget quick
```

### Full retrain from scratch (force, full budget)
```bash
python run.py train --budget full --force
```

### Retrain a specific model only
```bash
python run.py train --models catboost gru --budget standard
```

### Retrain a specific category
```bash
python run.py train --categories E --budget full
```

**What happens during retrain**:
1. Optuna searches hyperparameters using 5-fold CV on training partition only.
2. Best config trained on full training partition with early stopping.
3. 3-seed stability runs (seeds 42, 123, 999 — configurable in `config.yaml`).
4. Platt calibration fitted on validation partition.
5. Recall-constrained threshold optimised on validation partition.
6. All artifacts saved to `artifacts/models/<category>/<model_key>/`.
7. If `--force` is not set, models with existing `summary.json` are skipped (resumable).

---

## 2. How to Plug In Real University Data

### Overview
The pipeline is designed so that **only Phase 2 (data generation) needs to be replaced**. Phases 3–9 are data-agnostic — they consume the output of Phase 2 in a documented format. No model code, evaluation code, or API code changes.

### Step-by-Step

#### Step 1 — Obtain IRB clearance
Real student records are PII. Obtain written IRB/ethics committee approval before proceeding. Ensure: data anonymisation, purpose limitation, and data minimisation.

#### Step 2 — Place raw data in `/data/raw/`
```
data/raw/
├── student_records.csv      # One row per (student_id, semester)
└── README.md                # Provenance notes (who extracted, when, cohort definition)
```

#### Step 3 — Map your institution's columns to the schema

The schema (`data/schema/schema.json`) expects this exact long format:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `student_id` | string | alphanumeric, hyphen, underscore | Anonymised student identifier |
| `semester` | integer | 1–8 | Academic semester number |
| `gpa` | float | 0.0–4.0 | Semester GPA |
| `exam_marks` | float | 0.0–100.0 | Average exam score this semester |
| `attendance_pct` | float | 0.0–100.0 | Attendance percentage |
| `assignment_delay_days` | float | 0.0–45.0 | Average days late for assignments |
| `late_assignments_count` | integer | 0–15 | Number of late assignments |

**If your institution uses different column names**, create a mapping script in `data/raw/`:
```python
# data/raw/map_columns.py
import pandas as pd

df = pd.read_csv("data/raw/your_export.csv")
df = df.rename(columns={
    "StudentID": "student_id",
    "Term": "semester",
    "SemGPA": "gpa",
    "ExamAvg": "exam_marks",
    "AttPct": "attendance_pct",
    "AvgDelayDays": "assignment_delay_days",
    "LateSubmissions": "late_assignments_count",
})

# Ensure semester numbering is 1-indexed consecutive integers
# Anonymise student IDs if not already done:
id_map = {sid: f"REAL_{i:05d}" for i, sid in enumerate(df.student_id.unique())}
df["student_id"] = df["student_id"].map(id_map)

# Remove the attrited column if present (the pipeline derives labels from missingness/outcome):
# Add attrited column: 1 if student left before their maximum semester
outcomes = df.groupby("student_id")["semester"].max().reset_index()
# ... (institution-specific attrition label derivation)

df.to_csv("data/synthetic/academic_long.csv", index=False)  # Overwrite the synthetic file
```

> **Important**: The pipeline reads from `data/synthetic/academic_long.csv` by convention. Place your real data here, or update `config.yaml` → `data.long_csv_path`.

#### Step 4 — Create the outcome file

The pipeline expects `data/synthetic/outcomes.csv`:
| Column | Description |
|--------|-------------|
| `student_id` | Anonymised ID |
| `attrited` | 1 if student left before completing all semesters, 0 otherwise |
| `attrition_semester` | Semester in which attrition occurred (NaN if retained) |
| `cohort_year` | Entry year (e.g., 2021, 2022, 2023, 2024) |

#### Step 5 — Change `provenance_tag` in config
Open `config.yaml` and set:
```yaml
data:
  provenance: REAL          # Was: SYNTHETIC
  institution: SLIIT        # Your institution name
```
This propagates through the API (`/meta` returns `data_mode: REAL`) and the fusion contract.

#### Step 6 — Re-run Phases 3–9

```bash
python run.py preprocess   # Split by student, re-engineer features
python run.py train        # Retrain all models on real data
python run.py evaluate     # New test-set evaluation (new SHA-256 fingerprint)
python run.py explain      # Re-compute SHAP on real data
python run.py serve        # API now serves real-data models
```

> **The SYNTHETIC banner in the frontend disappears automatically** once `data_mode` is REAL.

#### Step 7 — Re-derive risk band cut-points
After Phase 5 (training), inspect `artifacts/models/Category_E/gru/summary.json` for the new validation set distribution. The risk band thresholds in `config.yaml` → `risk_bands` may need adjustment based on real-data attrition prevalence. Update and re-run Phase 7:
```bash
python run.py explain
```

#### Step 8 — Fairness audit
With real data and demographic labels (if ethically available), run:
```python
from backend.src.evaluation.fairness import audit_demographic_parity
# (implement this module — not yet in the codebase; see future work)
```

---

## 3. What Changes in Claims Once Real Data Is Used

| Claim | On Synthetic Data | On Real Data |
|-------|-------------------|-------------|
| ROC-AUC, PR-AUC, F2 | Demonstrates methodology | Real evidence for GRU > Gradient Boosting |
| Research question answer (POSITIVE) | Valid for synthetic distribution | May change direction |
| Risk band cut-points | Derived from synthetic val distribution | Must be re-derived |
| SHAP feature importances | Attendance and delay dominate | May change (real data may have different signals) |
| Calibration (ECE) | 0.04 (very good) | Unknown — must re-verify |
| NFR1 latency (18.2 ms) | Independent of data source | Unchanged — model architecture only |
| Fusion contract schema | Unchanged | Unchanged — schema is data-agnostic |
| SYNTHETIC banner | Shown everywhere | Hidden automatically |
| AI_USE_DISCLOSURE.md | Includes synthetic data caveat | Update to reflect real data evaluation |

---

## 4. Known Limitations

### 4.1 Synthetic Data Dependency
All published results are from synthetic data. Generator parameters (α=0.65, 7 archetypes, 23.6% attrition) were calibrated against literature but are not ground truth.

### 4.2 No Fairness Analysis
No demographic features in synthetic data. Fairness across gender, socioeconomic status, and disability status is untested. Required before deployment.

### 4.3 Single Institution Scope
Trained on SLIIT-calibrated data. Generalisation to other institutions requires retraining. A multi-institutional federated learning approach is needed for a shared model.

### 4.4 Fixed Feature Set
The 22 engineered features cover academic, attendance, and engagement signals. Psychosocial signals (stress, motivation), financial stress, and peer network effects are not included.

### 4.5 Static Model After Training
The model does not update incrementally. Each new cohort (new semester year) requires a full retrain cycle. Online learning would make it adaptive.

### 4.6 Small Model Leaderboard
Only 3 of 28 registered model classes have full leaderboard entries in the current artifact set. All 28 are implemented but require a full training run (~6–12 hours) to appear in the leaderboard.

### 4.7 Generator Sensitivity
The AR(1) momentum parameter α=0.65 and other generator settings have not been formally sensitivity-tested. Results may vary with different generator parameterisations.

---

## 5. Contact & Handover Notes

**Student**: V. S. H. Gunasekara · it23164680@my.sliit.lk  
**Supervisors**: Ms. Chathurika Koswatte · Ms. Anuththara Ekanayake  
**KND Fusion Framework interface**: `backend/src/explain/fusion_contract.py` — `FusionAcademicRiskPayload`  
**Teammate integration**: Share the `/predict/upload` or `/predict/student` endpoint URL; the fusion JSON contract is in every prediction response.

---

*Generated 2026-10-19. All steps verified on Python 3.12.10, Windows 11.*
