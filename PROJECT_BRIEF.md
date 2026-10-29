# PROJECT BRIEF — Environmental & Financial Factors

**Project ID:** KND-CDAP-IT-05  
**Component:** Environmental & Financial Factors  
**Index:** Environmental & Financial Index (EFI) — 0–100, higher = higher risk linked to study conditions  
**Website Product Name:** Study Conditions Index  

---

## Engineering Guidelines & Rules

- **G14 Small commits.** Work on a branch feature/<topic>, never directly on main. After EVERY numbered STEP of a prompt: run ruff, black --check and pytest for the touched code. If all pass, commit with a conventional message (feat:, fix:, docs:, test:, refactor:, chore:) describing only that step, then `git push -u origin <branch>`. If anything fails, fix it first; never commit failing code; never use --no-verify or --force. One commit = one logical change (code + its tests + its doc). At the end of a phase merge into main with --no-ff, push main and tag ef-NN. Never commit data/real/, .env, .venv, node_modules, model binaries or any file over 5 MB (check `git status` and file sizes before each commit). Show `git log --oneline -10` in every final reply.

---

## 1. Component Identity

| Field | Value |
|---|---|
| Component name | Environmental & Financial Factors |
| Index name | Environmental & Financial Index (EFI) |
| Scale | 0–100 (higher = greater study-condition risk) |
| Website product name | **Study Conditions Index** |
| Colour palette | Emerald / Teal (predictive), Purple (SHAP), Amber (causal) |

---

## 2. Data Truth

- **Data source:** One anonymous closed-ended questionnaire (17 items + shared outcome items).
- **Current state:** Generated sample data for pipeline validation. Real survey data integrated later through config only — no code changes required.
- **Config flag:** `data_source: sample | survey` must appear in every pipeline run record.
- **Rule:** Synthetic/sample numbers must **never** appear in production documentation or reports as real findings.

---

## 3. Questionnaire

### 3.1 Environmental & Financial Items (17)

Defined canonically in `docs/questionnaire_envfin_v2.md` and `config/schema_envfin.yaml`.

| # | Canonical Name | Type | Levels | Pressure Direction |
|---|---|---|---|---|
| Q23 | `income_band` | Ordinal | 5 | Reversed |
| Q24 | `aid_status` | Ordinal | 3 | Higher |
| Q25 | `family_support_duty` | Binary | 2 | Higher |
| Q26 | `cost_difficulty` | Ordinal | 4+PNS | Higher |
| Q27 | `commute_time` | Ordinal | 4+SNA | Higher |
| Q28 | `commute_spend` | Ordinal | 4+SNA | Higher |
| Q29 | `transport_mode` | Nominal | 5+SNA | Mapped |
| Q30 | `living_situation` | Nominal | 4 | Mapped |
| Q31 | `quiet_study_place` | Ordinal | 3 | Higher |
| Q32 | `home_internet` | Ordinal | 4 | Higher |
| Q33 | `main_device` | Nominal | 4 | Mapped |
| Q34 | `mobile_data_plan` | Ordinal | 3 | Higher |
| Q35 | `part_time_job` | Ordinal | 3 | Higher |
| Q36 | `job_hours` | Ordinal | 5+SNA | Higher (conditional on Q35) |
| Q37 | `financial_shock` | Ordinal | 3 | Higher |
| Q38 | `money_worry` | Ordinal | 5 | Higher |
| Q39 | `missed_due_to_barriers` | Ordinal | 4 | Higher |

### 3.2 Shared Outcome Items (from team questionnaire — never features)

| Canonical Name | Role |
|---|---|
| `attrition_intent_score` | continuous target (0–100) |
| `at_risk` | binary label (1 if attrition_intent_score ≥ 60) |
| `student_id` | pseudonymous ID (never a feature) |
| `timestamp` | survey timestamp (never a feature) |
| `consent` | consent flag (never a feature) |
| `attention_check` | quality gate (never a feature) |

### 3.3 Limitations (self-declared)

1. **Self-report bias** — items rely on respondent honesty and recall accuracy.
2. **Label is a proxy** — `attrition_intent_score` measures *stated intention*, not actual withdrawal behaviour.
3. **No time dimension** — a single cross-sectional measurement; no longitudinal trajectory.
4. **Income sensitivity** — household income is sensitive; permit non-disclosure and report its missingness without assuming a fixed non-response rate.
5. **Observational data** — causal estimates assume conditional exchangeability; unmeasured confounders cannot be ruled out.

---

## 4. Targets & EFI Score

| Item | Specification |
|---|---|
| Primary continuous target | `attrition_intent_score` (0–100) |
| Binary label | `at_risk = 1` if `attrition_intent_score ≥ 60` |
| EFI formula | `EFI = round(100 × calibrated P(at_risk))` |
| Bands | **Low** 0–34 · **Moderate** 35–64 · **High** 65–100 |
| Decision threshold | Tuned on validation set (Youden's J or F-beta); recorded in `artifacts/metrics/` |

---

## 5. Features & Risk Profiles

### 5.1 Feature Set
- **Raw items:** All 17 questionnaire items, rescaled to [0, 1] pressure units.
- **Four derived indices** (0–1, higher = more pressure):
  - **Financial Stress** — income (rev), aid (rev), family duty, cost difficulty, financial shock, money worry, job hours.
  - **Commute Burden** — time, spend, transport pressure (0.0 when studying fully online).
  - **Digital Access Gap** — internet reliability, device pressure, mobile data (rev).
  - **Study Environment Pressure** — quiet place (rev), living situation pressure.
- **Ratio & interaction features** — e.g., `job_hours × cost_difficulty`, `commute_spend / income_band`.

### 5.2 Risk Profiles (replaces Engagement concept of "states")
- Latent risk profiles identified by **clustering / GMM / latent class analysis** fitted strictly **inside cross-validation folds** (no leakage).
- **No HMM, no periods, no sequence models** — this is a cross-sectional design.
- Profile labels: e.g., "High Financial Pressure", "Digital Barrier", "Commute-Burdened", "Low Risk".
- Profiles used for stratified reporting and counsellor dashboards only; never as features for the main classifier.

---

## 6. Models & Two Tracks

### Track A — Interpretable & Deployable
Linear / Shallow models: Logistic Regression (L1, L2, ElasticNet), Ridge, Linear SVM, Gaussian NB.

### Track B — Non-Linear Benchmark
Tree Ensembles: Random Forest, Extra Trees, Gradient Boosting, HistGB, AdaBoost, Decision Tree.  
Deep Tabular: MLP (shallow & deep), ResNet-Tabular, TabNet, CNN-1D.  
Voting / Stacking ensembles.

### Metrics (ranked by priority)
1. PR-AUC (35%) — precision-recall on minority class.
2. ROC-AUC (25%) — discrimination.
3. Calibration / 1 − Brier Score (20%) — probability reliability.
4. Balanced Accuracy (10%).
5. Macro F1 (10%).

**Leaderboard composite score** = weighted sum of above.  
**Overfit rule:** if `val_composite − cv_composite > 0.04`, model is disqualified.  
**One-time test:** final test set accessed exactly once for the category winner and selected finalist.

---

## 7. Calibration & Production Score

- Calibrators: Platt Scaling (Logistic) and Isotonic Regression fitted on held-out fold.
- If ECE > 0.15, the score is relabeled `"normalised model-risk score"` rather than a calibrated probability.
- EFI output = `round(100 × calibrated_probability)`, clamped to [0, 100].

---

## 8. Website Pages

| Page | Description |
|---|---|
| **Overview** | EFI methodology, pipeline summary, synthetic-data transparency banner |
| **Model Leaderboard** | Composite scores, significance tests, parsimony rule, calibration status |
| **Risk Profiles** | Latent profile visualisations (GMM/clustering), profile descriptions |
| **What Drives Risk** | SHAP waterfall, global bar chart, causal risk differences |
| **Score Students** | Upload CSV/XLSX → per-student EFI, band, SHAP attribution, counselling narrative |
| **Train on New Data** | Upload labelled survey → guarded retraining (requires `--outcome` + `--confirm`) |

### Website Rules
- Sticky banner: `"SYNTHETIC DATA — pipeline validation only"` on every page until real survey data is loaded.
- Zero hardcoded numbers — all metrics pulled live from `artifacts/` via FastAPI.
- Colour scheme: Emerald/Teal for predictive, Purple for SHAP, Amber for causal.
- Differentiated styling between SHAP attributions (correlational) and causal estimates (policy effects).

---

## 9. Architecture (preserved from base)

```
data_gen → cleaning → features → profiles (GMM/cluster) → models
         → leaderboard (significance, calibration, parsimony)
         → SHAP (explain) → causal (sIPW/AIPW/CausalForest)
         → FastAPI → React dashboard
```

All stages are modular. Swapping real survey data requires only changing `data_source` in config; no architecture change.
