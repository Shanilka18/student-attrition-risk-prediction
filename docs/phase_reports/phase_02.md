# Phase 2 Report — Realistic Synthetic Survey Data Generation and Calibration

## 1. What Was Done
- **Synthetic Behavioral Model Implementation**: Implemented `src/mei/generator.py` with `SyntheticSurveyGenerator`, constructing a hierarchical latent engagement model across four periods (P1–P4) for $N = 2400$ students across 6 distinct longitudinal archetypes (`consistent_high`, `consistent_moderate`, `consistently_low`, `declining`, `sporadic`, `improving`).
- **Indicator Discretization & Margin Enforcement**: Generated continuous indicator signals using question-specific reliability loadings ($\lambda = 0.70\text{–}0.90$), shared student diligence random effects ($\sigma_u = 0.08$), and measurement noise ($\sigma_\nu = 0.10$). Enforced relative cut-points ensuring **no indicator exceeds 45% in any single category**.
- **Authentic Cognitive Responding & Mess Injection**:
  - Injected acquiescent response styles (10%), extreme responding (5%), midpoint responding (5%), straight-lining (2%), and attention check failures (2%).
  - Injected MCAR random item missingness (2%) and MAR period dropouts (3%, higher for low-engagement students).
  - Injected duplicate submissions (0.5% = 12 duplicates, yielding 2412 raw rows), label typographical variations (0.3%), and consent refusals (1% = 22 rows).
- **Signal Strength & Target Calibration**:
  - Implemented 5-fold cross-validation calibration loop comparing Logistic Regression and LightGBM on the 28 raw features with median imputation.
  - Converged on iteration 1 with `signal_strength = 1.000` and $w_0 = -1.823$, yielding a LightGBM CV AUC of `0.8906` (target: 0.86–0.92), LR CV AUC of `0.8795`, and at-risk prevalence of `25.71%` (target: 24%–32%).
- **Raw Export & Evaluation Samples Generation**:
  - Generated `data/raw/survey_raw_sample.csv` and `data/raw/survey_raw_sample.xlsx` in exact Google Forms format with long question headers and text option labels.
  - Generated `data/raw/ground_truth_hidden.csv` strictly isolated for Phase 5 HMM/clustering recovery evaluation.
  - Generated `data/samples/sample_upload_clean.xlsx` (200 students, canonical headers, clean data, NO outcomes) and `data/samples/sample_upload_messy.csv` (200 students, variant headers, blanks, out-of-range values, extra `campus_branch` column, NO outcomes) for website scoring page testing.
- **Publication-Quality Exploratory Figures**: Implemented `src/mei/eda.py` and rendered 6 figures adhering to the brand palette (§10 of `PROJECT_BRIEF.md`) at 300 DPI in both PNG and SVG formats in `artifacts/figures/` and `docs/figures/`.
- **CLI Command Implementation**: Implemented `python -m mei.cli make-sample-data`, executing the entire pipeline in **23.3 seconds** (well under the 2-minute ceiling).
- **Quality Assurance**: Added 7 comprehensive unit and integration tests in `tests/test_sample_data.py` and updated `tests/test_smoke.py`, achieving 22 passing tests with clean `ruff` and `black` formatting.
- **Git Commit Withheld**: Abided strictly by user instruction to withhold git commits until explicit authorization.

---

## 2. Files Created and Modified
- `config/project.yaml` *(Modified)*: Added `sample_data:` configuration block with all generation, archetype, and calibration hyperparameters.
- `src/mei/generator.py` *(Created)*: Core synthetic cohort generation, mess injection, outcome modeling, and calibration engine.
- `src/mei/eda.py` *(Created)*: Exploratory data analysis figure generation in brand palette (PNG + SVG).
- `src/mei/cli.py` *(Modified)*: Implemented `make-sample-data` command with Rich calibration, reality check, and artifacts tables.
- `data/raw/survey_raw_sample.csv` *(Created)*: Raw Google Forms CSV export (2412 rows, 35 columns).
- `data/raw/survey_raw_sample.xlsx` *(Created)*: Raw Google Forms Excel workbook (2412 rows, 35 columns).
- `data/raw/ground_truth_hidden.csv` *(Created)*: True latent engagement and archetypes (2400 rows, 9 columns).
- `data/samples/sample_upload_clean.xlsx` *(Created)*: Clean scoring test sample (200 rows, 29 columns).
- `data/samples/sample_upload_messy.csv` *(Created)*: Messy validation test sample (200 rows, 30 columns).
- `artifacts/figures/` & `docs/figures/` *(Created)*: 12 figure files (6 PNG, 6 SVG):
  - `eda_01_marginal_distributions`
  - `eda_02_indicator_correlations`
  - `eda_03_temporal_archetypes`
  - `eda_04_at_risk_by_archetype`
  - `eda_05_missingness_map`
  - `eda_06_response_anomalies`
- `docs/figures/INDEX.md` *(Modified)*: Cataloged the 6 newly generated EDA publication figures.
- `docs/03_sample_data.md` *(Created)*: Complete G8 documentation with math formulations, calibration log, reality check, and viva explanations.
- `docs/data_card_sample.md` *(Created)*: Full dataset card documenting provenance, schema, noise mechanisms, and ethical boundaries.
- `docs/DECISIONS.md` *(Modified)*: Logged architectural decisions D-009, D-010, D-011, and D-012.
- `tests/test_sample_data.py` *(Created)*: 7 unit tests verifying reproducibility, dimensions, leakage guards, and distributions.
- `tests/test_smoke.py` *(Modified)*: Integrated CLI test for `make-sample-data`.
- `docs/phase_reports/phase_02.md` *(Created)*: Phase 2 verification report.

---

## 3. Key Numbers Read Directly from Output Files
- **Total Students ($N$)**: `2400` profiles (2412 rows with duplicates in `data/raw/survey_raw_sample.csv`)
- **Survey Columns**: `35` canonical columns with long Google Forms header formats
- **Signal Strength**: `1.000` (converged on iteration 1)
- **Calibrated Intercept ($w_0$)**: `-1.823`
- **At-Risk Prevalence**: `25.71%` (617 at-risk students out of 2400, inside target [24%, 32%])
- **LightGBM 5-Fold CV AUC-ROC**: `0.8906` (inside target [0.86, 0.92])
- **Logistic Regression 5-Fold CV AUC-ROC**: `0.8795`
- **AUC-ROC Gap (LGB - LR)**: `+0.0110` (< 0.03, recorded as near-linear structure on raw features)
- **Marginal Category Proportions**: `< 45%` across all 28 indicators (maximum single category: 32.4%)
- **Duplicate Submissions**: `12` rows (`0.50%`)
- **Typographical Errors**: `205` cells (`0.30%`)
- **Consent Refusals**: `22` submissions (`0.92%`)
- **Missing Cells (MCAR)**: `1,873` / `67,200` (`2.79%`)
- **Period Dropout Students (MAR)**: `73` / `2,400` (`3.04%`)
- **Straight-Lining Responders**: `50` / `2,400` (`2.08%`)
- **Attention Check Failures**: `62` / `2,400` (`2.58%`)
- **Clean Sample Upload Size**: `200` rows, `29` columns (`student_id` + 28 features, 0 outcome columns)
- **Messy Sample Upload Size**: `200` rows, `30` columns (`Student_ID` + 28 features + `campus_branch`, 0 outcome columns)
- **make-sample-data Execution Time**: `23.29 seconds` (target: < 120 seconds)
- **Pytest Results**: `22 passed in 51.31s`

---

## 4. Checks Run with PASS/FAIL Status
| Check | Command / Target | Status | Details |
|---|---|---|---|
| CLI Command | `python -m mei.cli make-sample-data` | **PASS** | Completed in 23.3s, rendered all tables & figures |
| Unit Test Suite | `pytest tests/` | **PASS** | 22 passed in 51.31s |
| Linter | `ruff check src tests` | **PASS** | 0 warnings, 0 errors |
| Formatter | `black --check ...` | **PASS** | All source and test files clean |
| Target Leakage Guard (G3) | `test_no_target_leakage_in_evaluation_samples` | **PASS** | Outcome items strictly excluded from upload samples |
| Ground Truth Isolation | `test_ground_truth_hidden_from_feature_schema` | **PASS** | Zero overlap between hidden columns and feature schema |
| Seed Reproducibility | `test_generator_seed_reproducibility` | **PASS** | Identical dataframes generated on repeated runs |
| Marginal Distribution Ceiling | `test_marginal_distributions_rule` | **PASS** | Every indicator has max category < 45% |
| At-Risk Prevalence Bounds | `test_sample_cohort_dimensions_and_prevalence` | **PASS** | 25.71% in [24%, 32%] |
| Model Performance Calibration | 5-Fold Cross-Validation | **PASS** | LightGBM AUC 0.8906 in [0.86, 0.92] |
| Figure Generation | Directory inspection | **PASS** | 6 figures exist in both PNG and SVG formats in artifacts and docs |
| Figure Catalog Integrity | `docs/figures/INDEX.md` | **PASS** | All 6 figures indexed with captions |

---

## 5. Risks and Architectural Decisions Logged
- **Decision D-009**: Adopted hierarchical latent engagement process across 6 behavioral archetypes with diligence random effects.
- **Decision D-010**: Implemented 5-fold CV calibration loop to constrain LightGBM AUC into [0.86, 0.92] and hold prevalence near 28%.
- **Decision D-011**: Strictly isolated `data/raw/ground_truth_hidden.csv` from all feature extraction and modeling routines.
- **Decision D-012**: Created dual-split evaluation artifacts (`sample_upload_clean.xlsx` and `sample_upload_messy.csv`) to decouple model inference testing from ingestion validation.
- **Structure Finding Logged**: LightGBM AUC (0.8906) and LR AUC (0.8795) differ by +0.0110 (< 0.03), confirming that raw indicator combinations exhibit near-linear predictive structure prior to temporal feature engineering.
- **Git Commit Status**: Per explicit user instruction, no git commits have been executed.
