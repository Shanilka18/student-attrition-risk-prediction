# Psychosocial Intelligence Framework (PRI Component)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3-61dafb.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646cff.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38bdf8.svg)](https://tailwindcss.com/)
[![Tests](https://img.shields.io/badge/tests-69%20passed%20%7C%205%20vitest-success.svg)](./tests/)

> **RESEARCH NOTICE:**  
> The Psychosocial Risk Index (PRI) is an academic decision-support indicator developed for educational research at SLIIT Kandy, Sri Lanka. It is **not** a clinical psychological evaluation, psychiatric assessment, or medical diagnosis.

---

## 1. Project Overview

The **Psychosocial Intelligence Framework** is an individual research component in a multi-modal student attrition prediction system (SLIIT Kandy CDAP IT05). It ingests a 15-item Likert psychometric survey, quantifies student vulnerability on a continuous **Psychosocial Risk Index (PRI: 0 to 100)**, classifies students into tri-level risk tiers (**Low**, **Moderate**, **High**), computes local and global **SHAP attributions**, and delivers a clean, standardized export contract to a shared **Fusion Framework** alongside Academic, Engagement, and Environmental indices.

---

## 2. Directory Structure

```text
.
├── Makefile                      # One-command recipes (setup, train, eval, run)
├── requirements.txt              # Pinned Python 3.11 dependencies
├── config/
│   ├── default.yaml              # Central pipeline and API configuration
│   ├── schema.yaml               # 15-item Likert schema contract
│   └── synthetic.yaml            # Copula generation hyperparameters
├── data/
│   ├── raw/                      # Immutable survey input spreadsheets
│   ├── synthetic/                # Gaussian copula simulated cohort (N=3,000)
│   └── processed/                # Zero-leakage splits (train, val, test.lock)
├── src/
│   ├── data/                     # Schema validation, MICE imputation, generation
│   ├── features/                 # CTT factor calculation, reverse coding, PRI formula
│   ├── models/                   # Optuna-tuned model zoo (Ridge, Trees, Boosting)
│   ├── evaluation/               # Leaderboard, Wilcoxon tests, calibration audits
│   ├── explain/                  # TreeSHAP/LinearSHAP, sanity checks, narratives
│   └── api/                      # 14-endpoint FastAPI REST service
├── artifacts/
│   ├── models/                   # Serialized model weights & preprocessor (.joblib)
│   ├── metrics/                  # Reliability JSON and item-total correlations
│   ├── leaderboards/             # Benchmarks, champion JSON, and comparisons
│   ├── plots/                    # Diagnostic PNGs, confusion matrices, ROC curves
│   └── shap/                     # Beeswarm, waterfall, and factor importance CSVs
├── frontend/                     # React 18 + Vite + TypeScript + Tailwind UI console
├── docs/                         # Comprehensive viva documentation pack (00 to 18)
└── tests/                        # 69 automated pytest unit and integration tests
```

---

## 3. Quick Start: One-Command Execution

### Prerequisites
- Python 3.11
- Node.js 18+ and npm 9+

### One-Command Setup & Execution
Using the top-level `Makefile`:
```bash
# 1. Install all Python & Frontend dependencies
make install

# 2. Run end-to-end data generation, preprocessing, model training, and evaluation
make all

# 3. Launch both the backend API and frontend dashboard concurrently
make dev
```

### Manual Individual Commands
If running without `make`:
```bash
# Terminal 1: Launch Backend API (Port 8000)
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Launch Frontend Dashboard (Port 5173)
cd frontend
npm run dev
```
Open your browser at `http://localhost:5173`.

---

## 4. Automated Test Suites

- **Backend Pytest Suite (69 tests):**
  ```bash
  python -m pytest tests/ -q
  # Output: 69 passed, 9 warnings in ~16s
  ```
- **Frontend Vitest Suite (5 tests):**
  ```bash
  cd frontend
  npx vitest run
  # Output: 5 passed (100% pass rate)
  ```
- **Frontend Production Build:**
  ```bash
  cd frontend
  npm run build
  # Output: Clean build, 0 errors
  ```

---

## 5. How to Replace Synthetic Data with Real Survey Data

When institutional survey data is collected:
1. **Place Raw File:** Save the de-identified survey file as `data/raw/survey_responses.csv`.
2. **Execute Ingestion:** Run `python -m src.data.make_dataset`.
   - The contract validator (`src/data/schema.py`) checks all 15 items, removes straight-liners, fits MICE imputers on training folds, and creates new stratified splits.
3. **Retrain Model Zoo:** Run `python -m src.models.train`.
   - Optuna automatically tunes hyperparameters on the real student distributions.
4. **Evaluate & Update Champion:** Run `python -m src.evaluation.evaluate`.
   - Unlocks the sealed test set, runs pairwise Wilcoxon tests, and saves new champion bundles.
5. **Zero Code Changes:** The FastAPI service and React frontend dynamically load the updated serialized artifacts from `artifacts/models/final/` immediately.

---

## 6. Documentation Index (`/docs`)

- [`00_project_overview.md`](./docs/00_project_overview.md) — Research scope, role in Fusion, and non-negotiable rules.
- [`01_data_contract.md`](./docs/01_data_contract.md) — 15-item questionnaire schema and validation contract.
- [`02_synthetic_data.md`](./docs/02_synthetic_data.md) — Latent trait copula generation and realism injection.
- [`03_preprocessing_reliability.md`](./docs/03_preprocessing_reliability.md) — CTT metrics, Cronbach's $\alpha$, and MICE imputation.
- [`04_models_and_training.md`](./docs/04_models_and_training.md) — 10-model candidate zoo and Optuna Bayesian tuning.
- [`05_evaluation_leaderboard.md`](./docs/05_evaluation_leaderboard.md) — Scientific leaderboard, Wilcoxon tests, calibration error.
- [`06_explainability.md`](./docs/06_explainability.md) — TreeSHAP/LinearSHAP mathematical derivation and sanity checks.
- [`07_api.md`](./docs/07_api.md) — 14-endpoint FastAPI reference with `curl` examples.
- [`08_frontend.md`](./docs/08_frontend.md) — React 18 + Tailwind dashboard architecture and feature matrix.
- [`09_end_to_end_walkthrough.md`](./docs/09_end_to_end_walkthrough.md) — Stage-by-stage workflows and examiner defense cheat sheet.
- [`10_data_generation_explained.md`](./docs/10_data_generation_explained.md) — Copula mathematics, assumptions, and real data transition.
- [`11_model_selection_story.md`](./docs/11_model_selection_story.md) — Tournament results, Ridge vs. HistGradientBoosting, and trade-offs.
- [`12_metrics_cheat_sheet.md`](./docs/12_metrics_cheat_sheet.md) — Mathematical formulas, intuitions, and failure modes.
- [`13_shap_and_explainability.md`](./docs/13_shap_and_explainability.md) — Cooperative game theory axioms and audit verification.
- [`14_ethics_privacy_limitations.md`](./docs/14_ethics_privacy_limitations.md) — Non-clinical disclaimer, pseudonymization, and subgroup fairness.
- [`15_viva_qa.md`](./docs/15_viva_qa.md) — 40+ exhaustive viva examination defenses.
- [`16_presentation_script.md`](./docs/16_presentation_script.md) — 14-slide presentation script with 10-min and 5-min timings.
- [`17_demo_script.md`](./docs/17_demo_script.md) — Live demonstration walkthrough and emergency contingency plan.
- [`18_proposal_update_notes.md`](./docs/18_proposal_update_notes.md) — Line-by-line proposal corrections and Section 3 text insert.
