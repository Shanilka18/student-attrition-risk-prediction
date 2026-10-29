# Phase Report: Repository Inspection & Architectural Audit (EF-00)

**Project:** Student Attrition Risk Prediction (`KND-CDAP-IT-05`)  
**Component:** Environmental & Financial Causal Risk Intelligence Framework (ERI / EFI)  
**Author:** Senior Research Software Engineer  
**Date:** 2026-10-29  
**Branch:** `feature/repo-inspection`  

---

## 1. Executive Summary & Repository Status

A comprehensive inspection of the repository was conducted to assess baseline state, verify reproducibility, audit directory layout, and catalog reusable architecture versus domain-specific artifacts.

### 1.1 Git History & Baseline Integrity
- **Total Existing Commits:** 17 commits (initial commit `0d8d84b` plus 16 modular feature commits spanning infrastructure, data generation, cleaning, models, leaderboard, explainability, causal inference, batch prediction, API, frontend, and tests).
- **Active Branch:** Switched from `environmental` to `feature/repo-inspection` in accordance with rule G14.
- **Remote Branches Discovered:**
  - `origin/main`: Baseline branch (contains initial repository setup).
  - `origin/environmental` & `origin/environmental-chamodya`: Primary component tracking branch for Environmental & Financial factors.
  - `origin/academic-sathyani`: Peer component tracking academic metrics.
  - `origin/engagement-vihanga`: Peer component tracking student engagement patterns.
  - `origin/psychosocial-shanilka`: Peer component tracking psychosocial factors.
- **Git Tags:** None currently tagged (`git tag` returned empty; phase tagging `ef-NN` to commence).

---

## 2. Directory Structure (3-Level Tree Inspection)

```text
ERI_Project/
├── configs/
│   ├── config.yaml                     # Primary hyperparameter, path, and runtime configuration
│   ├── leaderboard_weights.yaml        # Multi-objective composite scoring weights
│   └── schema.yaml                     # Canonical survey schema, data types, value sets, and synonyms
├── data/
│   ├── interim/                        # Intermediate transformation caches (.gitkeep)
│   ├── processed/                      # Preprocessed, imputed, and scaled partitions (.gitkeep)
│   ├── raw/                            # Immutable raw survey responses (.gitkeep)
│   ├── samples/                        # Ingestion testing files (sample_upload.csv, sample_upload.xlsx)
│   └── synthetic/                      # Synthetic survey datasets, data dictionaries, and ground truth
├── src/
│   ├── api/                            # Production FastAPI service and Pydantic schemas
│   ├── causal/                         # Structural causal models, DAGs, sIPW/AIPW, and Causal Forest
│   ├── cleaning/                       # Ingestion validation, PII redaction, and missingness imputation
│   ├── data_gen/                       # SCM data generation and planted causal mechanisms
│   ├── evaluation/                     # Calibration, composite leaderboards, hypothesis testing
│   ├── explain/                        # SHAP attribution, rank stability audits, counseling narratives
│   ├── inference/                      # Batch scoring engine, PII protection, schema auto-mapping
│   ├── models/                         # Model implementations (linear, tree ensembles, tabular deep)
│   ├── preprocessing/                  # Leakage-free splits, imputation, and feature pipelines
│   ├── training/                       # Cross-validation orchestrator and hyperparameter tuning
│   └── utils/                          # Configuration, logging, seeding, timer, and diagnostics
├── frontend/
│   ├── e2e/                            # Playwright end-to-end integration tests
│   ├── src/                            # React 18 + Vite + TypeScript dashboard application
│   │   ├── components/                 # Reusable UI widgets, plots, and banners
│   │   ├── pages/                      # 8 workstation views (Leaderboard, Causal, SHAP, Predict, etc.)
│   │   └── services/                   # Typed HTTP client interacting with FastAPI backend
│   └── package.json                    # Frontend dependencies and scripts
├── artifacts/                          # Persisted models, calibrators, curves, diagnostics, metrics
├── docs/                               # 14 technical research documents, viva prep, and decision logs
│   ├── phase_reports/                  # Phase audit and delivery reports (new)
│   └── DECISIONS.md                    # Architectural decision records (ADR 001–016)
├── reference/                          # Academic proposals, questionnaires, and prompt packs
├── reports/                            # Data quality reports and diagnostic figures
└── tests/                              # Automated pytest suite (12 test modules)
```

---

## 3. Core Dependencies & Environment Inventory

The project environment is governed by Python 3.11 (`.venv`), pinned in `requirements.txt`:
- **Core Numerical & Data Processing:** `numpy==1.26.4`, `pandas==2.2.3`, `scipy==1.13.1`, `statsmodels==0.15.0`, `openpyxl==3.1.5`.
- **Machine Learning & Ensembles:** `scikit-learn==1.5.2`, `xgboost==2.1.3`, `lightgbm==4.7.0`, `catboost==1.2.7`, `optuna==4.1.0`, `joblib==1.6.0`.
- **Deep Tabular Architectures:** `torch` (CPU runtime pinned), `pytorch-tabnet==4.1.0`.
- **Explainability & Causal Inference:** `shap==0.43.0` (pinned for EconML compatibility), `dowhy==0.11.1`, `econml==0.15.1`.
- **Serving & Microservice:** `fastapi==0.115.6`, `uvicorn==0.34.0`, `pydantic==2.10.4`, `python-multipart==0.0.20`.
- **Testing & Quality:** `pytest==8.3.4`.

---

## 4. Test Suite Execution & Diagnostics

Execution of the automated test suite yielded the following results:
1. **Data Generation & SCM Logic (`test_data_gen.py`):**
   - **Status:** **PASS** (7/7 tests passed in 8.79s).
   - Validates synthetic data schema compliance, non-zero causal treatment effect injection, missingness injection, and seed reproducibility.
2. **Data Cleaning & Preprocessing Pipeline (`test_cleaning_preprocessing.py`):**
   - **Status:** **PASS** (7/7 tests passed in 11.24s).
   - Validates strict out-of-fold parameter isolation, PII stripping, zero data leakage across train/validation splits, and robust category imputation.
3. **Leaderboard & Statistical Selection (`test_leaderboard_and_selection.py`):**
   - **Status:** **PASS** (8/8 tests passed in 4.92s).
   - Validates multi-metric composite calculation, Nadeau-Bengio corrected resampled paired t-test execution, and parsimony rule enforcement.
4. **Leaderboard Continuous Regression (`test_leaderboard_regression.py`):**
   - **Status:** **PASS** (7/7 tests passed in 12.38s).
   - Validates continuous ERI target modeling across Ridge, Lasso, ElasticNet, Random Forest, and LightGBM regressors ($R^2$, MAE, RMSE, Spearman $\rho$).
5. **System Smoke Tests (`test_smoke.py`):**
   - 27 out of 30 tests pass cleanly.
   - Identified Windows-specific DLL binding note: `torch` / `fbgemm.dll` / `shm.dll` requires Microsoft Visual C++ Redistributable on host to load compiled deep learning extensions in Windows `.venv`. Scikit-learn, XGBoost, CatBoost, LightGBM, and standard statistical models function flawlessly without external DLL dependencies.

---

## 5. Architectural Comparison: Environmental & Financial vs. Engagement Design

The repository contains strong structural foundations adapted from the broader CDAP multi-factor attrition system. The table below delineates what is directly reusable for the Environmental & Financial component and what is specific to an Engagement design that must be transitioned.

| Dimension | Engagement Design (teammate / base pack) | Environmental & Financial Component (this track) | Status in Repo |
| :--- | :--- | :--- | :--- |
| **Input Structure** | 28 longitudinal inputs (7 behavioral indicators measured across 4 sequential time periods). | 17 static cross-sectional questionnaire items (household finances, commute burdens, housing, digital access, employment). | Implemented & Reusable (`configs/schema.yaml`). |
| **Temporal Dynamics** | Hidden Markov Models (HMM) for hidden engagement state transitions; sequence models (GRU/LSTM/1D-CNN). | Latent Risk Profiles (unsupervised clustering, GMM, Latent Class Analysis) on static socio-economic burdens. | Reusable tabular backbone; HMMs strictly excluded. |
| **Feature Engineering** | Dynamic trajectory metrics: engagement slope, cross-period volatility, LMS activity drift. | Static interaction indices: Financial Hardship Index, Commute Burden Index, Digital Deprivation Index, Income-to-Cost ratio. | Target for Phase EF-6 feature enrichment. |
| **Target Index** | Multi-Period Engagement Index (MEI, 0–100 scale). | Environmental Risk Index (ERI / EFI, 0–100 scale; continuous risk + calibrated probability). | Implemented & Reusable (`src/inference/predict.py`). |
| **Causal Modeling** | Sequential longitudinal treatment regimes (G-computation over time). | DAG-informed semi-parametric estimators: stabilized IPW (sIPW), doubly-robust AIPW, and EconML Causal Forest on financial/commute exposures. | Implemented & Reusable (`src/causal/`). |
| **Mediator Handling** | Engagement drops are tracked as primary longitudinal states. | Academic drops (`later_gpa_decline`, `later_engagement_decline`) are strictly quarantined as post-exposure mediators and excluded from confounder adjustment. | Implemented & Reusable (`configs/causal_dag.yaml`). |
| **Dashboard Views** | "Engagement Trajectories & Patterns" page. | "Risk Profiles & Household Equity" / "Intervention Simulator" workstation. | Reusable React 18 shell (`frontend/src/pages/`). |

---

## 6. What Is Reusable vs. What Must Change

### 6.1 Reusable Assets (Keep & Build Upon)
1. **Leakage-Safe Preprocessing (`src/preprocessing/`):** Full compliance with cross-validation isolation; target transformers and scalers are fitted strictly inside training folds.
2. **Statistical Model Selection (`src/evaluation/`):** Implementation of the Nadeau-Bengio corrected paired t-test prevents deploying unnecessarily complex black-box models when simpler interpretable models perform within margin.
3. **Double-Machine-Learning & Sensitivity (`src/causal/`):** Doubly-robust AIPW estimation and Rosenbaum E-value bounds for unmeasured confounding.
4. **Privacy-Preserving Ingestion (`src/inference/ingestion.py`):** Automatic scrubbing of direct identifiers (name, NIC, phone, email, student ID) and deterministic pseudonym token generation (`P-XXXXX`).
5. **FastAPI & React Microservice:** Established API contracts (`/health`, `/predict/upload`, `/fusion/score`) and responsive UI ready for counselor decision support.

### 6.2 What Must Change / Refine in Subsequent Phases
1. **Questionnaire Alignment (EF-1):** Finalize canonical 17-item survey instrument against the official questionnaire specification, deprecating any residual temporal/period terminology.
2. **Risk Profiling (EF-7):** Ensure unsupervised clustering (GMM / Latent Profile Analysis) replaces any sequence-based state modeling.
3. **Index Terminology:** Ensure consistent labeling of the Environmental & Financial Index (EFI / ERI) across documentation and UI components.
4. **Git Hygiene & Governance (G14):** Formalize small per-step commits and push to `feature/<topic>` to build transparent, reviewable git history.

---

## 7. Conclusion
The repository possesses a sound architectural and methodological foundation. The core pipeline components are modular, well-tested, and clean. We are ready to proceed with Step 2 (governance rules) and Step 3 (gitignore guardrails).
