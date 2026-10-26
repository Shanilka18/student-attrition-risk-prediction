# Project Rules and Specification: Psychosocial Intelligence Framework

## 1. Project Overview
- **Project Title:** Psychosocial Intelligence Framework (Psychosocial Risk Index - PRI component)
- **Host Institution:** SLIIT Kandy, Sri Lanka
- **Context:** Component of a multi-factor student attrition prediction system.
- **Role & Output:**
  - Quantifies student psychosocial risk on a continuous scale: **PRI from 0 to 100**.
  - Tri-level risk classification: **Low Risk**, **Moderate Risk**, **High Risk**.
  - Generates subscale / per-factor domain scores.
  - Computes local and global **SHAP (SHapley Additive exPlanations)** attribution scores.
  - Delivers a clean, standardized export contract for downstream integration with the **Fusion Framework** alongside Academic, Engagement, and Environmental indices.

---

## 2. Input Data Specification
- **Instrument:** 15-item Psychosocial Questionnaire administered on a 5-point Likert scale (1 = Strongly Disagree ... 5 = Strongly Agree).
- **Latent Factors & Items:**
  1. **Stress (Risk Factor: Higher score = higher risk):**
     - `S1`, `S2`, `S3`
  2. **Motivation (Protective Factor: Higher score = lower risk):**
     - `M1`, `M2`, `M3`
  3. **Belongingness (Protective Factor: Higher score = lower risk):**
     - `B1`, `B2`, `B3`
  4. **Self-Efficacy (Protective Factor: Higher score = lower risk):**
     - `SE1`, `SE2`, `SE3`
  5. **Peer Support (Protective Factor: Higher score = lower risk):**
     - `PS1`, `PS2`, `PS3`
- **Factor Scoring Formula:**
  - Raw factor score = arithmetic mean of its 3 constituent items.
  - Reverse orientation rule: Protective factors are reversed when mapped to risk dimensions ($Item_{rev} = 6 - Item_{raw}$ or equivalent normalized inversion).

---

## 3. Non-Negotiable Engineering & Scientific Rules

1. **Quality Bar & Reproducibility:**
   - Production-grade, publication-ready research code.
   - Fixed random seeds across all libraries (`numpy`, `random`, `torch`, `scikit-learn`).
   - Purely configuration-driven via YAML (`config/default.yaml`).
   - Comprehensive type hints (`typing`), NumPy/Google style docstrings, structured logging (`loguru` / standard `logging`), and unit tests (`pytest`) for all critical paths.

2. **Technology Stack:**
   - **Language:** Python 3.11.
   - **Data & Core ML:** `pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `optuna`, `shap`, `pingouin`.
   - **Deep Learning (Tabular):** `pytorch` (compact tabular architectures, e.g., Tabular MLP / ResNet embeddings).
   - **Visualization:** `matplotlib`, `seaborn`, `plotly`.
   - **Backend API:** `FastAPI`, `uvicorn`, `pydantic`.
   - **Frontend UI:** React 18+, Vite, TypeScript, Tailwind CSS, Recharts, Lucide Icons.
   - All versions strictly pinned in `requirements.txt` and `package.json`.

3. **Strict Data Leakage Prevention:**
   - Absolute zero data leakage policy.
   - Scalers, imputers, normalizers, and encoders must be fitted strictly on training folds.
   - Model selection, hyperparameter tuning, and threshold selection performed solely via nested / stratified K-fold cross-validation and validation sets.
   - The holdout test set is sealed and touched exactly once for the final verification of the shortlisted model.

4. **Phase Documentation Standard:**
   - Every phase must produce:
     1. Fully working, tested code.
     2. Automated tests or verification scripts that execute successfully.
     3. A dedicated markdown document in `/docs` containing:
        - **Simple Explanation:** An ELI5 everyday analogy for non-technical stakeholders.
        - **Technical Explanation:** Rigorous mathematical, statistical, and architectural justification for academic viva examiners.

5. **Empirical Truthfulness & Traceability:**
   - Zero fabrication of benchmark figures or evaluation metrics.
   - All documented tables, figures, and UI metrics must trace directly to persistent artifacts in `/artifacts` (`metrics/`, `leaderboards/`, `plots/`, `shap/`).

6. **Synthetic Data Integrity:**
   - Clearly label all synthetically synthesized records, simulations, and benchmark datasets as synthetic.
   - Explicitly disclaim that synthetic data cannot prove real-world epidemiological or clinical validity.

7. **Ethics & Research Governance:**
   - **Research Indicator Disclaimer:** The PRI is purely a statistical research and educational decision-support indicator, NOT a psychiatric or psychological clinical diagnosis.
   - This disclaimer must be prominently visible across UI dashboards, API responses, and technical documentation.
   - Strict de-identification: records are identified solely by pseudonymized Study IDs (`study_id`).

8. **Phase-by-Phase Execution Protocol:**
   - Work strictly one phase at a time.
   - Conclude each phase with a standardized summary table containing:
     - Files created / updated
     - Exact commands executed
     - What was verified
     - Prerequisites for the next phase
   - Pause execution and await explicit user confirmation before initiating subsequent phases.

---

## 4. Directory Structure Blueprint
```text
.
├── AGENTS.md                  # This file (project operational instructions)
├── Makefile                   # Top-level operational recipes (data, train, evaluate, etc.)
├── requirements.txt           # Pinned Python 3.11 dependencies
├── config/
│   └── default.yaml           # Centralized experiment & pipeline configuration
├── data/
│   ├── raw/                   # Immutable raw data (questionnaire survey responses)
│   ├── synthetic/             # Generative / simulated synthetic survey data
│   └── processed/             # Cleaned, validated, split data frames
├── src/
│   ├── data/                  # Ingestion, validation, generation, splitting
│   ├── features/              # Factor calculation, Likert reversals, scale aggregations
│   ├── models/                # Baseline, tree ensembles, tabular neural nets, tuning
│   ├── evaluation/            # Metrics, calibration, cross-validation runners
│   ├── explain/               # TreeSHAP, KernelSHAP, local & global attributions
│   ├── api/                   # FastAPI routes, schemas, Fusion export contracts
│   └── utils/                 # Logging, seeding, pathing, IO helpers
├── artifacts/
│   ├── models/                # Serialized model weights & pipelines (.joblib, .pt)
│   ├── metrics/               # Evaluation metrics, confusion matrices (JSON/CSV)
│   ├── leaderboards/          # Model comparison tables (CSV, Markdown)
│   ├── plots/                 # Diagnostic curves, distributions, feature importance
│   └── shap/                  # Summary plots, force plots, explanation artifacts
├── frontend/                  # React + Vite + TypeScript interactive dashboard
├── docs/                      # Scientific and technical documentation
├── tests/                     # Automated unit and integration test suite
└── notebooks/                 # Exploratory data analysis & prototyping
```
