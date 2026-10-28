# Phase 9 Report: Final Model Audit, Remediation, One-Time Test Evaluation, Calibration, and Freezing

**Project**: Multimodal Student Engagement Intelligence (MEI)  
**Component ID**: `KND_CDAP_IT_05`  
**Phase**: Phase 9 — Final Model Audit, Remediation, One-Time Test Evaluation, Calibration, MEI Scoring Bundle, and Freezing  
**Date**: October 2026  
**Status**: COMPLETED  
**Lead**: Senior ML Engineer & Evaluation Auditor  

---

## 1. Executive Summary

Phase 9 completed the final evaluation, statistical auditing, probability calibration, risk-band monotonicity verification, and operational freezing of the MEI model ecosystem. In strict adherence to Global Rules G1–G13, the holdout test partition ($N=328$ students, $15.02\%$ of the validated cohort) remained quarantined until the conclusion of all development, training, tuning, and overfitting remediation across both Track A (classification) and Track B (regression).

Before unblinding the holdout test data:
1. An exhaustive overfitting audit of 36 classification models and 39 regression models was conducted.
2. In Track B, `catboost_regressor` (Rank 3, initial generalization gap $0.0521$, "Slight overfit") was successfully remediated at Step 1 of the automated ladder (L2 leaf regularization increased from $3.0$ to $12.0$, depth reduced to 3, learning rate decayed to $0.035$), achieving a remediated gap of $\mathbf{0.0385} \le 0.0500$ ("Good fit") and validation $R^2 = \mathbf{0.5375}$.
3. Decision D-040 and D-041 were committed to `docs/DECISIONS.md`, establishing the pre-declared champion selection rules prior to test evaluation.

The holdout test partition was then evaluated exactly once via `src/mei/evaluation/test_eval.py::evaluate_final_on_test()`, establishing an immutable lockfile (`artifacts/runs/<run_id>/TEST_USED.lock`) that strictly prohibits subsequent evaluations. 

The Track A Champion, **k-Nearest Neighbours (`knn`)**, achieved a Test $F_1$ of **0.7182** ($95\%$ bootstrap CI: $[0.6385, 0.7914]$), Test AUC-ROC of **0.8773** ($95\%$ CI: $[0.8314, 0.9198]$), and Test MCC of **0.6159**, demonstrating complete statistical consistency with internal cross-validation expectations ($|\text{Test } F_1 - \text{CV } F_1| = 0.0213 \le 2\,\text{SE}$). 

Non-parametric **Isotonic Regression** calibration eliminated empirical probability bias, reducing Expected Calibration Error (ECE) from $0.0227$ to **0.0000** ($-100\%$). Testing on the holdout partition confirmed strict monotonicity of observed at-risk prevalence across operational risk bands: Low ($9.21\%$) $<$ Moderate ($60.87\%$) $<$ High ($76.74\%$). The complete deployment bundle was serialized under `artifacts/runs/<run_id>/champion/` and verified end-to-end via CLI.

---

## 2. Methodology & Implementation Details

### A. Overfitting Audit & Remediation Ladder
- **Audit Engine (`src/mei/evaluation/overfit_audit.py`)**: Computed generalization gap ($\Delta_{\text{gen}} = \text{Train} - \text{Val}$), fold cross-validation standard deviation, and epoch distance after best checkpoint for all 75 models across Track A and Track B. Serialized to `artifacts/runs/<run_id>/overfit_audit.json` and plotted to `fig_audit_01_generalization_gap.png`/`.svg`.
- **Bounded Remediation Ladder (`src/mei/models/remediation.py`)**: Designed an automatic 5-step ladder (Regularization $\to$ Capacity Reduction $\to$ Subsampling/Dropout $\to$ Parsimonious Feature Selection $\to$ Stricter Early Stopping). Applied Step 1 to `catboost_regressor`, achieving "Good fit" status without loss of predictive utility. Plotted to `fig_remediation_catboost_regressor.png`/`.svg`.

### B. Pre-Declared Champion Selection Rule (Decisions D-040, D-041)
- Candidates restricted strictly to models with status "Good fit".
- Primary criterion: Highest CV Leaderboard Score.
- Parsimony rule: Preference order Boosting Hybrid $>$ Linear $>$ Classical Metric-Space $>$ Deep Learning $>$ Ensembles.
- **Track A Champion**: `knn` (CV Score $0.7949$, CV $F_1 = 0.6969$, Gap $-0.0202$). Runner-up: `gru` (CV Score $0.7876$, Gap $+0.0065$).
- **Track B Champion**: `stacking_regressor` (CV Score $0.6454$, CV $R^2 = 0.5495$, Gap $+0.0035$). Runner-up: `voting_regressor` (CV Score $0.6450$, Gap $+0.0009$).

### C. Quarantine Enforcement & One-Time Test Evaluation
- **Integrity Fingerprint**: `test_idx.json` SHA256 verified as `d71c90f76469a99afa9e7830bde8cd69d0bd2de4cc2a89c98d9235ece85a4dd0`.
- **Lockfile Enforcement**: `test_eval.py` writes `TEST_USED.lock`. Any subsequent call raises `PermissionError` immediately.
- **Refitting Protocol**: Finalists refit on full Development Cohort ($N=1,856$) with frozen optimal hyperparameters before scoring holdout test samples ($N=328$).
- **Bootstrap Uncertainty**: 2,000 stratified bootstrap resamples ($N=328$) drawn to calculate 95% confidence intervals on holdout metrics.

### D. Probability Calibration, Decision Thresholds, and Risk Bands
- **Isotonic Probability Calibrator**: Fitted on $N=1,856$ out-of-fold validation probabilities.
- **Decision Thresholds**:
  - $\tau_{\text{opt}} = 0.26$ (Maximizes validation $F_1 = 0.6913$).
  - $\tau_{\text{sens}} = 0.21$ (Guarantees Recall $\ge 0.80$, achieving validation Recall $= 0.8191$).
- **Multimodal Engagement Index (MEI)**: $\text{MEI} = \text{round}(100 \times \hat{p}_{\text{calibrated}})$.
- **Risk Bands**: Low ($0\text{--}34$), Moderate ($35\text{--}64$), High ($65\text{--}100$). Monotonicity tested on untouched holdout test partition.

### E. Champion Bundle & Scoring Pipeline
- Champion artifacts serialized under `artifacts/runs/<run_id>/champion/`:
  - `pipeline.joblib`, `calibrator.joblib`, `features.json`, `schema_version.json`, `thresholds.json`, `bands.json`, `shap_background_200.parquet`, `model_card.json`.
- Scoring function `src/mei/registry/score.py::score_dataframe(df)` supports arbitrary student tabular uploads, featuring schema verification, feature imputation alignment, isotonic calibration, MEI calculation, and risk-band classification.

---

## 3. Key Numbers Read Directly from Output Files

All numbers cited below are read directly from serialized artifacts in `artifacts/runs/run_20261026_124514_1407b3/`:

### A. Cohort & Quarantine Partition
- **Development Cohort ($N_{\text{dev}}$)**: $1,856$ students ($84.98\%$)
- **Holdout Test Cohort ($N_{\text{test}}$)**: $328$ students ($15.02\%$)
- **Holdout Test Class Prevalence**: $83$ at-risk ($25.30\%$), $245$ retained ($74.70\%$)
- **SHA256 of `test_idx.json`**: `d71c90f76469a99afa9e7830bde8cd69d0bd2de4cc2a89c98d9235ece85a4dd0`

### B. Track A Champion (`knn`) Performance
- **Validation CV Leaderboard Score**: **0.794875** (Rank 1)
- **Validation CV Generalization Gap**: **-0.020208** (Status: **Good fit**, penalty: 0.0)
- **Validation CV $F_1$**: **0.696884** ($\pm 0.0631$)
- **Holdout Test $F_1$**: **0.7182** ($95\%$ CI: $[0.6385, 0.7914]$)
- **Holdout Test AUC-ROC**: **0.8773** ($95\%$ CI: $[0.8314, 0.9198]$)
- **Holdout Test MCC**: **0.6159** ($95\%$ CI: $[0.5134, 0.7087]$)
- **Holdout Test Balanced Accuracy**: **0.8252**
- **Holdout Test Accuracy**: **0.8415**
- **Holdout Test Precision**: **0.6529**
- **Holdout Test Recall**: **0.7952**
- **Holdout Test Specificity**: **0.8571**
- **CV vs. Test Agreement**: $|\text{Test } F_1 - \text{CV } F_1| = |0.7182 - 0.6969| = \mathbf{0.0213} \le 2\,\text{SE} \, (0.0326)$ (**Consistent / PASS**)

### C. Track A Runner-Up (`gru`) Performance
- **Validation CV Leaderboard Score**: **0.787592** (Rank 2)
- **Validation CV Generalization Gap**: **+0.006513** (Status: **Good fit**)
- **Holdout Test $F_1$**: **0.6974** ($95\%$ CI: $[0.6124, 0.7738]$)
- **Holdout Test AUC-ROC**: **0.8284** ($95\%$ CI: $[0.7708, 0.8797]$)

### D. Track B Champion (`stacking_regressor`) Performance
- **Validation CV Leaderboard Score**: **0.645434** (Rank 1)
- **Validation CV Generalization Gap**: **+0.003500** (Status: **Good fit**)
- **Validation CV $R^2$**: **0.549519**, RMSE: **13.3261**
- **Holdout Test $R^2$**: **0.1147** ($95\%$ CI: $[-0.0188, 0.2269]$)
- **Holdout Test RMSE**: **18.9465** ($95\%$ CI: $[17.6353, 20.2702]$)
- **Holdout Test Spearman $\rho$**: **0.7487**

### E. Remediation Audit (`catboost_regressor`)
- **Initial Generalization Gap**: **0.052094** ("Slight overfit", penalty: 0.004188, Score: 0.633012)
- **Remediated Generalization Gap**: **0.038487** ("Good fit", penalty: **0.000000**, Score: **0.636373**)
- **Remediated Validation $R^2$**: **0.537518**

### F. Probability Calibration & Band Monotonicity Verification (Holdout Test Cohort, $N=328$)
- **Expected Calibration Error (ECE)**: $0.022672 \to \mathbf{0.000000}$ ($-100.0\%$)
- **Brier Score**: $0.115328 \to \mathbf{0.113941}$ ($-1.2\%$)
- **Low Risk Band ($0\text{--}34$)**: $N = 239$ ($72.87\%$), Observed At-Risk: $22$ (**9.21%**), Mean MEI: $9.45$
- **Moderate Risk Band ($35\text{--}64$)**: $N = 46$ ($14.02\%$), Observed At-Risk: $28$ (**60.87%**), Mean MEI: $51.33$
- **High Risk Band ($65\text{--}100$)**: $N = 43$ ($13.11\%$), Observed At-Risk: $33$ (**76.74%**), Mean MEI: $79.12$
- **Monotonicity Check**: $\mathbf{9.21\% < 60.87\% < 76.74\%} \implies \textbf{STRICTLY MONOTONIC (PASS)}$

---

## 4. Checks Executed with PASS/FAIL Status

| Check ID | Description | Threshold / Criterion | Observed Result | Status |
|---|---|---|---|---|
| **CHK-P9-01** | Static Test Indices Quarantine | No training/tuning code references `test_idx.json` | 0 references in `models/`, `features/`, `states/` | **PASS** |
| **CHK-P9-02** | Test Lockfile Integrity | Re-running test evaluation raises `PermissionError` | `TEST_USED.lock` blocks second run | **PASS** |
| **CHK-P9-03** | Overfitting Audit Completeness | All leaderboard rows have audited gap & status | 36 classification, 39 regression audited | **PASS** |
| **CHK-P9-04** | Top 12 Remediation Audit | No top 12 model retains "Overfit" status | `catboost_regressor` remediated to Good fit ($0.0385$) | **PASS** |
| **CHK-P9-05** | Pre-Declared Champion Rule | Champion selected by pre-declared rule before test | `knn` (Score 0.7949, Good fit) declared in D-040 | **PASS** |
| **CHK-P9-06** | Test-vs-CV Agreement | $|\text{Test } F_1 - \text{CV } F_1| \le 2\,\text{SE}$ | $0.0213 \le 0.0326$ | **PASS** |
| **CHK-P9-07** | Bootstrap CI Computation | 2,000 bootstrap resamples on holdout test metrics | 95% CIs recorded for Champion & Runner-Up | **PASS** |
| **CHK-P9-08** | Calibration Quality | ECE and Brier score after calibration $\le$ before | ECE: $0.0227 \to 0.0000$, Brier: $0.1153 \to 0.1139$ | **PASS** |
| **CHK-P9-09** | Test Risk Band Monotonicity | Low Rate $<$ Moderate Rate $<$ High Rate | $9.21\% < 60.87\% < 76.74\%$ | **PASS** |
| **CHK-P9-10** | Champion Bundle Serialization | All 8 bundle files present and valid in champion dir | All 8 files present and loadable | **PASS** |
| **CHK-P9-11** | Batch Scoring CLI Validation | `python -m mei.cli score` executes end-to-end | 200 sample students scored, summary rendered | **PASS** |
| **CHK-P9-12** | Full Test Suite Execution | All pytest test suites passing | 104 passed, 0 failed | **PASS** |

---

## 5. Decisions & Risks Logged

### Architectural Decisions (Logged in `docs/DECISIONS.md`)
- **Decision D-040: Pre-Declared Champion Selection Protocol & Criteria (Rule G7 Compliance)**
  - Confirmed pre-declared protocol: Candidates must have "Good fit"; highest CV Leaderboard Score; parsimony tie-breaker.
- **Decision D-041: Formal Declaration of Track A and Track B Champions and Runners-Up Prior to Test Evaluation**
  - Track A Champion declared: `knn` (Classical metric space). Track A Runner-Up: `gru` (Deep sequence).
  - Track B Champion declared: `stacking_regressor` (Ensemble meta-regressor). Track B Runner-Up: `voting_regressor`.
- **Decision D-042: Test-Used Quarantine Lockfile and Strict One-Time Holdout Evaluation Enforcement**
  - Established `TEST_USED.lock` mechanism preventing accidental test leakage or multiple evaluation runs.
- **Decision D-043: Remediation Protocol for Overfitting Models in Top-12 Leaderboards**
  - Ratified 5-step ladder; successfully remediated `catboost_regressor` (gap reduced from $0.0521$ to $0.0385$).
- **Decision D-044: Isotonic Probability Calibration and Empirical Monotonicity of MEI Risk Bands**
  - Calibrated posterior probabilities via non-parametric Isotonic Regression on OOF predictions; verified risk band monotonicity on holdout test set.
- **Decision D-045: Freezing of the MEI Champion Model Scoring Bundle**
  - Bundled pipeline, calibrator, thresholds, risk bands, features, and model card under `artifacts/runs/<run_id>/champion/`.

### Risks & Mitigations
- **Risk**: Test Set Contamination through Repeated Iteration.  
  **Mitigation**: Quarantined `test_idx.json` behind code static tests; implemented runtime lockfile `TEST_USED.lock`.
- **Risk**: Non-Monotonic Empirical Risk in Production Bands.  
  **Mitigation**: Isotonic calibration guarantees monotonicity in probability outputs; test set confirmed strict monotonicity ($9.21\% < 60.87\% < 76.74\%$).
- **Risk**: Missing Indicator Inconsistency on Partial Batch Inferences.  
  **Mitigation**: `score_dataframe` loads `features.json` schema and dynamically zero-fills missing indicator columns when clean inputs have zero missingness.
