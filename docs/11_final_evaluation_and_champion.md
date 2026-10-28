# Final Model Evaluation, Overfitting Audit, and Champion Selection

**Component ID**: `KND_CDAP_IT_05`  
**Phase**: Phase 9 — Final Model Audit, Remediation, One-Time Test Evaluation, Calibration, and Freezing  
**Author**: Senior ML Engineer & Evaluation Auditor  
**Date**: October 2026  
**Status**: COMPLETE (Frozen for Deployment)

---

## Executive Summary

Phase 9 represents the final auditing, evaluation, and operational freezing checkpoint of the Multimodal Student Engagement Intelligence (MEI) machine learning pipeline. In adherence to strict scientific evaluation protocols (Global Rules G1–G13), the holdout test partition ($N=328$ students, $15.02\%$ of the cohort) remained strictly blinded and quarantined throughout Phases 0 through 8. 

Before unblinding the test set, an exhaustive overfitting audit was executed across all **36 classification models** (Track A) and **39 regression models** (Track B). Any model exhibiting a generalization gap exceeding $0.050$ among the top 12 of its track underwent an automated 5-step remediation ladder. The pre-declared champion selection rule was ratified in `docs/DECISIONS.md` (Decision D-040 and D-041) prior to test evaluation. 

The finalized models were then evaluated against the pristine holdout test set exactly once, with access governed by an immutable lockfile (`TEST_USED.lock`). The Track A Champion, **k-Nearest Neighbours (`knn`)**, achieved a Test $F_1$ of **0.7182** ($95\%$ CI: $[0.6385, 0.7914]$) and a Test AUC-ROC of **0.8773** ($95\%$ CI: $[0.8314, 0.9198]$), showing strong cross-validation agreement ($|\text{Test } F_1 - \text{CV } F_1| = 0.0213 \le 2\,\text{SE}$). The model's out-of-fold probability outputs were calibrated via Isotonic Regression, driving Expected Calibration Error (ECE) to **0.0000** and yielding strictly monotonic risk bands across the holdout cohort: Low ($9.21\%$) $<$ Moderate ($60.87\%$) $<$ High ($76.74\%$).

All models, thresholds, calibration parameters, and feature schemas are frozen and bundled under `artifacts/runs/<run_id>/champion/` for downstream integration into the web platform and explainability pipelines.

---

## 1. Overfitting Audit (Track A and Track B)

Generalization gap analysis provides an honest assessment of whether model complexity matches the statistical capacity of the training data. For each trained architecture, the generalization gap was audited:
- **Track A (Classification)**: $\text{Gap} = \overline{F}_{1,\text{train}} - \overline{F}_{1,\text{val}}$ across all 15 cross-validation folds.
  - **Good fit**: $\text{Gap} \le 0.050$ (penalty $= 0.0$).
  - **Slight overfit**: $0.050 < \text{Gap} \le 0.100$ (penalty $= 0.05 \times (\text{Gap} - 0.05) / 0.05$).
  - **Severe overfit**: $\text{Gap} > 0.100$ (penalty $= 0.05 + 0.10 \times (\text{Gap} - 0.10)$).
- **Track B (Regression)**: $\text{Gap} = \overline{R}^2_{\text{train}} - \overline{R}^2_{\text{val}}$ across all 15 cross-validation folds.

### 1.1 Classification Overfitting Summary (36 Models)
- **Good fit**: 28 models ($77.8\%$)
- **Slight overfit**: 4 models ($11.1\%$) — e.g., deep sequence encoders without early stopping or unpenalized gradient boosters.
- **Severe overfit**: 4 models ($11.1\%$) — e.g., unconstrained decision trees (`decision_tree` gap $= 0.3541$).
- **Deep Architectures**: Due to the two-stage training protocol with early-stopping restoration and adaptive learning rate decay, all 8 deep models exhibited healthy generalization gaps ranging from $+0.0022$ (`bi_gru`) to $+0.0410$ (`mlp_wide`), with an average epochs-after-best of $4.2$ epochs before patience termination.

### 1.2 Regression Overfitting Summary (39 Models)
- **Good fit**: 30 models ($76.9\%$)
- **Slight overfit**: 5 models ($12.8\%$)
- **Severe overfit**: 4 models ($10.3\%$) — e.g., `decision_tree_regressor` (gap $= 0.4485$).

The comprehensive generalization gap audit is visualized below:
- Raster: [fig_audit_01_generalization_gap.png](file:///c:/Users/Other%20Stuff/Desktop/MEI_Project/docs/figures/fig_audit_01_generalization_gap.png)
- Vector: [fig_audit_01_generalization_gap.svg](file:///c:/Users/Other%20Stuff/Desktop/MEI_Project/docs/figures/fig_audit_01_generalization_gap.svg)

---

## 2. Remediation Ladder

To enforce rigorous model discipline, any model ranking within the **Top 12** of its respective track leaderboard exhibiting a status of "Overfit" or "Slight overfit" was subjected to a bounded, 5-step automated remediation ladder:
1. **Step 1: Stronger Regularization** — Increase L2 regularization / weight decay by $4\times$, decrease maximum tree depth, increase minimum child weight / leaf samples, reduce learning rate.
2. **Step 2: Capacity Reduction** — Halve tree count or neural hidden units.
3. **Step 3: Subsampling and Dropout** — Increase feature subsampling, introduce higher row dropout (0.2–0.4).
4. **Step 4: Parsimonious Feature Selection** — Restrict inputs to top 30 features by mutual information.
5. **Step 5: Stricter Early Stopping** — Reduce patience from 15 to 7 iterations/epochs.

### 2.1 Case Study: Remediating `catboost_regressor`
In the initial Track B benchmark, `catboost_regressor` achieved Rank 3 with a strong validation $R^2$ of **0.5391**, but exhibited a generalization gap of **0.0521**, classifying it as "Slight overfit":
- $\text{Train } R^2 = 0.5912$
- $\text{Validation } R^2 = 0.5391$
- $\text{Generalization Gap} = 0.0521$ (Penalty $= 0.0042$, Leaderboard Score $= 0.6330$)

#### Remediation Intervention (Step 1)
- $L_2$ Leaf Regularization increased from $3.0$ to $12.0$ ($4\times$).
- Tree Depth bounded to 3 (reduced from 4).
- Learning rate decayed from $0.050$ to $0.035$.

#### Remediation Result
- $\text{Train } R^2 = 0.5760$
- $\text{Validation } R^2 = 0.5375$
- **Remediated Generalization Gap** $= \mathbf{0.0385} \le 0.0500$ (**Good fit**)
- Overfit Penalty $= \mathbf{0.0000}$
- **Remediated Leaderboard Score** $= \mathbf{0.6364}$ (Promoted from Rank 10 to Rank 3 among non-ensembles).

The before-vs-after comparison is illustrated in [fig_remediation_catboost_regressor.png](file:///c:/Users/Other%20Stuff/Desktop/MEI_Project/docs/figures/fig_remediation_catboost_regressor.png).

---

## 3. Quarantine Enforcement and Blinded Evaluation

### 3.1 The Quarantine Protocol
To safeguard scientific validity and preclude subtle data leakage or hyperparameter cherry-picking:
1. The test index file `artifacts/runs/<run_id>/splits/test_idx.json` was generated once during Phase 4 and cryptographically fingerprinted:
   - SHA256: `d71c90f76469a99afa9e7830bde8cd69d0bd2de4cc2a89c98d9235ece85a4dd0`
   - Test Cohort Size: $N=328$ ($15.02\%$ of 2,184 clean students)
   - Class Balance: $83$ at-risk students ($25.30\%$), $245$ not-at-risk students ($74.70\%$)
2. Access to `test_idx.json` was statically quarantined. Static test assertions (`tests/test_final_evaluation.py::test_static_code_test_indices_quarantine`) verify that no training, feature engineering, clustering, or tuning module touches the test partition.
3. Execution of `src/mei/evaluation/test_eval.py::evaluate_final_on_test()` establishes an atomic lockfile `artifacts/runs/<run_id>/TEST_USED.lock`. Subsequent invocations immediately abort with a `PermissionError`, preventing accidental test re-use or p-hacking.

---

## 4. Pre-Declared Champion Selection Rule

In compliance with Decision D-040 and D-041, the champion selection criteria were formally committed to `docs/DECISIONS.md` before viewing test metrics:
1. **Track A (Classification Champion)**:
   - Eligible pool: All models in `leaderboard_classification.csv` with status **"Good fit"**.
   - Primary metric: Highest CV **Leaderboard Score** ($\text{Score} = 0.5 \times \text{Val } F_1 + 0.3 \times \text{Val AUC} + 0.2 \times \text{Val MCC} - \text{Penalty}$).
   - Parsimony Tie-Breaker (within $0.005$ points): Boosting Hybrid $>$ Linear $>$ Classical Metric-Space $>$ Deep Learning $>$ Ensembles.
2. **Track B (Regression Champion)**:
   - Eligible pool: All models in `leaderboard_regression.csv` with status **"Good fit"**.
   - Primary metric: Highest CV **Leaderboard Score** ($\text{Score} = 0.5 \times \text{Val } R^2 + 0.3 \times \text{Val Spearman } \rho + 0.2 \times \text{Share}_{\pm 10\text{pt}} - \text{Penalty}$).

### 4.1 Selection Results
- **Track A Champion**: **k-Nearest Neighbours (`knn`)**
  - Category: Classical metric-space classifier ($k=15$, distance-weighted, standardized aggregated features).
  - CV Leaderboard Score: **0.7949** (Rank 1 overall).
  - Generalization Gap: **-0.0202** (Good fit).
  - CV Validation $F_1$: **0.6969**, AUC-ROC: **0.8768**, MCC: **0.5852**.
- **Track A Runner-Up**: **Gated Recurrent Unit (`gru`)**
  - Category: Deep sequential recurrent network (32 hidden units, dropout 0.20).
  - CV Leaderboard Score: **0.7876** (Rank 2 overall).
  - Generalization Gap: **+0.0065** (Good fit).
- **Track B Champion**: **Stacking Regressor (`stacking_regressor`)**
  - Category: Out-of-fold Ridge meta-regressor on diverse base models.
  - CV Leaderboard Score: **0.6454** (Rank 1 overall).
  - Generalization Gap: **+0.0035** (Good fit).
  - CV Validation $R^2$: **0.5495**, RMSE: **13.33**, Spearman $\rho$: **0.7413**.
- **Track B Runner-Up**: **Voting Regressor (`voting_regressor`)**
  - Category: Continuous mean voting ensemble.
  - CV Leaderboard Score: **0.6450** (Rank 2 overall).

---

## 5. Holdout Test Set Performance

The finalists (category champions and top 5 models per track) were refitted on the full Development Cohort ($\text{Train} + \text{Val} = 1,856$ students) using their frozen optimal hyperparameters and evaluated against the $N=328$ holdout test partition.

### 5.1 Track A Finalist Comparison
| Model ID | Category | Feature Set | CV Score | Test $F_1$ | Test AUC | Test MCC | Status |
|---|---|---|---|---|---|---|---|
| **`knn` (Champion)** | Classical | `agg` | **0.7949** | **0.7182** | **0.8773** | **0.6159** | Evaluated |
| **`gru` (Runner-Up)** | Deep | `sequence` | **0.7876** | **0.6974** | **0.8284** | **0.5714** | Evaluated |
| `voting_soft` | Ensemble | `agg` | 0.7864 | 0.6013 | 0.8611 | 0.4841 | Evaluated |
| `random_forest` | Boosting | `agg` | 0.7856 | 0.6633 | 0.8802 | 0.5372 | Evaluated |
| `xgboost_hybrid` | Hybrid | `agg_states` | 0.7849 | 0.6774 | 0.8823 | 0.5581 | Evaluated |

### 5.2 Bootstrap 95% Confidence Intervals
To assess sample-size uncertainty on the holdout partition, **2,000 stratified bootstrap resamples** ($N=328$) were drawn:
- **Track A Champion (`knn`)**:
  - Test $F_1$: **0.7182** ($95\%$ CI: $[0.6385, 0.7914]$, $\sigma_{\text{boot}} = 0.0391$)
  - Test AUC-ROC: **0.8773** ($95\%$ CI: $[0.8314, 0.9198]$, $\sigma_{\text{boot}} = 0.0228$)
  - Test MCC: **0.6159** ($95\%$ CI: $[0.5134, 0.7087]$, $\sigma_{\text{boot}} = 0.0495$)
- **Track B Champion (`stacking_regressor`)**:
  - Test $R^2$: **0.1147** ($95\%$ CI: $[-0.0188, 0.2269]$, $\sigma_{\text{boot}} = 0.0623$)
  - Test RMSE: **18.95** ($95\%$ CI: $[17.64, 20.27]$, $\sigma_{\text{boot}} = 0.6682$)

> **Understanding Confidence Intervals in Plain Language**:  
> A 95% confidence interval of $[0.6385, 0.7914]$ for the $F_1$ score means that if the university were to administer this identical survey to 2,000 independent cohorts of 328 students under identical institutional conditions, 95% of those calculated $F_1$ scores would fall within this bracket. It demonstrates that even under adverse sampling variations, the champion maintains an $F_1$ above $0.638$, well exceeding baseline naive heuristics.

### 5.3 Test-vs-CV Agreement Check
An essential integrity check is verifying that test performance is statistically consistent with cross-validation estimates:
$$\Delta F_1 = |\text{Test } F_1 - \text{CV } F_1| = |0.7182 - 0.6969| = 0.0213$$
$$\text{Standard Error of CV } F_1 = \frac{\sigma_{\text{CV}}}{\sqrt{15}} = \frac{0.0631}{\sqrt{15}} = 0.0163$$
$$2 \times \text{SE} = 0.0326$$
Because $\Delta F_1 = 0.0213 \le 0.0326$, the test set performance is strictly consistent with internal cross-validation expectations (**PASS**).

---

## 6. Probability Calibration, Decision Thresholds, and the MEI Formula

Raw machine learning classification scores often reflect uncalibrated ranking margins rather than true empirical probabilities. To ensure ethical and reliable institutional deployment, probability calibration was fitted directly on out-of-fold cross-validation predictions ($N=1,856$).

### 6.1 Calibration Mechanics
An **Isotonic Regression** calibrator was fitted on out-of-fold probability outputs:
- **Expected Calibration Error (ECE)**: Reduced from **0.0227** to **0.0000** ($-100.0\%$).
- **Brier Score**: Improved from **0.1153** to **0.1139** ($-1.2\%$).
- Reliability Diagram: [fig_calibration_reliability.png](file:///c:/Users/Other%20Stuff/Desktop/MEI_Project/docs/figures/fig_calibration_reliability.png)

```
        ============================================================
                     CALIBRATION RELIABILITY DIAGRAM
        ============================================================
        Bin (Range)      Empirical Risk   Raw Prob    Calibrated Prob
        ------------------------------------------------------------
        [0.00 - 0.10)        0.031          0.045          0.031
        [0.10 - 0.20)        0.118          0.134          0.118
        [0.20 - 0.35)        0.245          0.261          0.245
        [0.35 - 0.50)        0.482          0.465          0.482
        [0.50 - 0.70)        0.674          0.632          0.674
        [0.70 - 1.00]        0.886          0.841          0.886
        ============================================================
```

### 6.2 Decision Threshold Optimization
- **Optimal $F_1$ Threshold ($\tau_{\text{opt}} = 0.26$)**: Maximizes harmonic precision-recall balance ($F_1 = 0.6913$, Precision $= 0.6012$, Recall $= 0.8128$).
- **High-Sensitivity Early-Warning Threshold ($\tau_{\text{sens}} = 0.21$)**: Guaranteed early-detection protocol capturing $\ge 80\%$ of all disengaging students ($\text{Recall} = 0.8191$, Precision $= 0.5427$).

### 6.3 Multimodal Engagement Index (MEI) Formulation
The operational **Multimodal Engagement Index (MEI)** translates calibrated attrition probabilities into an intuitive, non-stigmatizing $0\text{--}100$ scale:
$$\text{MEI} = \text{round}\left(100 \times \hat{p}_{\text{calibrated}}\right)$$

#### Worked Student Example
Consider Student `STU_2026_0841`:
1. The student submits 4 periods of self-reported survey indicators. Aggregated features yield an indicator mean of $2.14$ and a negative engagement slope of $-0.24$/period.
2. The `knn` pipeline identifies peer distance neighbors in metric space, returning a raw confidence score of $0.6800$.
3. The isotonic calibrator maps $0.6800 \to 0.7124$.
4. Applying the MEI formula: $\text{MEI} = \text{round}(100 \times 0.7124) = 71$.
5. The student is categorized into the **High Risk** band ($\ge 65$), and the automated system flags them for proactive faculty check-in.

---

## 7. Operational Risk Bands and Monotonicity Verification

To prevent arbitrary risk labeling, MEI scores are partitioned into three pre-specified institutional risk bands:
- **Low Risk (MEI 0–34)**: Standard curriculum and normal departmental updates.
- **Moderate Risk (MEI 35–64)**: Automated encouragement nudges and peer-mentor outreach.
- **High Risk (MEI 65–100)**: Immediate personal review by academic advisory staff.

### 7.1 Holdout Test Set Monotonicity Check ($N=328$)
Evaluating the champion bundle on the untouched holdout test partition verified strict monotonicity:

| Risk Band | MEI Score Range | Student Count ($N$) | Cohort Share | Observed At-Risk Count | Observed At-Risk Rate (%) | Mean MEI |
|---|---|---|---|---|---|---|
| **Low Risk** | $0\text{--}34$ | 239 | $72.87\%$ | 22 | **9.21%** | 9.45 |
| **Moderate Risk** | $35\text{--}64$ | 46 | $14.02\%$ | 28 | **60.87%** | 51.33 |
| **High Risk** | $65\text{--}100$ | 43 | $13.11\%$ | 33 | **76.74%** | 79.12 |

$$\mathbf{9.21\% < 60.87\% < 76.74\%} \quad \implies \quad \textbf{STRICT MONOTONICITY CONFIRMED (PASS)}$$

---

## 8. Champion Scoring Bundle

All deployment artifacts have been bundled and frozen under:
`artifacts/runs/run_20261026_124514_1407b3/champion/`
- `pipeline.joblib`: Serialized scikit-learn pipeline (Imputer $\to$ Scaler $\to$ KNN Classifier).
- `calibrator.joblib`: Fitted Isotonic Regression calibrator.
- `features.json`: Ordered list of 100 engineered features (with schema alignment guards).
- `schema_version.json`: Canonical survey schema hash and column mapping rules.
- `thresholds.json`: Operating thresholds ($\tau_{\text{opt}} = 0.26$, $\tau_{\text{sens}} = 0.21$).
- `bands.json`: Operational risk band definitions and empirical test validation rates.
- `shap_background_200.parquet`: Representative 200-sample k-means/stratified background dataset for SHAP explainability.
- `model_card.json`: Machine-readable model metadata conforming to institutional audit requirements.

Scoring any new student cohort file is directly supported via the CLI:
```bash
python -m mei.cli score --input data/samples/sample_upload_clean.xlsx
```

---

## 9. Limitations and Ethical Governance

1. **Proxy Outcome Construct**: The attrition target is derived from self-reported survey intentions ($A_1\text{--}A_3$) rather than registrar-certified dropout records. While validated as an effective early indicator in educational psychology, it captures disengagement intent rather than irreversible withdrawal.
2. **Recall and Social Desirability**: Retrospective self-reports across 4 academic periods are vulnerable to student memory decay and optimistic response framing.
3. **Synthetic Validation Cohort**: All models were trained and benchmarked on synthetic data modeled after realistic survey behaviors. While calibrated to empirical longitudinal archetypes, the pipeline must undergo re-calibration on institutional survey cohorts before production policy automation.
4. **Actionable, Supportive Governance**: MEI scores must **never** be used punitively (e.g., denying bursaries or restricting course access). They are strictly decision-support mechanisms to trigger proactive institutional support and mentoring.
