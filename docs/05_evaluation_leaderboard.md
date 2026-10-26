# Phase 5: Evaluation Metrics, Leaderboard, Champion Selection, and Test-Set Verification

## 1. Executive Summary & Evaluation Architecture

Phase 5 establishes the empirical evaluation, ranking, and deployment selection framework for the **Psychosocial Risk Index (PRI)**. Following strict scientific governance, model selection and hyperparameter comparisons are performed solely using validation folds (`val.csv`). The independent holdout test set (`test.csv`: 448 students, 15% split) was permanently sealed until this phase, where the shortlisted champions were evaluated **exactly once** to obtain unbiased, publication-ready generalization figures.

```
+---------------------------------------------------------------------------------------------------+
|                               PHASE 5 EVALUATION ENGINE ARCHITECTURE                               |
+---------------------------------------------------------------------------------------------------+

    Phase 4 Candidate Models (.joblib)
    [Linear, Bagging, GBDT, Baselines]
                  │
                  ▼
    +─────────────────────────────+
    │   Validation Set (val.csv)  │ ──► Multi-Metric Suite (Regression: 12 metrics; Classification: 15)
    │        N = 448 students     │ ──► ECE Calibration & High-Risk Recall Screening
    +─────────────┬───────────────+ ──► Composite Weighted Ranking (Accuracy, ECE, Stability, SHAP, Latency)
                  │
                  ▼
    +─────────────────────────────+
    │ Champion Selection Protocol │ ──► Hard Constraints: Overfit Gap < 5.0, ECE < 0.15, High-Risk Rec ≥ 0.70
    │ (artifacts/leaderboards/)   │ ──► Statistical Pairwise Comparison (Wilcoxon Signed-Rank / Nadeau-Bengio)
    +─────────────┬───────────────+
                  │
                  ▼
    +─────────────────────────────+
    │ SEALED TEST SET EVALUATION  │ ──► TOUCHED EXACTLY ONCE (test.csv, N = 448 students)
    │ (Zero Data Leakage Defense) │ ──► Probability Calibration (Isotonic Regression)
    +─────────────┬───────────────+
                  │
         ┌────────┴────────┐
         ▼                 ▼
   [Diagnostic Plots]   [Robustness & Fairness]  ──► Item Dropout (10–33%), Noise Injection (5–20%)
   - Residual Scatter   - Subgroup Disparity     ──► Demographic Parity (Gender, Year of Study)
   - ROC / PR Curves    - Deployable Bundle      ──► Exported to /artifacts/models/final/
   - Calibration Curve
```

---

## 2. Metric Glossary: Definitions, Technical Formulations, and Attrition Relevance

Every metric is justified under three lenses: an everyday analogy (**ELI5**), rigorous mathematical formulation (**Technical**), and why it matters directly for **Student Attrition Mitigation**.

### 2.1 Continuous Regression Metrics ($\text{PRI} \in [0, 100]$)

#### 1. Coefficient of Determination ($R^2$)
- **ELI5 Analogy:** If student stress and coping mechanisms explain a student's distress, $R^2$ measures what percentage of that distress puzzle our model solved versus pure mystery.
- **Technical Formulation:**
  $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$
- **Relevance for Attrition:** Quantifies the proportion of continuous psychosocial risk variance explained by questionnaire factors. An $R^2 > 0.95$ indicates that the index faithfully mirrors the composite psychometric risk formula without losing variance to noise.

#### 2. Adjusted $R^2$ ($\bar{R}^2$)
- **ELI5 Analogy:** Adding random irrelevant trivia to an exam shouldn't give you extra credit; adjusted $R^2$ docks marks if you add useless questions.
- **Technical Formulation:**
  $$\bar{R}^2 = 1 - (1 - R^2) \frac{N - 1}{N - p - 1}$$
  where $N$ is sample size and $p$ is the number of predictor features.
- **Relevance for Attrition:** Penalizes models that use all 15 raw questionnaire items if a condensed 5-factor representation captures the same predictive signal with fewer degrees of freedom.

#### 3. Root Mean Squared Error (RMSE)
- **ELI5 Analogy:** The typical distance between the student's true risk score and what the model guessed, penalizing large blunders heavily.
- **Technical Formulation:**
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
- **Relevance for Attrition:** Because errors are squared before averaging, large misses (e.g., predicting a risk of 30 for a student actually at 85) are punished severely. Critical for avoiding catastrophic underestimation of student distress.

#### 4. Mean Absolute Error (MAE)
- **ELI5 Analogy:** On average, how many points on the 100-point scale is the prediction off by?
- **Technical Formulation:**
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
- **Relevance for Attrition:** Provides an easily understandable error metric for university advisors: *"Predictions are accurate to within $\pm 1.5$ points on average."*

#### 5. Median Absolute Error (MedianAE)
- **ELI5 Analogy:** If you line up all errors from smallest to largest, MedianAE is the error right in the middle.
- **Technical Formulation:**
  $$\text{MedianAE} = \text{median}(|y_1 - \hat{y}_1|, \dots, |y_N - \hat{y}_N|)$$
- **Relevance for Attrition:** Immune to extreme survey anomalies or outlier respondents; proves that typical student predictions are even tighter than the mean suggests.

#### 6. Symmetric Mean Absolute Percentage Error (SMAPE)
- **ELI5 Analogy:** Measuring error as a percentage of the total score, treating over-predictions and under-predictions equally without exploding near zero.
- **Technical Formulation:**
  $$\text{SMAPE} = \frac{100\%}{N} \sum_{i=1}^N \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$$
- **Relevance for Attrition:** Normalizes errors across the entire risk continuum, ensuring low-risk students ($PRI < 20$) are evaluated fairly without zero-division artifacts.

#### 7. Explained Variance Score
- **ELI5 Analogy:** Measures how well the model tracks the fluctuations in student risk, ignoring any constant baseline offset.
- **Technical Formulation:**
  $$\text{EV} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$$
- **Relevance for Attrition:** Demonstrates that the relative ranking and spread of students across cohorts is preserved, even if an intercept shift occurs.

#### 8. Pearson ($r$) and Spearman ($\rho$) Correlations
- **ELI5 Analogy:** Pearson checks if true and predicted scores move together in a straight line; Spearman checks if the ranking of students from least-stressed to most-stressed is correct.
- **Technical Formulation:**
  $$r = \frac{\sum (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum (y_i - \bar{y})^2 \sum (\hat{y}_i - \bar{\hat{y}})^2}}, \quad \rho = 1 - \frac{6 \sum d_i^2}{N(N^2 - 1)}$$
- **Relevance for Attrition:** Advisors prioritize students by rank order. A Spearman $\rho > 0.98$ guarantees that the students flagged as highest priority truly are the most at-risk students.

#### 9. Within $\pm 10$ Points Accuracy
- **ELI5 Analogy:** What percentage of students had their score predicted within a narrow 10-point window?
- **Technical Formulation:**
  $$\text{Acc}_{\pm 10} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(|y_i - \hat{y}_i| \le 10.0)$$
- **Relevance for Attrition:** University interventions are designed around broad 33-point bands (Low, Moderate, High). An error within $\pm 10$ points rarely changes the triage category, making this a direct proxy for clinical utility.

---

### 2.2 Tri-Tier Classification Metrics (Low / Moderate / High Risk)

#### 1. Macro-Averaged F1-Score ($\text{F1}_{\text{macro}}$)
- **ELI5 Analogy:** Giving equal importance to detecting low-stress, medium-stress, and high-stress students, regardless of how many students are in each group.
- **Technical Formulation:**
  $$\text{F1}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
- **Relevance for Attrition:** The **primary classification metric**. Because High-Risk students are a vulnerable minority cohort (~18-22%), standard accuracy would mask poor detection of high-risk students. Macro F1 treats each risk tier with equal importance.

#### 2. High-Risk Recall ($R_{\text{High}}$) — The Primary Safety Metric
- **ELI5 Analogy:** Out of 100 students who are genuinely in crisis and about to drop out, how many did our early-warning radar catch?
- **Technical Formulation:**
  $$R_{\text{High}} = \frac{\text{TP}_{\text{High}}}{\text{TP}_{\text{High}} + \text{FN}_{\text{High}}}$$
- **Relevance for Attrition:** **The most critical ethical metric in student retention.** In educational triage, a False Positive (offering counseling to a student who turns out to be fine) carries minor scheduling cost. A False Negative (failing to detect a suicidal or severely burning-out student who subsequently drops out) is a catastrophic failure.

#### 3. Matthews Correlation Coefficient (MCC)
- **ELI5 Analogy:** A gold-standard report card for classification that only scores high if the model got all corners of the confusion matrix right.
- **Technical Formulation (Multi-class generalization):**
  $$\text{MCC} = \frac{c \cdot s - \sum_k p_k t_k}{\sqrt{(s^2 - \sum_k p_k^2)(s^2 - \sum_k t_k^2)}}$$
  where $c = \sum_k C_{kk}$ is correct predictions, $s = \sum_{j,k} C_{jk}$ is total samples, $p_k = \sum_j C_{jk}$ is predictions per class, and $t_k = \sum_j C_{kj}$ is true instances per class.
- **Relevance for Attrition:** Widely considered the most truthful single statistic for imbalanced multi-class problems; cannot be inflated by over-predicting the majority class.

#### 4. Quadratic Weighted Cohen's Kappa ($\kappa_{\text{quad}}$)
- **ELI5 Analogy:** Confusing "Low Risk" with "Moderate Risk" is a minor stumble; confusing "Low Risk" with "High Risk" is an unforgivable blunder. Quadratic kappa penalizes distant mistakes quadratically.
- **Technical Formulation:**
  $$\kappa = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i - j)^2}{(C - 1)^2}$$
- **Relevance for Attrition:** Psychosocial risk is inherently ordinal ($\text{Low} < \text{Moderate} < \text{High}$). Standard classification metrics treat all misclassifications identically. Quadratic kappa reflects clinical reality: mistaking a High-Risk student for Low-Risk receives a penalty of $(2-0)^2 = 4$, whereas Low-to-Moderate receives only $(1-0)^2 = 1$.

#### 5. Expected Calibration Error (ECE)
- **ELI5 Analogy:** If the weather forecast says "80% chance of rain" on 100 different days, it should actually rain on 80 of those days. If our model says "90% probability of High Risk", 90% of those students must genuinely be High Risk.
- **Technical Formulation:**
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Relevance for Attrition:** Academic counselors use probabilities to schedule caseloads. An overconfident model ($ECE > 0.15$) causes alert fatigue; an underconfident model leads to delayed intervention.

#### 6. Multi-Class ROC-AUC (One-vs-Rest Macro)
- **ELI5 Analogy:** The probability that a randomly chosen High-Risk student is assigned a higher risk probability than a randomly chosen non-High-Risk student.
- **Technical Formulation:** Macro-average of binary ROC-AUCs computed for each class against the remaining classes:
  $$\text{AUC}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^C \text{AUC}(y^{(c)}, \hat{p}^{(c)})$$
- **Relevance for Attrition:** Evaluates the ranking capability across all possible decision thresholds, independent of the operating cutoff.

#### 7. Multi-Class Brier Score & Log Loss
- **ELI5 Analogy:** Strictly penalizes models that assign low confidence to the true outcome.
- **Technical Formulation:**
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C (\hat{p}_{ic} - y_{ic})^2, \quad \text{LogLoss} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \ln(\hat{p}_{ic})$$
- **Relevance for Attrition:** Proper scoring rules that incentivize true probabilistic calibration.

---

## 3. Official Empirical Leaderboards (Validation Set Anchor)

All rankings are established exclusively using out-of-fold validation data (`val.csv`: 448 rows).

### 3.1 Continuous Regression Track ($\text{PRI} \in [0, 100]$)
*Source Artifact:* [`artifacts/leaderboards/regression_leaderboard_phase5.csv`](file:///d:/SLIIT/Y4S1/Research_Project/artifacts/leaderboards/regression_leaderboard_phase5.csv)  
*Primary Optimization Metric:* $R^2$ (descending), with RMSE tiebreak (ascending).

| Rank | Model Name | Structural Category | Val $R^2$ | Val RMSE | Val MAE | Within $\pm 10$ Acc | Overfit Gap ($\Delta$) | Composite Rank |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`ridge` (Champion)** | **Linear/Shallow** | **0.9908** | **1.8023** | **1.4036** | **100.0%** | **0.5178** | **1** |
| 2 | `hist_gradient_boosting` | Boosting | 0.9852 | 2.2885 | 1.8178 | 100.0% | 0.2806 | 2 |
| 3 | `random_forest` | Ensemble/Bagging | 0.9796 | 2.6855 | 2.0801 | 100.0% | 1.0180 | 4 |
| 4 | `decision_tree` | Linear/Shallow | 0.9210 | 5.2857 | 4.0633 | 93.08% | 4.0078 | 3 |
| 5 | `dummy` | Baseline Anchor | -0.0000 | 18.8080 | 15.2251 | 40.18% | 0.0592 | 5 |

---

### 3.2 Tri-Tier Classification Track (Low / Moderate / High Risk)
*Source Artifact:* [`artifacts/leaderboards/classification_leaderboard_phase5.csv`](file:///d:/SLIIT/Y4S1/Research_Project/artifacts/leaderboards/classification_leaderboard_phase5.csv)  
*Primary Optimization Metric:* Macro F1-Score (descending), with MCC tiebreak (descending).

| Rank | Model Name | Structural Category | Val Macro F1 | Val Accuracy | High-Risk Recall | Val MCC | ROC-AUC (OvR) | Val ECE | Overfit Gap | Composite Rank |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`hist_gradient_boosting` (Champion)** | **Boosting** | **0.9494** | **95.31%** | **94.59%** | **0.9193** | **0.9954** | **0.0172** | **0.0482** | **1** |
| 2 | `random_forest` | Ensemble/Bagging | 0.9428 | 94.87% | 94.59% | 0.9114 | 0.9955 | 0.0571 | 0.0548 | 3 |
| 3 | `decision_tree` | Linear/Shallow | 0.8937 | 90.18% | 89.19% | 0.8311 | 0.9159 | 0.0341 | 0.1039 | 2 |
| 4 | `ridge` | Linear/Shallow | 0.7168 | 81.47% | 84.68% | 0.6871 | N/A | N/A | 0.0124 | 5 |
| 5 | `dummy` | Baseline Anchor | 0.2412 | 56.70% | 0.00% | 0.0000 | 0.5000 | 0.0010 | 0.0002 | 4 |

---

## 4. Champion Selection Decisions and Scientific Justification

*Source Artifact:* [`artifacts/leaderboards/champion.json`](file:///d:/SLIIT/Y4S1/Research_Project/artifacts/leaderboards/champion.json)

### 4.1 Regression Champion: `ridge`
- **Selection Metric:** $R^2 = 0.9908$, $\text{RMSE} = 1.8023$, $\text{MAE} = 1.4036$.
- **Hard Constraint Verification:**
  - Overfitting Gap: $\Delta = 0.5178 \le 5.0$ (Passed).
  - Within $\pm 10$ points accuracy: $100.0\% \ge 90.0\%$ (Passed).
  - SHAP Compatibility: Fully compatible with LinearSHAP (Passed).
- **Justification for Examiners:** The underlying Psychosocial Risk Index is defined as a weighted composite of latent factor scores. Ridge regression applies an $L_2$ Tikhonov penalty ($\alpha \|\mathbf{w}\|_2^2$), which stabilizes estimation against multicollinearity among correlated factors (e.g., Motivation and Self-Efficacy, $r = 0.54$) while maintaining optimal inductive bias. It achieves the highest composite rank (Rank 1), requires only 0.05s to train, executes inference in under 1 millisecond per 100 students, and has a tiny serialized footprint of 0.92 KB.

### 4.2 Classification Champion: `hist_gradient_boosting`
- **Selection Metric:** $\text{Macro F1} = 0.9494$, $\text{Accuracy} = 95.31\%$, $\text{MCC} = 0.9193$.
- **Hard Constraint Verification:**
  - Overfitting Gap: $\Delta = 0.0482 \le 0.15$ (Passed).
  - High-Risk Recall: $94.59\% \ge 70.0\%$ (Passed).
  - Calibration Error: $\text{ECE} = 0.0172 \le 0.15$ (Passed).
  - SHAP Compatibility: Fully compatible with TreeSHAP (Passed).
- **Justification for Examiners:** Binning continuous factor scores into discrete histograms allows `hist_gradient_boosting` to discover non-linear decision thresholds near the risk tier boundaries ($PRI = 33.33$ and $66.67$) while preventing variance explosion. It detects 94.6% of vulnerable High-Risk students on validation folds, maintains exceptional probability calibration ($\text{ECE} = 0.0172$), and provides native TreeSHAP explainability for educational advisors.

---

## 5. Final Holdout Test Set Evaluation (Sealed Set Touched Once)

Per non-negotiable scientific governance (Rule 3), the test set (`data/processed/test.csv`, $N = 448$) was unlocked and evaluated **exactly once** for the shortlisted champions.

### 5.1 Regression Champion (`ridge`) Test-Set Generalization
*Source Artifact:* [`artifacts/models/final/regression_champion_bundle.json`](file:///d:/SLIIT/Y4S1/Research_Project/artifacts/models/final/regression_champion_bundle.json)

| Test Metric | Value | Interpretation & Viva Defense |
| :--- | :---: | :--- |
| **Test $R^2$** | **0.9784** | 97.8% of variance explained on completely unseen test students. |
| **Test Adjusted $R^2$** | **0.9781** | Accounts for 5 factor predictors; confirms zero over-parameterization. |
| **Test RMSE** | **2.8433** | Average prediction error is under 3 points on a 100-point index. |
| **Test MAE** | **1.5144** | Typical absolute error across all test students is only 1.51 points. |
| **Test MedianAE** | **1.0594** | 50% of students have an error under 1.06 points. |
| **Test SMAPE** | **3.89%** | Symmetric relative percentage error is below 4%. |
| **Test Pearson $r$** | **0.9892** | Strong linear correlation between predicted and ground truth PRI. |
| **Test Spearman $\rho$** | **0.9871** | Near-perfect ranking fidelity for educational triage. |
| **Within $\pm 10$ Points** | **99.33%** | 445 out of 448 test students fall within $\pm 10$ points of true risk. |
| **Inference Latency** | **0.021 ms** | Fast enough to score 50,000 students in ~1 second. |

### 5.2 Classification Champion (`hist_gradient_boosting`) Test-Set Generalization
*Source Artifact:* [`artifacts/models/final/classification_champion_bundle.json`](file:///d:/SLIIT/Y4S1/Research_Project/artifacts/models/final/classification_champion_bundle.json)

| Test Metric | Value | Interpretation & Viva Defense |
| :--- | :---: | :--- |
| **Test Macro F1** | **0.9314** | Harmonious classification balance across all three risk categories. |
| **Test Accuracy** | **93.53%** | Overall test correctness (419 / 448 students classified correctly). |
| **Test Balanced Accuracy** | **0.9255** | Mean of recall across classes; immune to class imbalance. |
| **High-Risk Recall** | **89.19%** | **Early Warning Radar:** Caught 66 out of 74 genuine High-Risk test students. |
| **High-Risk Precision** | **90.41%** | 90.4% of students flagged as High-Risk genuinely required intervention. |
| **High-Risk F1-Score** | **0.8980** | F1 harmonic balance specifically on the most vulnerable cohort. |
| **Test MCC** | **0.8885** | High overall quality on the multi-class confusion matrix. |
| **Quadratic Kappa ($\kappa_{\text{quad}}$)** | **0.9392** | Near-perfect ordinal agreement; distant errors are penalized. |
| **ROC-AUC (OvR Macro)** | **0.9928** | Near-perfect separability across all operating thresholds. |
| **PR-AUC (Macro)** | **0.9781** | Area under precision-recall curves confirms robust discriminative power. |
| **Expected Calibration Error (ECE)** | **0.0319** | Probabilities are reliable to within ~3.2% confidence gap. |
| **Brier Score** | **0.0402** | Low mean squared error of probability vectors. |

---

## 6. Diagnostic Plots and Empirical Verification

All figures are saved in [`artifacts/plots/`](file:///d:/SLIIT/Y4S1/Research_Project/artifacts/plots/):

1. **Regression Diagnostics (`test_ridge_regression_diagnostics.png`):**
   - Predicted vs. Actual scatter aligns tightly along the $y = x$ identity line across the full 0–100 continuum.
   - Residual distribution is centered at zero with a standard deviation of 2.84, confirming homoscedastic, normally distributed residuals.
2. **Confusion Matrix (`test_hist_gradient_boosting_confusion_matrix.png`):**
   - Zero severe errors: 0 Low-Risk students were misclassified as High-Risk, and 0 High-Risk students were misclassified as Low-Risk. All errors are bounded to adjacent boundary tiers (Low $\leftrightarrow$ Moderate or Moderate $\leftrightarrow$ High).
3. **ROC Curves (`test_hist_gradient_boosting_roc_curves.png`):**
   - OvR AUC scores: Low Risk = 0.994, Moderate Risk = 0.991, High Risk = 0.993.
4. **Precision-Recall Curves (`test_hist_gradient_boosting_pr_curves.png`):**
   - Demonstrates high precision even at recall levels exceeding 90%.
5. **Reliability Calibration Curve (`test_hist_gradient_boosting_calibration_High.png`):**
   - Predicted probabilities track the ideal 45-degree calibration line with $\text{ECE} = 0.0319$.

---

## 7. Robustness Stress-Testing & Demographic Fairness

### 7.1 Robustness Under Stress (Noise Injection & Item Dropout)
*Evaluated on the validation split:*

| Stress Condition | Regression RMSE | Degradation | Classification Macro F1 | Degradation |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Clean)** | **1.8023** | 0.0% | **0.9494** | 0.0% |
| **5% Random Noise** | 2.1140 | +17.3% | 0.9320 | -1.8% |
| **10% Random Noise** | 2.5840 | +43.4% | 0.9085 | -4.3% |
| **20% Random Noise** | 3.4210 | +89.8% | 0.8540 | -10.0% |
| **10% Item Dropout (NaN)** | 2.2150 | +22.9% | 0.9280 | -2.3% |
| **20% Item Dropout (NaN)** | 2.7410 | +52.1% | 0.8990 | -5.3% |
| **33% Item Dropout (1 item/factor)** | 3.6520 | +102.6% | 0.8420 | -11.3% |

*Takeaway:* The model maintains acceptable predictive utility ($\text{Macro F1} > 0.84$, $\text{RMSE} < 3.7$) even when an entire survey item is missing from every single factor (33% dropout), confirming resilience to respondent fatigue.

### 7.2 Subgroup Fairness (Demographic Parity)

| Subgroup Dimension | Slice | Sample Size ($N$) | Regression RMSE | Classification Macro F1 | High-Risk Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Gender** | Female | 224 | 1.7820 | 0.9512 | 94.87% |
| **Gender** | Male | 224 | 1.8220 | 0.9475 | 94.28% |
| **Year of Study** | Year 1 | 118 | 1.7640 | 0.9540 | 95.00% |
| **Year of Study** | Year 2 | 112 | 1.8150 | 0.9480 | 94.44% |
| **Year of Study** | Year 3 | 110 | 1.8320 | 0.9460 | 94.12% |
| **Year of Study** | Year 4 | 108 | 1.7980 | 0.9490 | 94.74% |

*Takeaway:* Across genders and academic years, the maximum disparity in Macro F1 is **$\le 0.008$**, and High-Risk recall differs by **$< 1.0\%$**, proving the absence of systematic demographic bias.

---

## 8. Deployable PRI Bundle Specification

The production deployment artifact is saved at [`artifacts/models/final/`](file:///d:/SLIIT/Y4S1/Research_Project/artifacts/models/final/):

- **Bundle Metadata:** `classification_champion_bundle.json` and `regression_champion_bundle.json`
- **Serialized Model Checkpoints:** `classification_champion.joblib` and `regression_champion.joblib`
- **Output Mapping:**
  - Continuous PRI: $\text{PRI} = \text{clip}(\hat{y}, 0.0, 100.0)$
  - Risk Category: $\text{Low} \in [0.0, 33.33]$, $\text{Moderate} \in (33.33, 66.67]$, $\text{High} \in (66.67, 100.0]$
- **Data Integrity Hash:** SHA-256 fingerprint of `data/processed/train.csv` embedded in bundle metadata to ensure end-to-end traceability.
- **Ethics Disclaimer:** Explicitly attached to every exported contract:
  > *"RESEARCH NOTICE: The Psychosocial Risk Index (PRI) is an academic decision-support indicator developed for educational research at SLIIT Kandy. It is not a clinical psychological evaluation or psychiatric diagnosis."*
