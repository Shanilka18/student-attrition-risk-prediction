# 12 — Metrics Cheat Sheet: Mathematical Definitions, Intuitions, & Attrition Context

## 1. Simple Explanation (Age-5 Level)

When teachers grade tests, they don't just say "good job"—they have different ways of checking:
- Did you get the exact right number?
- If you made a mistake, was it a tiny mistake (off by 1) or a gigantic mistake (off by 50)?
- When looking for students who need urgent help, did you accidentally miss anyone who was crying for help?

Different math rulers check different things. We use multiple rulers so we never trick ourselves into thinking a model is good when it is secretly failing our students.

---

## 2. Regression Track Metrics (Continuous PRI 0 to 100)

### 1. Coefficient of Determination ($R^2$)
- **Formula:** 
  $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
- **Intuition:** Proportion of variance in the Psychosocial Risk Index explained by the model compared to predicting the cohort mean.
- **When it Misleads:** $R^2$ can be artificially high if outliers exist at extreme ends, and does not tell you if predictions are systematically biased.
- **Why Chosen:** Standard standardized metric across machine learning benchmarks ($R^2 = 0.9908$ for Ridge champion; `artifacts/leaderboards/regression_leaderboard.md`).

### 2. Root Mean Squared Error (RMSE)
- **Formula:** 
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
- **Intuition:** Average prediction error on the original 0–100 scale, heavily penalizing large blunders.
- **When it Misleads:** Sensitive to occasional extreme outliers; a single large error inflates RMSE.
- **Why Chosen:** Serves as our primary tiebreak metric ($1.8023$ points for Ridge). An RMSE of $1.8$ on a 100-point scale indicates exceptional precision.

### 3. Mean Absolute Error (MAE) & MedianAE
- **Formula:** 
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|, \quad \text{MedianAE} = \text{median}(|y_1 - \hat{y}_1|, \dots, |y_n - \hat{y}_n|)$$
- **Intuition:** Robust, intuitive measure of typical error. For our Ridge champion, $\text{MAE} = 1.4036$ and $\text{MedianAE} = 1.0935$ (`artifacts/models/final/regression_champion_bundle.json`).
- **Why Chosen:** Direct real-world meaning: on average, predicted PRI is off by less than $1.5$ percentage points.

### 4. "Within $\pm10$ Points" Accuracy
- **Formula:** 
  $$\text{Acc}_{\pm10} = \frac{1}{n} \sum_{i=1}^n \mathbb{I}(|y_i - \hat{y}_i| \le 10)$$
- **Intuition:** Percentage of students whose predicted risk index falls within an actionable 10-point tolerance window.
- **Why Chosen:** Highly practical for academic advisors. Our champion achieved **99.33%** within $\pm10$ points on the sealed holdout test set (`artifacts/models/final/regression_champion_bundle.json` line 20).

---

## 3. Classification Track Metrics (Low / Moderate / High Risk)

### 1. Macro-Averaged F1-Score (Primary Ranker)
- **Formula:** 
  $$\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^C F1_c, \quad F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
- **Intuition:** Arithmetic mean of F1 scores across Low, Moderate, and High classes, treating all classes with equal importance regardless of size.
- **When it Misleads:** It treats rare minority classes equally, which is desirable in our case because high-risk students are a minority.
- **Why Chosen:** Prevents models from ignoring the smaller High-Risk cohort ($24.8\%$) while inflating accuracy on the Moderate class ($56.7\%$). Achieved **0.9494** on validation folds (`artifacts/leaderboards/classification_leaderboard.md`).

### 2. High-Risk Recall (Sensitivity)
- **Formula:** 
  $$\text{Recall}_{\text{High}} = \frac{TP_{\text{High}}}{TP_{\text{High}} + FN_{\text{High}}}$$
- **Intuition:** Out of 100 students truly at high risk of dropping out, how many did the system identify?
- **Why Chosen:** **Cost-sensitive priority.** In student retention, a False Positive triggers a harmless support check-in; a False Negative means an at-risk student drops out undetected. Our champion achieves **94.59%** on CV and **89.19%** on test (`artifacts/models/final/classification_champion_bundle.json` line 31).

### 3. Expected Calibration Error (ECE)
- **Formula:** 
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Intuition:** When the model outputs a $80\%$ probability of high risk, is the student actually at high risk $80\%$ of the time?
- **Why Chosen:** Ensures probabilities passed downstream to the Fusion Framework are statistically reliable decision inputs. Achieved **0.0172** ($1.7\%$) on validation and **0.0319** on test.

### 4. Quadratic Weighted Kappa ($\kappa_w$)
- **Formula:** 
  $$\kappa_w = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i - j)^2}{(C - 1)^2}$$
- **Intuition:** Measures inter-rater agreement for ordinal classes, penalizing severe misclassifications (confusing Low with High) four times more severely than adjacent errors (Low with Moderate).
- **Why Chosen:** Achieved **0.9238** on the test set (`artifacts/models/final/classification_champion_bundle.json` line 33), confirming near-perfect ordinal alignment.

---

## 4. Operational & Engineering Metrics

| Metric | Champion Value | Purpose in Attrition Production |
| :--- | :--- | :--- |
| **Overfitting Gap** | $0.0482$ (Clf) / $0.5178$ (Reg) | Difference between train and validation scores. Confirms model has not memorized training folds. |
| **Inference Latency** | $0.967$ ms (Ridge) | Milliseconds required per prediction. Ensures instant UI response during live student counseling. |
| **Model Size** | $0.92$ KB (Ridge) | Serialized disk footprint. Enables lightweight deployment on edge or university server hardware. |
