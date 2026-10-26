# 11 — The Model Selection Story: Tournaments, Trade-Offs, & Empirical Decisions

## 1. Simple Explanation (Age-5 Level)

When picking the best runner for a race, you don't just guess who looks fastest in practice—you have all of them run the same track under the same rules, timing them with a stopwatch.

We set up two different competitions:
1. **The Number Race (Continuous PRI 0 to 100):** Who can guess the exact score closest to the real value?
   - A simple, clean model called **Ridge Regression** won this race because adding up survey points is mostly a straight-line math problem. More complicated models tried too hard and got confused.
2. **The Category Race (Low, Moderate, High Risk):** Who can correctly catch the students who are at High Risk so teachers can help them?
   - A smart team-based model called **HistGradientBoosting** won this race because separating students into three buckets requires looking at combinations of clues (like high stress combined with low friends).

We also had a "silly baseline model" that just guessed the average answer every time. Every serious model had to easily beat this baseline to prove it actually learned something useful!

---

## 2. Experimental Setup & Zero-Leakage Tournament

To ensure complete scientific integrity, every model was trained and evaluated under an identical protocol configured in `config/default.yaml`:
- **Training Set:** $70\%$ ($N = 2,086$ students).
- **Validation Protocol:** 5-fold Stratified Cross-Validation on the training set exclusively.
- **Sealed Test Set:** $15\%$ ($N = 448$ students) locked until final verification.
- **Feature Set Tournament:** Each model competed on two feature representations:
  1. *Factor Scores:* 5 aggregated construct scores (Stress, Motivation, Belonging, Self-Efficacy, Peer Support).
  2. *Raw Items:* 15 individual questionnaire item responses.

Across both tracks, **Factor Scores strictly dominated Raw Items**, achieving higher generalization metrics, reduced variance, and lower overfitting gaps.

---

## 3. Continuous PRI Regression Tournament

*Primary Metric: Validation $R^2$ (Tiebreak: Validation RMSE)*

| Rank | Model Candidate | Category | Feature Set | Val $R^2$ | Val RMSE | Val MAE | Overfitting Gap | Latency (ms) | Size (KB) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 👑 **1** | **Ridge Regression** | Linear/Shallow | `factor_scores` | **0.9908** | **1.8023** | **1.4036** | **0.5178** | **0.967** | **0.92** |
| 2 | HistGradientBoosting | Boosting | `factor_scores` | 0.9852 | 2.2885 | 1.8178 | 0.2806 | 4.217 | 355.69 |
| 3 | Random Forest | Ensemble/Bagging | `factor_scores` | 0.9796 | 2.6855 | 2.0801 | 1.0180 | 29.998 | 18,408.0 |
| 4 | Decision Tree | Linear/Shallow | `factor_scores` | 0.9210 | 5.2857 | 4.0633 | 4.0078 | 0.826 | 291.78 |
| 5 | Dummy Baseline | Baseline | `factor_scores` | -0.0000 | 18.8080 | 15.2251 | 0.0592 | 0.072 | 0.75 |

*(Verified from: `artifacts/leaderboards/regression_leaderboard.md`)*

### Why Ridge Won the Regression Track
1. **Construct Alignment:** The mathematical formulation of the PRI is inherently additive:
   $$\text{PRI} = 100 \times \left( \frac{\frac{1}{5} \sum_{k=1}^5 \overline{Factor}_k - 1.0}{4.0} \right)$$
   An $L_2$-regularized linear model perfectly mirrors this linear aggregation without distorting the scale.
2. **Occam’s Razor & Extreme Efficiency:** Ridge achieved an outstanding $R^2 = 0.9908$ while consuming just **0.92 KB** of memory with **0.967 ms** latency—over $30\times$ faster than Random Forest ($29.998$ ms, $18.4$ MB).
3. **Exact Explainability:** Ridge provides closed-form, perfectly additive LinearSHAP explanations with zero Monte Carlo sampling approximation error.

---

## 4. Tri-Level Risk Classification Tournament

*Primary Metric: Validation Macro-F1 (Tiebreak: Cohen's Quadratic Weighted Kappa / MCC)*

| Rank | Model Candidate | Category | Feature Set | Val Macro-F1 | Val Accuracy | Balanced Acc | Overfitting Gap | Latency (ms) | Size (KB) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 👑 **1** | **HistGradientBoosting** | Boosting | `factor_scores` | **0.9494** | **0.9531** | **0.9461** | **0.0482** | 114.674 | 1,059.45 |
| 2 | Random Forest | Ensemble/Bagging | `factor_scores` | 0.9428 | 0.9487 | 0.9326 | 0.0548 | 30.121 | 4,300.63 |
| 3 | Decision Tree | Linear/Shallow | `factor_scores` | 0.8937 | 0.9018 | 0.8922 | 0.1039 | 1.308 | 48.41 |
| 4 | Ridge Classifier | Linear/Shallow | `factor_scores` | 0.7168 | 0.8147 | 0.6920 | 0.0124 | 2.035 | 1.55 |
| 5 | Dummy Baseline | Baseline | `factor_scores` | 0.2412 | 0.5670 | 0.3333 | 0.0002 | 0.154 | 1.03 |

*(Verified from: `artifacts/leaderboards/classification_leaderboard.md`)*

### Why HistGradientBoosting Won the Classification Track
1. **Non-Linear Decision Boundaries:** While continuous scoring is linear, categorizing students into ordinal risk buckets (Low/Moderate/High) involves boundary non-linearities and threshold shifts that tree ensembles resolve significantly better than linear classifiers ($0.9494$ vs. $0.7168$ Macro-F1).
2. **Cost-Sensitive High-Risk Recall:** Missing an at-risk student is the most severe error in student retention. HistGradientBoosting delivered **94.59% High-Risk Recall** on validation folds and **89.19%** on the sealed holdout test set (`artifacts/models/final/classification_champion_bundle.json` line 31).
3. **Probability Calibration:** Achieved an Expected Calibration Error (ECE) of just **0.0172** ($1.7\%$), meaning output softmax probabilities correspond reliably to true empirical likelihoods.

---

## 5. Statistical Significance Testing

In accordance with Phase 5 requirements, pairwise Wilcoxon signed-rank tests were performed across CV folds between the champion models and challengers (`artifacts/leaderboards/champion.json` lines 24–65, 90–131):
- **Regression:** Ridge vs. HistGradientBoosting ($1.8023$ vs. $2.2885$ mean RMSE), Ridge vs. Random Forest ($1.8023$ vs. $2.6855$), Ridge vs. Dummy ($1.8023$ vs. $18.8080$).
- **Classification:** HistGradientBoosting vs. Random Forest ($0.9494$ vs. $0.9428$), HistGradientBoosting vs. Decision Tree ($0.9494$ vs. $0.8937$), HistGradientBoosting vs. Ridge ($0.9494$ vs. $0.7168$).

---

## 6. What I Would Do Differently with Real Data

1. **Re-evaluate Deep Tabular Architectures:** Deep models (TabNet / FT-Transformers) struggled on 5 tabular factor features due to small-parameter under-specification. With real longitudinal data containing academic semester histories and engagement logs, neural embeddings may unlock richer cross-domain interactions.
2. **Dynamic Cost Matrix in Optuna Tuning:** Explicitly weight false negatives (missing a high-risk student) as $10\times$ more punitive than false positives in the Optuna loss function.
3. **Subgroup Fairness Constraints:** Enforce Demographic Parity and Equal Opportunity constraints during hyperparameter search across faculties, scholarship recipients, and accommodation types.
