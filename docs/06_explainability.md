# Phase 6: Explainability and Attribution Engine (SHAP)

> **Document Classification:** Scientific, Algorithmic & Architectural Documentation  
> **Target Audience:** External Examiners, Viva Panelists, Academic Evaluators, University Welfare Counselors  
> **System Component:** Psychosocial Risk Index (PRI) Attribution Pipeline  
> **Status:** Production-Ready / Fully Tested  

---

## 1. Simple Explanation (ELI5: The Fair Team Game Analogy)

Imagine a university football team wins a crucial championship match with a final score of **5 goals to 1** (a goal difference of $+4$). After the match, the team receives a \$1,000 performance award. 

Now, how should the team manager fairly distribute that award among the 11 players?
- Should the striker get everything because they kicked the final ball into the net? That would be completely unfair to the midfielder who intercepted the ball and made a pinpoint 40-yard cross.
- Should the defender get nothing because they didn't score? That would be unfair because without their sliding tackles, the opposing team would have scored 6 goals.
- What if two players only perform brilliantly when playing together, but poorly when separated?

In 1953, a Nobel Prize-winning mathematician named **Lloyd Shapley** solved this exact puzzle using cooperative game theory:
> To calculate a player's fair share, simulate the game with **every possible combination** of players. Add that player to each group and measure how much the score changes because of their arrival (their *marginal contribution*). Then, take the weighted average of all those marginal contributions.

### Applying the Analogy to Student Psychosocial Risk
In our **Psychosocial Intelligence Framework**:
- The **"Match Score"** is the student's predicted **Psychosocial Risk Index (PRI)** (e.g., $72.4/100$, High Risk) or the log-odds of requiring targeted academic support.
- The **"Baseline Score"** (expected value) is the average risk score across the entire student population (e.g., $50.0/100$).
- The **"Players"** are the 5 psychosocial latent factors: **Stress**, **Motivation**, **Belongingness**, **Self-Efficacy**, and **Peer Support** (measured across 15 survey items).

When our model flags a student at **$72.4/100$**, the SHAP engine answers:
> *"Out of the $+22.4$ points above average, exactly how many points came from situational stress ($+14.2$), how many from low self-efficacy ($+9.1$), and how many points were successfully held back by high peer support ($-3.8$ points of protective buffering)?"*

Because factors interact in complex non-linear ways, SHAP guarantees that every point of risk is allocated **fairly, mathematically provably, and with zero leftover error**.

---

## 2. Technical Explanation (Viva & Academic Defense)

### 2.1 Theoretical Foundation: Cooperative Game Theory and Shapley Values
Let $N = \{1, 2, \dots, M\}$ denote the grand coalition of all $M$ input features. A predictive model $f(x)$ maps an instance $x \in \mathbb{R}^M$ to a scalar prediction. For any subset $S \subseteq N$, let $v(S)$ be the characteristic value function representing the model's expected output when only the subset of features in $S$ is conditioned upon:
$$v(S) = \mathbb{E}_{X_{\bar{S}} | X_S = x_S}[f(X)]$$

The classical Shapley attribution value $\phi_i(v)$ for feature $i \in N$ is defined as the weighted average marginal contribution across all possible feature subsets:
$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \, (|N| - |S| - 1)!}{|N|!} \Big[ v(S \cup \{i\}) - v(S) \Big]$$

### 2.2 The Four Non-Negotiable Axioms
Shapley values represent the **unique** attribution method that satisfies the following four axiomatic properties simultaneously:

1. **Efficiency (Additivity / Local Accuracy):**
   The sum of all feature attributions plus the expected base value $\phi_0 = v(\emptyset) = \mathbb{E}[f(X)]$ strictly equals the model output $f(x)$:
   $$\sum_{i=1}^M \phi_i(v) = f(x) - \mathbb{E}[f(X)] \implies f(x) = \phi_0 + \sum_{i=1}^M \phi_i$$
2. **Symmetry (Equal Treatment of Equals):**
   If two features $i$ and $j$ contribute identically across all possible subsets, their attributions must be identical:
   $$\forall S \subseteq N \setminus \{i, j\}, \; v(S \cup \{i\}) = v(S \cup \{j\}) \implies \phi_i(v) = \phi_j(v)$$
3. **Dummy Player (Null Effect):**
   If feature $i$ provides zero marginal contribution to all subsets, its attribution is strictly zero:
   $$\forall S \subseteq N \setminus \{i\}, \; v(S \cup \{i\}) = v(S) \implies \phi_i(v) = 0$$
4. **Additivity (Linear Linearity across Models):**
   For any combined game or ensemble $u + v$:
   $$\phi_i(u + v) = \phi_i(u) + \phi_i(v)$$

---

### 2.3 Algorithmic Implementations

#### 1. LinearSHAP (Exact Closed-Form for Champion Regression: `Ridge`)
For linear models $f(x) = \beta_0 + \sum_{i=1}^M \beta_i x_i$ under feature independence or empirical background expectations, the Shapley value reduces to an exact closed-form linear identity:
$$\phi_i(x) = \beta_i \left(x_i - \mathbb{E}[X_i]\right), \quad \phi_0 = \beta_0 + \sum_{i=1}^M \beta_i \mathbb{E}[X_i] = \mathbb{E}[f(X)]$$
Because this evaluation requires zero sampling or permutation loops, computation latency is $O(M)$ per instance.

#### 2. TreeSHAP (Polynomial-Time Algorithm for Champion Classification: `HistGradientBoosting`)
Naive evaluation of Shapley values requires $O(M \cdot 2^M)$ evaluations, which is computationally intractable for complex models.
We utilize the **TreeSHAP** algorithm (Lundberg et al., *Nature Machine Intelligence*, 2020), which computes exact Shapley attributions in polynomial time:
$$O\left(T \cdot L \cdot D^2\right)$$
where $T$ is the number of trees, $L$ is the maximum number of leaves, and $D$ is the maximum tree depth. 
TreeSHAP tracks the proportion of training instances flowing down each subtree to compute conditional expectations $\mathbb{E}[f(X) | X_S]$ in a single recursive tree traversal.

For multi-class gradient boosting (`HistGradientBoostingClassifier`), TreeSHAP computes attributions in raw margin logit space for each class $c \in \{\text{High}, \text{Low}, \text{Moderate}\}$:
$$f_c(x) = \phi_{0, c} + \sum_{i=1}^M \phi_{i, c}(x)$$
To ensure strict clinical and academic alignment, our explainability engine targets class $c = \text{High Risk}$ (class index $0$ in scikit-learn's lexicographical order).

---

## 3. Global Explainability & Factor Aggregation

### 3.1 Global Feature Importance Tables

Global importance is quantified as the mean absolute SHAP value across the entire validation cohort ($N = 448$ students):
$$I_i = \frac{1}{N} \sum_{k=1}^N |\phi_i(x^{(k)})|$$

#### Regression Track Champion (`Ridge` predicting continuous PRI 0–100)
| Factor Domain | Mean $\|SHAP\|$ ($\pm \text{Points}$) | Relative Contribution (\%) | Ground-Truth Generating Weight |
| :--- | :---: | :---: | :---: |
| **Stress** | **6.0651** | **27.04%** | 0.28 (28.0%) |
| **Motivation** | **4.9867** | **22.23%** | 0.22 (22.0%) |
| **Self-Efficacy** | **4.3521** | **19.40%** | 0.18 (18.0%) |
| **Belongingness** | **3.9081** | **17.42%** | 0.18 (18.0%) |
| **Peer Support** | **3.1174** | **13.90%** | 0.14 (14.0%) |

#### Classification Track Champion (`HistGradientBoosting` predicting High Risk)
| Factor Domain | Mean $\|SHAP\|$ (Margin Logits) | Relative Contribution (\%) | Empirical Rank |
| :--- | :---: | :---: | :---: |
| **Motivation** | **1.8781** | **23.93%** | 1 |
| **Stress** | **1.8700** | **23.83%** | 2 |
| **Self-Efficacy** | **1.5606** | **19.88%** | 3 |
| **Belongingness** | **1.5010** | **19.12%** | 4 |
| **Peer Support** | **1.0388** | **13.24%** | 5 |

### 3.2 Visual Explanation Artifacts
All visualizations are generated and persisted in `/artifacts/shap/`:
1. **Global Summary Bar Plot (`global_bar_*.png`):** Horizontal ranking of factors by mean $|SHAP|$, communicating dominant risk drivers to administrative leadership.
2. **Beeswarm Summary Plot (`beeswarm_*.png`):** Depicts each individual student as a point colored by feature value (red = high score, blue = low score). It clearly illustrates the directional mechanics: high stress values push predictions far to the right (increased risk), while high motivation and self-efficacy push predictions far to the left (protective buffering).
3. **Partial Dependence Plot (`dependence_*.png`):** Displays non-linear transition thresholds and feature interaction effects along the continuous spectrum of the top factor.

---

## 4. Local Per-Student Explanations & Plain-Language Narratives

### 4.1 Per-Student Waterfall Attribution Contract
For every student profile, the pipeline generates a standardized waterfall JSON object detailing the step-by-step composition from baseline to final risk index.

```json
{
  "study_id": "SLIIT_01986",
  "base_value": 50.0482,
  "predicted_value": 43.1974,
  "net_attribution_sum": -6.8508,
  "top_risk_increasing": [
    {
      "feature": "factor_stress",
      "factor": "stress",
      "shap_value": 4.1482,
      "feature_value": 3.6667,
      "direction": "risk_increasing"
    }
  ],
  "top_risk_decreasing": [
    {
      "feature": "factor_motivation",
      "factor": "motivation",
      "shap_value": -5.3812,
      "feature_value": 4.0,
      "direction": "risk_decreasing"
    },
    {
      "feature": "factor_belongingness",
      "factor": "belongingness",
      "shap_value": -3.1204,
      "feature_value": 3.5,
      "direction": "risk_decreasing"
    },
    {
      "feature": "factor_self_efficacy",
      "factor": "self_efficacy",
      "shap_value": -2.4974,
      "feature_value": 3.3333,
      "direction": "risk_decreasing"
    }
  ]
}
```

### 4.2 Deterministic Plain-Language Narrative Engine
To prevent catastrophic LLM hallucinations in sensitive educational interventions, explanations are generated by a deterministic, rule-based narrative engine adhering to two strict scientific guidelines:

1. **Wording Rule:** Psychosocial factors are explicitly designated as **"contributing indicators, not causes"**. Statistical attributions describe correlative and predictive importance; they do not establish causal medical etiology.
2. **Mandatory Non-Clinical Disclaimer:** Appended to every report, dashboard card, and API response.

#### Sample Generated Narrative (Student `SLIIT_01986`):
> **Headline:** Student SLIIT_01986: Psychosocial Risk Index is MODERATE (43.2/100).  
> **Primary Risk Drivers:** The primary contributing indicator elevating psychosocial risk is moderate levels of situational stress (score: 3.7).  
> **Protective Buffers:** This profile is actively buffered by strong intrinsic academic motivation (score: 4.0), moderate sense of campus belongingness (score: 3.5), and moderate academic self-efficacy and belief in capability (score: 3.3).  
> **Actionable Guidance:** Offer time-management workshops, workload review, or academic counseling check-in.  
> **Research Notice:** RESEARCH NOTICE: The Psychosocial Risk Index (PRI) is an academic decision-support indicator developed for educational research at SLIIT Kandy. It is not a clinical psychological evaluation or psychiatric diagnosis.

---

## 5. Scientific Sanity Checks & Ground-Truth Recovery

To ensure absolute publication-grade integrity, all SHAP attributions are systematically audited against four automated sanity checks:

```text
====================================================================================
                        PHASE 6 SANITY CHECK AUDIT REPORT
====================================================================================
 Check Description                  Tolerance / Metric       Regression   Classification
------------------------------------------------------------------------------------
 1. Additivity Axiom Verification   Max Error < 0.01          PASS (0.0)   PASS (0.0)
 2. Subsample Rank Stability        Spearman rho >= 0.85      PASS (1.0)   PASS (1.0)
 3. Permutation Importance Concord. Spearman rho >= 0.70      PASS (1.0)   PASS (0.8)
 4. Ground-Truth Weight Recovery    Spearman rho >= 0.80      PASS (0.97)  PASS (0.87)
====================================================================================
```

### 5.1 Verification Outcomes
1. **Additivity Axiom:** Both Ridge and TreeSHAP reconstruct the model prediction with exact numerical precision (Maximum Absolute Error $= 0.000000$), confirming zero numerical leakage or calculation distortion.
2. **Subsample Stability:** Computing SHAP on a random 50% split of validation data yields a Spearman rank correlation of **$\rho = 1.0$**, confirming that feature rankings are invariant to sampling fluctuations.
3. **Permutation Feature Importance Concordance:** Model-agnostic permutation importance on validation data correlates strongly with global SHAP importance ($\rho = 1.0$ on regression, $\rho = 0.80$ on classification), confirming that SHAP faithfully mirrors the model's true empirical dependency structure.
4. **Empirical Ground-Truth Weight Recovery:**
   Because synthetic data generation (Phase 2) synthesized responses using known ground-truth factor weights:
   - $\text{Stress} = 0.28$
   - $\text{Motivation} = 0.22$
   - $\text{Self-Efficacy} = 0.18$
   - $\text{Belongingness} = 0.18$
   - $\text{Peer Support} = 0.14$

   Our empirical SHAP analysis demonstrates **outstanding recovery fidelity**:
   - **Regression Model (`Ridge`):** Spearman rank correlation $\rho = \mathbf{0.9747}$ ($p < 0.005$), Pearson correlation $r = \mathbf{0.9984}$. The empirical feature importance perfectly recapitulates the underlying generative equations.
   - **Classification Model (`HistGradientBoosting`):** Spearman rank correlation $\rho = \mathbf{0.8721}$, accurately capturing Motivation and Stress as the leading drivers, with Peer Support consistently identified as the least dominant factor.

---

## 6. Ethical Governance and Research Governance

> ### ⚠️ RESEARCH INDICATOR DISCLAIMER
> The Psychosocial Risk Index (PRI), its factor aggregations, and its SHAP attribution values are **purely statistical research and educational decision-support indicators**. They are engineered specifically for early academic warning systems at SLIIT Kandy, Sri Lanka.
> 
> **They DO NOT constitute, and must NEVER be interpreted as:**
> 1. A clinical psychological evaluation.
> 2. A formal psychiatric diagnosis of depression, anxiety, or affective disorders.
> 3. An autonomous determinant of disciplinary or administrative student sanction.
> 
> Any flagged high-risk score must serve exclusively as an invitation for supportive, voluntary human intervention by qualified university student welfare and counseling personnel. All student identifiers are strictly pseudonymized using cryptographic Study IDs (`study_id`).
