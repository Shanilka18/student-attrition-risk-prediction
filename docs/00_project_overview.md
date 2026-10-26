# Project Overview: Psychosocial Intelligence Framework

> **RESEARCH NOTICE & ETHICAL DISCLAIMER**  
> The Psychosocial Risk Index (PRI) is an academic decision-support and educational analytics indicator developed for research purposes at SLIIT Kandy, Sri Lanka. It is **not** a clinical psychological assessment, psychiatric evaluation, or medical diagnostic instrument.

---

## 1. Executive Summary & Problem Formulation

Undergraduate student attrition in computing and engineering disciplines represents a complex, multi-dimensional challenge. While academic performance (e.g., GPA, assignment submissions) and digital engagement metrics (LMS interaction logs) provide retrospective indicators of student struggle, **psychosocial dynamics** frequently serve as leading, upstream indicators of disengagement and potential dropout.

The **Psychosocial Intelligence Framework** constitutes a foundational sub-system of the larger multi-factor student attrition prediction platform at **SLIIT Kandy**. The specialized role of this component is to:
1. Process standardized 15-item psychometric questionnaire responses.
2. Quantify an individual student's psychosocial attrition vulnerability via a continuous **Psychosocial Risk Index (PRI)** scaled from **0 to 100**.
3. Categorize students into actionable risk tiers: **Low Risk**, **Moderate Risk**, and **High Risk**.
4. Generate subscale factor diagnostic scores across five psychological dimensions.
5. Provide local and global feature attribution via **SHAP (SHapley Additive exPlanations)** to uncover the specific root drivers of student vulnerability.
6. Deliver a standardized, leak-free export contract for downstream consumption by the shared **Fusion Framework**, which synthesizes Academic, Engagement, Environmental, and Psychosocial indices.

---

## 2. Psychometric Instrument & Latent Factor Foundations

The input data originates from a structured 15-item survey scored on a standard 5-point Likert scale:
- $1 =$ Strongly Disagree
- $2 =$ Disagree
- $3 =$ Neutral
- $4 =$ Agree
- $5 =$ Strongly Agree

### Latent Factor Decomposition
The instrument measures five distinct psychological constructs (3 items per construct):

| Factor Domain | Items | Construct Type | Risk Direction | Conceptual Description |
| :--- | :--- | :--- | :--- | :--- |
| **Stress** | `S1`, `S2`, `S3` | Risk Factor | Direct ($\uparrow$ Score $\implies \uparrow$ Risk) | Academic pressure, workload anxiety, and personal strain. |
| **Motivation** | `M1`, `M2`, `M3` | Protective Factor | Inverted ($\uparrow$ Score $\implies \downarrow$ Risk) | Intrinsic drive, goal persistence, and career clarity. |
| **Belongingness** | `B1`, `B2`, `B3` | Protective Factor | Inverted ($\uparrow$ Score $\implies \downarrow$ Risk) | Campus integration, institutional fit, and faculty rapport. |
| **Self-Efficacy** | `SE1`, `SE2`, `SE3` | Protective Factor | Inverted ($\uparrow$ Score $\implies \downarrow$ Risk) | Academic self-confidence and resilient coping capacity. |
| **Peer Support** | `PS1`, `PS2`, `PS3` | Protective Factor | Inverted ($\uparrow$ Score $\implies \downarrow$ Risk) | Study groups, emotional camaraderie, and collaborative networks. |

---

## 3. Simple Explanation (Explain Like I am 5 Years Old)

### The Everyday Analogy: The "Student Resilience Ship" Navigating Academic Seas

Imagine every university student is the captain of a small ship sailing across a wide ocean to reach Graduation Island. 

- **Stress (`S1-S3`)** represents the **Storm and Heavy Waves**. When the storm is violent and waves crash over the deck (high stress), the ship is pushed closer to danger.
- **Motivation (`M1-M3`)** is the ship's **Engine and Fuel**. When the engine runs strong, the ship has the energy to push forward despite the storm.
- **Belongingness (`B1-B3`)** is the **Anchor and Compass**. Feeling at home at university keeps the ship steady so it doesn't drift away into loneliness.
- **Self-Efficacy (`SE1-SE3`)** is the **Captain's Self-Belief and Repair Kit**. When things break down or an exam goes poorly, a resilient captain knows how to fix the rudder and keep going.
- **Peer Support (`PS1-PS3`)** is the **Flotilla of Friendly Ships Sailing Alongside**. When you need help, your coursemates toss you a tow rope.

### What the Psychosocial Risk Index (PRI) Does
The PRI is like a **Smart Navigational Alarm** on the bridge of the ship:
- If the storms are raging (high stress) and the captain has run out of fuel (low motivation) with no friends nearby (low peer support), the alarm flashes **"High Risk" (PRI 67 - 100)**.
- If the waters are calm and the engines and support network are strong, the screen shows a green light: **"Low Risk" (PRI 0 - 33)**.
- Crucially, the system doesn't just ring a generic bell; it points directly to the dashboard and explains: *"Captain, your risk is high right now mainly because your Peer Support is low and Workload Stress is overwhelming."* This allows student advisors to offer immediate, personalized help before the ship gets lost at sea.

---

## 4. Technical Explanation (For Viva Panel & ML Specialists)

### 4.1 Mathematical Formulation of Latent Factors and Ground Truth PRI
Let $x_{i,j} \in \{1, 2, 3, 4, 5\}$ represent the Likert response of student $i$ to item $j$. For each factor $k \in \{\text{Stress}, \text{Motivation}, \text{Belongingness}, \text{Self-Efficacy}, \text{Peer Support}\}$, the raw domain score $F_k$ is the arithmetic mean of its three constituent items:

$$F_k = \frac{1}{3} \sum_{j \in \mathcal{I}_k} x_{i,j}, \quad \text{where } |\mathcal{I}_k| = 3, \; F_k \in [1.0, 5.0]$$

Because higher scores on protective factors indicate psychological resilience, protective subscales are mapped to a risk-oriented dimension using an affine linear transformation:

$$R_{\text{Stress}} = F_{\text{Stress}}$$

$$R_k = 6.0 - F_k, \quad \forall k \in \{\text{Motivation}, \text{Belongingness}, \text{Self-Efficacy}, \text{Peer Support}\}$$

The unified composite risk metric $\bar{R} \in [1.0, 5.0]$ is computed as:

$$\bar{R} = \frac{1}{5} \left( R_{\text{Stress}} + R_{\text{Motivation}} + R_{\text{Belongingness}} + R_{\text{Self-Efficacy}} + R_{\text{Peer Support}} \right)$$

To provide an interpretable, standardized index for student affairs professionals and downstream ML modules, $\bar{R}$ is scaled to the interval $[0.0, 100.0]$:

$$\text{PRI} = \left( \frac{\bar{R} - 1.0}{4.0} \right) \times 100.0$$

The continuous index is subsequently discretized into tri-level risk bands:
$$\text{Risk Category} = \begin{cases} 
\text{Low}, & 0.00 \le \text{PRI} \le 33.33 \\
\text{Moderate}, & 33.34 \le \text{PRI} \le 66.66 \\
\text{High}, & 66.67 \le \text{PRI} \le 100.00 
\end{cases}$$

### 4.2 Machine Learning Architecture & Benchmark Pipeline
To capture non-linear feature interactions, threshold effects, and latent interdependencies, the modeling framework implements:
1. **Baselines:** Stratified Dummy Classifier and ElasticNet / Regularized Logistic Regression.
2. **Gradient-Boosted Decision Trees:** XGBoost, LightGBM, and CatBoost with Bayesian hyperparameter optimization via Optuna.
3. **Tabular Deep Learning:** A specialized `TabularMLP` featuring batch normalization, ReLU non-linearities, and dropout regularization ($p=0.2$) to handle tabular psychometric profiles without overfitting.

### 4.3 Data Leakage Prevention Protocol
The framework adheres to an uncompromising anti-leakage policy:
- **Partitioning:** Stratified splitting partitions the corpus into training (64%), validation (16%), and holdout test (20%) sets.
- **Isolation:** All feature scalers, normalizers, and psychometric factor weighting matrices are fitted solely on training folds within nested cross-validation loops.
- **Sealed Test Set:** The test set is evaluated precisely once at the conclusion of model shortlisting to report unbiased generalization error.

### 4.4 Explainability via Game-Theoretic Attributions (SHAP)
To guarantee transparency and actionable intervention, predictions are explained using **TreeSHAP** (for tree ensembles) and **KernelSHAP** (for neural architectures). For a prediction $f(x)$, local explanations satisfy efficiency, symmetry, dummy, and additivity axioms:

$$f(x) = \phi_0 + \sum_{m=1}^{M} \phi_m(x)$$

where $\phi_0$ is the base expected value and $\phi_m(x)$ is the Shapley attribution value for feature $m$. Positive $\phi_m$ values quantify risk elevation, while negative $\phi_m$ values quantify risk mitigation.

### 4.5 Fusion Framework Export Contract
The PRI service outputs a clean, strongly typed schema (via Pydantic) to interface seamlessly with the shared multi-modal attrition system:
```json
{
  "study_id": "STUDY_0042",
  "psychosocial_risk_index": 72.45,
  "risk_category": "High",
  "subscale_scores": {
    "stress": 4.33,
    "motivation": 2.00,
    "belongingness": 2.33,
    "self_efficacy": 2.67,
    "peer_support": 1.67
  },
  "model_version": "1.0.0",
  "timestamp_utc": "2026-10-17T10:15:30Z",
  "disclaimer": "RESEARCH NOTICE: The Psychosocial Risk Index (PRI) is an academic decision-support indicator..."
}
```

---

## 5. Architectural Improvements Proposed for the Framework

1. **Explicit Schema Validation Pipeline (`pydantic` + `pandera`):** Enforce runtime DataFrame schema validation at ingestion to guarantee strict Likert range integrity $[1, 5]$ and prevent silent NaN propagations before feature transformations.
2. **Decoupled Scoring Pipelines (Pure Psychometric Formula vs. Supervised ML):** Maintain a deterministic psychometric baseline score alongside supervised non-linear tree/deep learning models. This guarantees transparency and creates an unassailable benchmark during viva defense.
3. **Dedicated Artifact Registry & Experiment Tracking (`/artifacts/runs`):** Structure run manifests with commit hashes, hyperparameter logs, and timestamped metrics to ensure every leaderboard entry maps directly to a deterministic model checkpoint.
4. **Synthetic Data Generation via Copula / Latent Correlational Synthesis:** Instead of naive independent random draws, generate synthetic data using multivariate Gaussian copulas or latent factor correlation structures that mirror real-world psychometric covariance matrices (e.g., negative correlation between stress and self-efficacy).
5. **Contract-First OpenAPI / Mock Generator for Fusion Team:** Provide a standalone mock server and export script (`scripts/export_fusion_mock.py`) so teammates developing the Academic, Engagement, and Environmental models can integrate immediately without waiting for final trained model checkpoints.
