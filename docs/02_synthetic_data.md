# Phase 2 Documentation: Realistic Synthetic Survey Data Generation

> **RESEARCH NOTICE & ETHICAL DISCLAIMER**  
> All datasets generated in Phase 2 (`/data/synthetic/`) are strictly synthetic, mathematically simulated records created solely for software verification, machine learning pipeline prototyping, and psychometric architecture development. They do **not** represent actual human clinical diagnoses or real student educational records at SLIIT Kandy, Sri Lanka.

---

## 1. Executive Summary & Objective

In educational data mining and psychometric modeling, real-world student survey data is subject to institutional review board (IRB) ethics clearance, participant privacy constraints, and limited cohort availability. Developing machine learning architectures on naive, uniformly distributed mock data causes catastrophic failure when transitioning to real data because real psychometric responses exhibit **multivariate latent correlations**, **ordinal response thresholds**, **subgroup shifts**, and **non-random missingness**.

Phase 2 introduces a **scientifically grounded synthetic data generator** (`src/data/generate_synthetic.py` configured via `config/synthetic.yaml`) modeling a Sri Lankan undergraduate computing student population. It generates:
1. A default research-scale dataset ($N = 3{,}000$).
2. A small sample dataset ($N = 300$) mimicking real preliminary pilot study collections.
3. A corrupted ("dirty") dataset ($N = 3{,}000$) verifying the Phase 1 schema validation engine.
4. Comprehensive metadata (`/data/synthetic/metadata.json`) documenting seeds, parameters, achieved internal reliability coefficients, and correlation matrices.
5. High-resolution diagnostic profiling charts (`/artifacts/plots/`).

---

## 2. Simple Explanation (Explain Like I am 5 Years Old)

### The Everyday Analogy: The "High-Tech Flight Simulator"

Imagine you are training to be an airline pilot. Before you are allowed to fly a commercial Boeing 777 carrying hundreds of passengers through real thunderstorms, the aviation academy places you inside a **Multi-Million Dollar Flight Simulator**:

1. **Why Not Use a Toy Video Game? (Avoiding Trivial Mock Data):**  
   If the flight simulator just had random, arcade buttons that didn't behave like real physics, you wouldn't learn how a real aircraft responds to turbulence. In the same way, if our mock student survey just picked numbers from 1 to 5 completely at random, our machine learning models would learn nothing about real human psychology.

2. **Realistic Weather Physics (Correlated Latent Mindsets):**  
   In the simulator, when a storm hits (High Stress), the winds naturally drag down the plane's airspeed (Low Motivation) and shake the navigation instruments (Lower Self-Efficacy). Our generator mimics this: students with high stress naturally tend to report lower motivation and lower belongingness, exactly like real human students.

3. **Subgroups in the Academy (Freshmen vs. Seniors):**  
   First-year pilots face a steep learning curve and initial anxiety, whereas senior fourth-year pilots have higher confidence and career focus. The simulator injects these small, realistic demographic adjustments.

4. **Realistic Glitches (Missing Answers & Sleepy Students):**  
   Sometimes a student skips a question because they were rushed (Missing Values), or a student is so tired they circle number 3 for all fifteen questions without reading (Straight-Lining). Our simulator injects these real-world imperfections so our validation gatekeeper knows how to handle them.

5. **A Safe Proving Ground:**  
   The simulator allows our data science team to test every algorithm, train models, and tune alerts without putting any real student at risk. Once the system works flawlessly in the simulator, it is ready to be loaded with real university survey responses!

---

## 3. Technical Explanation (For Viva Panel & ML Specialists)

### 3.1 Multivariate Latent Trait Modeling

Let $\boldsymbol{\theta}_i = [\theta_{i,\text{stress}}, \theta_{i,\text{mot}}, \theta_{i,\text{bel}}, \theta_{i,\text{se}}, \theta_{i,\text{peer}}]^T \in \mathbb{R}^5$ represent the vector of continuous, unobserved latent psychological constructs for respondent $i$. 

Latent traits are drawn from a Multivariate Normal distribution:

$$\boldsymbol{\theta}_i \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad \boldsymbol{\mu} = \mathbf{0}$$

The covariance/correlation matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{5 \times 5}$ is specified in `config/synthetic.yaml` according to established educational psychology literature:

$$\boldsymbol{\Sigma} = \begin{pmatrix} 
1.00 & -0.38 & -0.35 & -0.42 & -0.28 \\
-0.38 & 1.00 & 0.45 & 0.55 & 0.35 \\
-0.35 & 0.45 & 1.00 & 0.40 & 0.58 \\
-0.42 & 0.55 & 0.40 & 1.00 & 0.38 \\
-0.28 & 0.35 & 0.58 & 0.38 & 1.00 
\end{pmatrix}$$

### 3.2 Demographic Subgroup Shifts

Subgroup conditional shifts $\boldsymbol{\delta}_{\text{demo}}$ are introduced to reflect empirical cohort differences:
$$\boldsymbol{\theta}_i^* = \boldsymbol{\theta}_i + \boldsymbol{\delta}_{\text{year}}(y_i) + \boldsymbol{\delta}_{\text{gender}}(g_i)$$
- **Year 1 Transition:** $\delta_{\text{stress}} = +0.30$, $\delta_{\text{bel}} = -0.25$, $\delta_{\text{se}} = -0.20$ (transition shock).
- **Year 4 Seniors:** $\delta_{\text{mot}} = +0.30$, $\delta_{\text{se}} = +0.25$, $\delta_{\text{bel}} = +0.20$ (degree completion proximity).

### 3.3 Ordinal Likert Discretization & Cronbach's Alpha Calibration

To convert continuous latent trait $\theta_{i,k}^*$ into three observable 5-point Likert items $x_{i,j} \in \{1, 2, 3, 4, 5\}$, an ordinal probit threshold model with independent Gaussian item-level noise is applied:

$$x_{i,j}^* = \theta_{i,k}^* + \epsilon_{i,j}, \quad \epsilon_{i,j} \sim \mathcal{N}(0, \sigma_{\text{item}}^2)$$

$$x_{i,j} = \begin{cases}
1, & x_{i,j}^* \le \tau_1 \\
2, & \tau_1 < x_{i,j}^* \le \tau_2 \\
3, & \tau_2 < x_{i,j}^* \le \tau_3 \\
4, & \tau_3 < x_{i,j}^* \le \tau_4 \\
5, & x_{i,j}^* > \tau_4
\end{cases}$$

Where cutoffs $\boldsymbol{\tau} = [-1.35, -0.42, 0.42, 1.35]$.

#### Internal Consistency Calibration (Cronbach's $\alpha$)
Internal reliability for a $K$-item scale ($K=3$) is defined as:

$$\alpha = \frac{K}{K - 1} \left( 1 - \frac{\sum_{j=1}^K \sigma_j^2}{\sigma_X^2} \right), \quad X = \sum_{j=1}^K x_j$$

By setting $\sigma_{\text{item}} = 0.72$, the inter-item correlation lands at $\bar{r} \approx 0.61$, guaranteeing that achieved Cronbach's alphas across all five factors fall strictly within the publication-grade range of $[0.80, 0.85]$:

| Factor Construct | Constituent Items | Target $\alpha$ Range | Achieved $\alpha$ ($N=3000$) | Achieved $\alpha$ ($N=300$) |
| :--- | :--- | :--- | :--- | :--- |
| **Stress** | `S1`, `S2`, `S3` | $[0.70, 0.88]$ | **0.826** | **0.829** |
| **Motivation** | `M1`, `M2`, `M3` | $[0.70, 0.88]$ | **0.832** | **0.840** |
| **Belongingness** | `B1`, `B2`, `B3` | $[0.70, 0.88]$ | **0.827** | **0.813** |
| **Self-Efficacy** | `SE1`, `SE2`, `SE3` | $[0.70, 0.88]$ | **0.840** | **0.806** |
| **Peer Support** | `PS1`, `PS2`, `PS3` | $[0.70, 0.88]$ | **0.826** | **0.826** |

### 3.4 Non-Deterministic Attrition Outcome Formulation

To prevent trivial or perfectly collinear target labels, the binary outcome `attrition_label` is modeled as a Bernoulli trial governed by a latent logistic propensity incorporating **unobserved confounders**:

$$\text{logit}(P(Y_i = 1 \mid \boldsymbol{\theta}_i)) = \beta_0 + \sum_{k=1}^5 \beta_k \theta_{i,k}^* + \eta_i, \quad \eta_i \sim \mathcal{N}(0, \sigma_{\text{unobserved}}^2)$$

Where:
- Intercept $\beta_0 = -1.95$, yielding an empirical base rate of **26.5%** (realistic class imbalance).
- Psychosocial weights: $\beta_{\text{stress}} = +0.75$, $\beta_{\text{mot}} = -0.70$, $\beta_{\text{bel}} = -0.50$, $\beta_{\text{se}} = -0.45$, $\beta_{\text{peer}} = -0.35$.
- $\eta_i$ ($\sigma = 0.65$) captures external unmeasured drivers (e.g., sudden financial crises, medical emergencies, familial relocation).

### 3.5 Ground-Truth PRI Metric Formulation

The continuous target `pri_true` is computed as an affine mapping of weighted reversed protective factors and direct stress:

$$\bar{R}_i = w_S F_{i,S} + w_M (6 - F_{i,M}) + w_B (6 - F_{i,B}) + w_{SE} (6 - F_{i,SE}) + w_{PS} (6 - F_{i,PS})$$

$$\text{PRI}_{\text{true}, i} = \text{clip}\left( \frac{\bar{R}_i - 1.0}{4.0} \times 100.0 + \epsilon_{\text{pri}}, \, 0.0, \, 100.0 \right)$$

Weights: $\mathbf{w} = [0.28, 0.22, 0.18, 0.18, 0.14]$. Discretization cutoffs yield:
- **Low Risk ($\le 34.0$):** 18.4%
- **Moderate Risk ($34.0 < \text{PRI} \le 66.0$):** 56.7%
- **High Risk ($> 66.0$):** 24.8%

---

## 4. Limitations of Synthetic Data (Strict Scientific Disclosure)

In strict accordance with Rule 6 of our Project Governance:
1. **Lack of Epidemiological / Clinical Validity:** Synthetic data represents mathematically specified Gaussian priors and linear threshold models. It does **not** prove real-world psychological construct validity or psychological generalizability.
2. **Simplified Confounding:** While unobserved Gaussian noise is injected, real human student attrition involves complex, non-linear longitudinal dynamics, institutional policy shifts, and external economic shocks that cannot be fully captured in cross-sectional synthetic data.
3. **Absence of Real Cultural Idioms of Distress:** The synthetic data assumes standard Likert interpretation. Real survey responses in Sri Lankan higher education may reflect cultural response styles (e.g. central tendency bias or reluctance to express negative sentiment about faculty).
4. **Purpose Boundary:** These datasets serve exclusively to verify data pipelines, validate anti-leakage cross-validation harnesses, test API contracts, and calibrate explainability routines before deploying onto IRB-cleared real student cohorts.

---

## 5. How to Defend This in an Academic Viva

When viva examiners question the use of synthetic data, present the following defense:

> *"Esteemed Examiners, our research methodology strictly distinguishes between **architectural validation** and **empirical clinical claim**.*
>
> 1. *Developing an end-to-end ML pipeline—including nested cross-validation, TreeSHAP explainability, and multi-modal Fusion API contracts—directly on real survey data without a verified contract risks silent data leakage, overfitting to small cohorts, and privacy breaches.*
> 2. *Our synthetic data generator is not a naive random uniform sampler. It is a psychometrically calibrated **probabilistic latent simulation** using Multivariate Gaussian Copulas, verified internal reliability coefficients ($\alpha \in [0.80, 0.84]$ matching educational psychology benchmarks), realistic item-level noise, and unobserved confounders.*
> 3. *We deliberately generated a small $N=300$ version to simulate sample sizes typical of preliminary university pilot surveys, and a corrupted 'dirty' dataset to prove our schema validator's ability to reject out-of-range, non-integer, and duplicate records.*
> 4. *All models and pipelines developed herein are purely configuration-driven; once IRB-cleared institutional survey data is ingested, the exact same pipeline executes seamlessly without modifying a single line of feature extraction or model code."*

---

## 6. Generated Profiling Artifacts

The following diagnostic charts have been generated and permanently saved to `/artifacts/plots/`:
- `artifacts/plots/correlation_heatmap.png`: Empirical correlation matrix across all 15 survey items showing distinct intra-factor clustering and inter-factor correlations.
- `artifacts/plots/factor_distributions.png`: Kernel Density Estimates and histograms for the 5 latent factors.
- `artifacts/plots/pri_distribution.png`: Stacked distribution of continuous PRI scores across Low, Moderate, and High risk categories.
- `artifacts/plots/attrition_class_balance.png`: Attrition outcome distribution (26.5% attrited) and 5-point dropout intention counts.
