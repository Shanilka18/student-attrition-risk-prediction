# Phase 3 Documentation: Preprocessing, Psychometric Reliability Analysis, Feature Engineering & Anti-Leakage Splits

> **RESEARCH NOTICE & ETHICAL DISCLAIMER**  
> The Psychosocial Risk Index (PRI) is an academic decision-support indicator developed for educational analytics research at SLIIT Kandy, Sri Lanka. It is **not** a psychiatric diagnosis or clinical psychological evaluation. All analyses utilize pseudonymized Study IDs (`study_id`).

---

## 1. Executive Summary & Objective

Before supervised machine learning models or tree ensembles can be trained on psychometric survey data, two foundational scientific questions must be rigorously resolved:
1. **Instrument Reliability:** Do the three constituent survey items for each latent factor reliably measure the same underlying construct?
2. **Methodological Leakage Prevention:** Are imputers, normalizers, and scalers strictly isolated to training folds so that validation and test sets remain uncontaminated?

Phase 3 implements an end-to-end preprocessing, psychometric validation, and feature engineering pipeline:
- Automated cleaning, straight-liner identification, and removal.
- Empirical comparison of **Median vs. Iterative MICE imputation** (fitted strictly on train).
- Full psychometric reliability suite: **Cronbach’s $\alpha$**, **$\alpha$-if-item-deleted**, **corrected item-total correlations ($r_{it}$)**, **McDonald’s $\omega$**, and inter-factor correlation matrices saved to `/artifacts/metrics/reliability.json`.
- Psychometric feature engineering: 5 raw factor scores, standardized z-scores, risk-oriented inversions, and cognitive disparity interactions.
- Stratified 70/15/15 Train/Val/Test partitioning with a **code-enforced sealed test set guard** (`SealedTestAccessError`).
- Serialized production `preprocessor.joblib` pipeline for reproducible inference on new uploads.

---

## 2. Simple Explanation (Explain Like I am 5 Years Old)

### Analogy 1: What is Cronbach's Alpha? "The Three Thermometers in the Same Pot"

Imagine you are boiling water on a stove and you want to know if the water is boiling hot ($100^\circ\text{C}$). To be certain, you dip **three different thermometers** into the exact same pot:
- If Thermometer 1 says $99^\circ\text{C}$, Thermometer 2 says $101^\circ\text{C}$, and Thermometer 3 says $100^\circ\text{C}$, you smile! All three thermometers agree with each other. You have **High Reliability ($\alpha > 0.80$)**.
- But if Thermometer 1 says $100^\circ\text{C}$, Thermometer 2 says $25^\circ\text{C}$ (ice cold!), and Thermometer 3 says $180^\circ\text{C}$, you would be alarmed. The three tools do not agree; one or more of them is broken. You have **Low Reliability ($\alpha < 0.70$)**.

In our student questionnaire, we have **three separate questions measuring Stress** (`S1`, `S2`, `S3`). **Cronbach's alpha asks a very simple question:**  
*"If a student is genuinely feeling stressed, do their answers across all three stress questions tell the same story?"*  
When $\alpha \ge 0.70$, it proves mathematically that the three questions are working together like a synchronized team of thermometers!

### Analogy 2: What is Alpha-If-Item-Deleted? "Finding the Confused Witness"

Imagine three detectives interview three eyewitnesses to a car crash:
- Witness A and Witness B both say: *"A red sports car ran the red light."*
- Witness C says: *"I think an elephant was eating a banana on the road."*

If the detectives calculate how reliable the testimony is **without Witness C**, the reliability shoots way up! That is **$\alpha$-if-item-deleted**: it tests whether removing any single question makes the overall score clearer. If removing a question causes the score to drop, it proves that question was valuable and necessary.

### Analogy 3: Why Lock the Test Set? "The Sealed Exam Envelope"

Imagine a teacher who is preparing the final exam for a class. If the teacher accidentally leaves the exam paper on the table before the exam, students will look at the answers and get 100%. But they didn't actually learn—they just memorized the test!

In machine learning, this mistake is called **Data Leakage**. If our model gets to see or calculate averages from the test set during training, its accuracy will look artificially perfect, but it will fail on real students. To stop this, our code puts the test set inside a **Locked Digital Vault** with an alarm (`SealedTestAccessError`). The test set can only be opened once at the very end to give an honest, uncheated grade.

---

## 3. Technical Explanation (For Viva Panel & ML Specialists)

### 3.1 Cleaning Pipeline & Straight-Lining Isolation

Survey satisficing occurs when respondents select an unvarying Likert choice across all battery items. Using the intra-individual variance metric:

$$s_i^2 = \frac{1}{K - 1} \sum_{j=1}^{K} (x_{i,j} - \bar{x}_i)^2 = 0.0, \quad K = 15$$

The pipeline detected and removed **18 straight-lining respondents** from the raw $N=3{,}000$ cohort, yielding an effective sample of $N=2{,}982$ authentic records.

### 3.2 Imputation Strategy Benchmark: Median vs. Iterative MICE

To resolve missing values without data leakage, imputers were fitted **strictly on the training split ($N=2{,}086$)**. To select the optimal imputation strategy, an empirical reconstruction test was conducted on validation data with 10% artificial masking:

$$\text{MAE} = \frac{1}{|\mathcal{M}|} \sum_{(i,j) \in \mathcal{M}} |x_{i,j}^{\text{imputed}} - x_{i,j}^{\text{true}}|$$

| Imputation Method | Algorithm | Reconstruction MAE | Recommendation |
| :--- | :--- | :--- | :--- |
| **Simple Imputer** | Column-wise univariate Median | $1.0367$ | Sub-optimal |
| **Iterative Imputer (MICE)** | Multivariate Chained Equations (Bayesian Ridge) | **$0.7743$** | **Selected ($\approx 25\%$ error reduction)** |

**Mathematical Justification:** Psychometric Likert items exhibit non-zero covariance ($\boldsymbol{\Sigma}_{\text{items}} \ne \mathbf{I}$). While univariate median imputation flattens variance and underestimates item correlations, **MICE preserves inter-item covariance structures** by iteratively modeling each item as a linear combination of all other observed items.

### 3.3 Psychometric Reliability Foundations

#### 3.3.1 Cronbach's Alpha ($\alpha$)
For each subscale consisting of $k=3$ items:

$$\alpha = \frac{k}{k - 1} \left( 1 - \frac{\sum_{j=1}^k \sigma_j^2}{\sigma_X^2} \right), \quad X = \sum_{j=1}^k y_j$$

#### 3.3.2 Corrected Item-Total Correlation ($r_{it}$)
Quantifies the correlation of individual item $y_j$ with the sum of the remaining $k-1$ items:

$$r_{j, (X - y_j)} = \frac{\text{Cov}(y_j, X - y_j)}{\sigma_{y_j} \cdot \sigma_{X - y_j}}$$

Items with $r_{it} \ge 0.30$ are deemed acceptable; values $\ge 0.50$ indicate high discrimination.

#### 3.3.3 McDonald's Omega ($\omega_t$)
Because Cronbach's $\alpha$ assumes tau-equivalence (equal factor loadings across all items) and can underestimate reliability under congeneric models, McDonald's $\omega$ is computed using principal factor loadings $\lambda_j$:

$$\omega_t = \frac{\left( \sum_{j=1}^k \lambda_j \right)^2}{\left( \sum_{j=1}^k \lambda_j \right)^2 + \sum_{j=1}^k (1 - \lambda_j^2)}$$

### 3.4 Empirical Psychometric Reliability Audit Table

*Data extracted directly from persistent artifact [`artifacts/metrics/reliability.json`](file:///d:/SLIIT/Y4S1/Research_Project/artifacts/metrics/reliability.json) on Training Split ($N=2{,}086$):*

| Factor Construct | Item | Item Mean $\pm$ SD | Item-Total $r_{it}$ | $\alpha$-If-Deleted | Factor $\alpha$ | Factor $\omega$ | Viva Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stress** | `S1`<br>`S2`<br>`S3` | $3.190 \pm 1.224$<br>$3.228 \pm 1.235$<br>$3.224 \pm 1.245$ | $0.6912$<br>$0.6939$<br>$0.6925$ | $0.7698$<br>$0.7671$<br>$0.7686$ | **$0.8328$** | **$0.8997$** | **EXCELLENT** |
| **Motivation** | `M1`<br>`M2`<br>`M3` | $3.042 \pm 1.235$<br>$3.029 \pm 1.252$<br>$3.013 \pm 1.248$ | $0.7198$<br>$0.7165$<br>$0.6927$ | $0.7730$<br>$0.7762$<br>$0.7991$ | **$0.8439$** | **$0.9058$** | **EXCELLENT** |
| **Belongingness** | `B1`<br>`B2`<br>`B3` | $3.084 \pm 1.246$<br>$3.044 \pm 1.243$<br>$3.076 \pm 1.257$ | $0.7095$<br>$0.7067$<br>$0.7119$ | $0.7797$<br>$0.7824$<br>$0.7778$ | **$0.8411$** | **$0.9042$** | **EXCELLENT** |
| **Self-Efficacy** | `SE1`<br>`SE2`<br>`SE3` | $3.090 \pm 1.240$<br>$3.096 \pm 1.247$<br>$3.064 \pm 1.258$ | $0.7126$<br>$0.7188$<br>$0.7162$ | $0.7858$<br>$0.7806$<br>$0.7827$ | **$0.8455$** | **$0.9066$** | **EXCELLENT** |
| **Peer Support** | `PS1`<br>`PS2`<br>`PS3` | $3.085 \pm 1.230$<br>$3.072 \pm 1.229$<br>$3.047 \pm 1.218$ | $0.6868$<br>$0.6826$<br>$0.6806$ | $0.7588$<br>$0.7628$<br>$0.7645$ | **$0.8263$** | **$0.8962$** | **EXCELLENT** |

**Key Psychometric Observations:**
1. **Unanimous Reliability Pass:** All five factors significantly exceed the standard psychometric threshold of $\alpha \ge 0.70$ (ranging from $0.826$ to $0.845$).
2. **Item Discrimination:** All corrected item-total correlations $r_{it} \approx 0.68 - 0.72$, demonstrating strong construct validity.
3. **No Redundant Items:** For every factor, $\alpha$-if-item-deleted is lower than the factor $\alpha$ (e.g. deleting `S1` drops stress alpha from $0.8328$ to $0.7698$), proving that all 15 questions contribute positively to scale reliability.

---

### 3.5 Inter-Factor Correlation Structure

*Empirical correlation matrix across factor mean scores ($N=2{,}086$):*

| Construct | Stress | Motivation | Belongingness | Self-Efficacy | Peer Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stress** | $1.0000$ | $-0.3204$ | $-0.2977$ | $-0.3704$ | $-0.2408$ |
| **Motivation** | $-0.3204$ | $1.0000$ | $0.3809$ | $0.4678$ | $0.2941$ |
| **Belongingness** | $-0.2977$ | $0.3809$ | $1.0000$ | $0.3392$ | $0.4907$ |
| **Self-Efficacy** | $-0.3704$ | $0.4678$ | $0.3392$ | $1.0000$ | $0.3188$ |
| **Peer Support** | $-0.2408$ | $0.2941$ | $0.4907$ | $0.3188$ | $1.0000$ |

The observed correlations mirror authentic psychological phenomena: Stress shares moderate negative correlations with Self-Efficacy ($r = -0.370$) and Motivation ($r = -0.320$), while Belongingness and Peer Support share strong positive integration ($r = +0.491$).

---

### 3.6 Anti-Leakage Partitioning & Sealed Test Guard

The dataset was partitioned using stratified sampling on `risk_category` with fixed random seed $42$:
- **Training Set (`data/processed/train.csv`):** $2{,}086$ rows ($70.0\%$).
- **Validation Set (`data/processed/val.csv`):** $448$ rows ($15.0\%$).
- **Holdout Test Set (`data/processed/test.csv`):** $448$ rows ($15.0\%$).

#### The Code-Enforced Test Lock
To satisfy Rule 3 of our Project Governance, `src/data/splits.py` implements a runtime exception guard:
```python
def load_split(split_name: str, processed_dir: str = "data/processed", allow_test: bool = False):
    if split_name == "test" and not allow_test:
        raise SealedTestAccessError(
            "TEST SET ACCESS VIOLATION: The holdout test set is sealed to prevent data leakage. "
            "It must not be loaded during feature selection, model training, or hyperparameter tuning."
        )
    ...
```
Any training or hyperparameter optimization script that attempts to read `test.csv` without passing `allow_test=True` is halted immediately.

---

### 3.7 Feature Engineering & Configurable Benchmarking Matrices

The module `src/features/build_features.py` extracts decoupled feature subsets to enable head-to-head empirical comparisons during model benchmarking:
1. **`raw_items` ($p=15$):** $S1 \dots S3, M1 \dots M3, B1 \dots B3, SE1 \dots SE3, PS1 \dots PS3$.
2. **`factor_scores` ($p=5$):** $\bar{F}_{\text{stress}}, \bar{F}_{\text{mot}}, \bar{F}_{\text{bel}}, \bar{F}_{\text{se}}, \bar{F}_{\text{peer}}$.
3. **`risk_factors` ($p=5$):** Protective factors inverted to common risk orientation ($6 - F$).
4. **`interactions` ($p=5$):**
   - $\text{stress\_minus\_efficacy} = F_{\text{stress}} - F_{\text{se}}$
   - $\text{stress\_to\_support\_ratio} = F_{\text{stress}} / (F_{\text{peer}} + 0.01)$
   - $\text{protective\_mean} = \text{mean}(M, B, SE, PS)$
   - $\text{resilience\_deficit} = (6 - \text{protective\_mean}) - F_{\text{stress}}$
   - $\text{stress\_motivation\_interaction} = F_{\text{stress}} \times (6 - F_{\text{mot}})$
5. **`combined` ($p=25$):** Raw items + factor scores + interaction disparity features.

---

### 3.8 Production Serialization Contract

The entire feature transformation and MICE imputation logic is encapsulated in a Scikit-Learn Pipeline (`PsychometricFeatureTransformer` + `IterativeImputer`) and serialized to:
```text
artifacts/models/preprocessor.joblib
```
When new student survey responses are submitted via the FastAPI endpoint or UI CSV upload, the exact same fitted pipeline transforms the inputs without re-fitting or parameter shifts.
