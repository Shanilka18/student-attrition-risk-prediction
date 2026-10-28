# Architecture and Engineering Decision Log

Every design, architectural, and methodological decision is recorded here chronologically as required by Global Rule G11.

---

### D-001: Adoption of PROJECT_BRIEF.md Survey Data Truth Over Proposal LMS Logs
- **Date**: 2026-10-26
- **Decision**: Adopt the closed-ended 4-period, 7-indicator questionnaire schema as the sole ground truth for behavioral data, superseding outdated proposal descriptions of Moodle/LMS/Zoom/mouse server telemetry.
- **Reason**: Proposal descriptions were written prior to data governance constraints. Retrospective closed-ended questionnaires allow capturing multidimensional engagement (e.g. offline study time, review regularity) without relying on restricted university IT server access or intrusive telemetry.
- **Alternatives Rejected**:
  - Waiting for LMS administrator database dumps: Rejected due to privacy regulations and non-feasibility across faculties.
  - Hybrid server logging: Rejected as inconsistent with the unified questionnaire administered across all 4 project components.

---

### D-002: PyTorch CPU Fallback for Deep Learning Layer
- **Date**: 2026-10-26
- **Decision**: Install and pin PyTorch CPU (`torch==2.14.1`) as the deep learning backend instead of TensorFlow CPU.
- **Reason**: TensorFlow CPU on Windows 64-bit relies on `tensorflow-intel` (~300.9 MB) which repeatedly stalled during download and network handshake on this system. PyTorch CPU installed cleanly, operates reliably within the 8 GB RAM budget on Windows with no GPU, and maintains parity with the architecture of sibling project components.
- **Alternatives Rejected**:
  - TensorFlow GPU: Machine lacks a dedicated CUDA GPU.
  - TensorFlow CPU via forced mirrors: Stalled during pip wheel transport.

---

### D-003: Portable Pathlib Architecture and Native PowerShell Task Runner
- **Date**: 2026-10-26
- **Decision**: Use `pathlib.Path` relative resolution exclusively across all modules and scripts, and provide `scripts/tasks.ps1` alongside `Makefile`.
- **Reason**: The developer machine is running Windows without native GNU Make installed by default. A dedicated PowerShell task runner ensures full local CLI execution parity on Windows without requiring WSL or MinGW.
- **Alternatives Rejected**:
  - Requiring developers to install GNU Make via Chocolatey: Unnecessary external dependency friction.
  - Absolute path strings: Violates project portability rules.

---

### D-004: Direct Isolation of Survey Outcome Items A1–A3
- **Date**: 2026-10-26
- **Decision**: Enforce an automated configuration assertion and unit test verifying that outcome items $A_1\text{–}A_3$ cannot intersect with indicator features.
- **Reason**: Global Rule G3 mandates zero target leakage. Since $A_1\text{–}A_3$ define the attrition intent targets, any inclusion in the feature engineering pipeline would invalidate research claims.
- **Alternatives Rejected**:
  - Manual review of feature columns: Prone to human oversight.

---

### D-005: Canonical 35-Column Survey Schema Specification
- **Date**: 2026-10-26
- **Decision**: Establish exactly 35 canonical columns in `config/schema.yaml`: 1 `id`, 1 `meta`, 1 `consent`, 28 `feature` ($7 \text{ indicators} \times 4 \text{ periods}$), 1 `attention_check`, and 3 `outcome` items (`intent_leave`, `intent_defer`, `intent_not_continue`).
- **Reason**: Unified ground truth across sample simulation, real survey upload, data cleaning, and web UI. Fixed column naming (`<indicator>_p<k>`) ensures consistent feature extraction and reproducibility.
- **Alternatives Rejected**:
  - Wide raw Google Forms headers as column names: Unwieldy and brittle across survey revisions.
  - Multi-index column structures: Complicates export to standard CSV/Excel and interoperability with sibling components.

---

### D-006: RapidFuzz Suggestion-Only Mapping Policy
- **Date**: 2026-10-26
- **Decision**: Restrict RapidFuzz header matching (similarity threshold 88.0%) to suggestion mode only (`auto_accept: false` in `config/mapping.yaml`), requiring human confirmation for any unmatched column.
- **Reason**: Prevents silent column corruption or misattribution of temporal indicators (e.g. confusing Period 1 with Period 2 due to high string overlap in long question titles).
- **Alternatives Rejected**:
  - Silent auto-accept above threshold: High risk of mislabeling similar questions.
  - Strict exact-matching only: Fails on trivial punctuation or spacing differences produced by Google Forms exports.

---

### D-007: Resilient Normalization in Survey Codec
- **Date**: 2026-10-26
- **Decision**: Implement `normalize_label()` in `src/mei/io/codec.py` to standardize unicode dashes (en-dash `\u2013`, em-dash `\u2014`, minus `\u2212`), convert smart quotes, collapse whitespace, and apply case-folding prior to code lookup.
- **Reason**: Google Forms exports and manual typing often introduce typographic differences (e.g., `1–2 days` with en-dash vs `1-2 days` with ASCII hyphen). Standardizing these variations prevents spurious encoding errors.
- **Alternatives Rejected**:
  - Strict exact string lookup: Brittle and causes unnecessary data dropouts during cleaning.

---

### D-008: ASCII-Safe Console Output for Windows Shell Interoperability
- **Date**: 2026-10-26
- **Decision**: Use ASCII markers (e.g. `[SUCCESS]` and `x`) in CLI Rich console tables instead of Unicode checkmarks (`\u2713`) or multiplication symbols (`\u00d7`).
- **Reason**: Standard Windows console shells default to legacy code pages (e.g., `cp1252` or `cp437`), throwing `UnicodeEncodeError` when unmapped characters are emitted.
- **Alternatives Rejected**:
  - Requiring users to configure `chcp 65001` before running the CLI: Unfriendly developer experience.

---

### D-009: Latent Variable Data Generation Model with Mixed Behavioral Archetypes
- **Date**: 2026-10-26
- **Decision**: Generate synthetic survey responses from an unobserved 4-period continuous latent engagement process $e_t \in [0, 1]$ parameterized across 6 realistic behavioral archetypes (`consistent_high` 28%, `consistent_moderate` 24%, `consistently_low` 10%, `declining` 17%, `sporadic` 13%, `improving` 8%), incorporating individual diligence random effects ($\sigma = 0.08$) and indicator-specific noise ($\sigma = 0.10$).
- **Reason**: Pure independent Gaussian noise lacks realistic psychological covariance and temporal autocorrelation. Grounding indicator responses in latent trajectories produces authentic covariance across time and items, providing a credible testbed for downstream clustering, Hidden Markov Models, and predictive models.
- **Alternatives Rejected**:
  - Independent marginal sampling: Fails to capture temporal autocorrelation and inter-indicator correlation.
  - Fully deterministic trajectory templates: Lacks natural student heterogeneity and produces artificially separated clusters.

---

### D-010: Signal Strength Calibration Strategy for Controlled Difficulty
- **Date**: 2026-10-26
- **Decision**: Dynamically calibrate the outcome model's `signal_strength` parameter using 5-fold cross-validation until LightGBM achieves an AUC-ROC in the realistic range $[0.86, 0.92]$, with intercept $w_0$ tuned to hold positive at-risk prevalence at $26\%\text{–}28\%$.
- **Reason**: Synthetic data that produces trivial separations (e.g., AUC $> 0.98$) gives false confidence and fails to expose real pipeline weaknesses. Conversely, weak signal (AUC $< 0.75$) frustrates methodological benchmarking. Calibrating into $[0.86, 0.92]$ ensures the challenge level mirrors genuine behavioral survey research.
- **Alternatives Rejected**:
  - Hard-coded static coefficients: Produces unpredictable difficulty shifts whenever indicator noise or cut-points are modified.
  - Relying exclusively on linear logistic regression for calibration: Fails to detect non-linear temporal interactions ($\max(0, e_1 - e_4)$ and volatility $\text{std}(e)$).

---

### D-011: Strict Isolation of Ground Truth Artifacts from Feature Pipelines
- **Date**: 2026-10-26
- **Decision**: Persist latent states and true archetypes to `data/raw/ground_truth_hidden.csv` strictly for downstream Phase 5 HMM/clustering recovery evaluation, and enforce automated test assertions that no column in this file can be accessed by feature engineering or model training pipelines.
- **Reason**: Comply with Global Rule G3 (zero leakage). Real survey deployments never possess true latent engagement values; models must rely solely on observed, noisy, discretized questionnaire text responses.
- **Alternatives Rejected**:
  - Storing ground truth alongside raw export data: High risk of accidental feature leakage.

---

### D-012: Evaluation Upload Artifact Dual-Split (Clean vs. Messy)
- **Date**: 2026-10-26
- **Decision**: Generate two distinct sample test files in `data/samples/`: `sample_upload_clean.xlsx` (200 students, canonical headers, no outcomes, clean data) for scoring engine tests, and `sample_upload_messy.csv` (200 students, variant headers, blanks, out-of-range values, extra unknown column) for validation and column mapping stress tests.
- **Reason**: Separation of concerns: testing model inference requires clean inputs, whereas testing data ingestion, column mapping, and schema validation requires intentionally malformed headers and out-of-range entries.
- **Alternatives Rejected**:
  - Single evaluation file: Conflates mapping errors with model prediction errors during UI testing.

---

### D-013: Fastparquet Engine for Apache Parquet Persistence on NumPy 1.26.4
- **Date**: 2026-10-26
- **Decision**: Adopt `fastparquet` (pinned at `2024.11.0` with `cramjam` `2.9.1`) as the default parquet serialization backend instead of `pyarrow`.
- **Reason**: `pyarrow>=18.0.0` strictly requires NumPy 2.0+ on Windows, whereas our project environment is pinned to `numpy==1.26.4` for scientific computing compatibility (scipy, statsmodels, hmmlearn). `fastparquet` seamlessly integrates with pandas `to_parquet` and `read_parquet` under NumPy 1.26.4 without ABI incompatibilities or memory overhead.
- **Alternatives Rejected**:
  - Upgrading NumPy to 2.x: Breaks pinned C-extension packages across the workspace.
  - Downgrading pyarrow to legacy builds: Unreliable binary wheels on Python 3.11 Windows.
  - Storing interim data solely as CSV: Loses data types, increases disk footprint 4x, and slows down downstream pipeline reads.

---

### D-014: Zero-Imputation Policy in the Cleaning Pipeline (Strict Rule G3 Compliance)
- **Date**: 2026-10-26
- **Decision**: In the cleaning stage, filter out student rows with excessive feature missingness (> 25% of the 28 indicator cells), but leave all remaining missing values as `NaN` (unimputed) in `data/interim/clean_<run_id>.parquet`.
- **Reason**: Strictly adhere to Global Rule G3 (leakage prevention). Imputing missing values prior to train/test partitioning or cross-validation folds leaks target and validation distribution information across splits. Imputation transformers must be fitted strictly inside the downstream feature/model cross-validation pipeline.
- **Alternatives Rejected**:
  - Mean/median/mode imputation at cleaning stage: Target and validation leakage across CV splits.
  - KNN / MICE imputation at cleaning stage: Severe computational overhead and global distribution leakage across test sets.

---

### D-015: Dual-Mode Contract: Dropping Invalid Rows in Training vs. Detailed Audit Rejection in Scoring
- **Date**: 2026-10-26
- **Decision**: Implement two operational modes for `CleaningPipeline`:
  - `training`: Drops rows violating consent, attention checks, straight-lining, excessive missingness (> 25%), or missing outcome labels, returning only clean, model-ready training records.
  - `scoring`: Never silently drops survey submissions. It attempts to parse every row, marks un-scorable submissions (e.g. straight-lining, missing student ID, excessive missingness) in the output metadata with explicit human-readable rejection reasons, and preserves rows lacking outcome labels (since prospective students do not have attrition labels).
- **Reason**: Production scoring and website uploads must never drop student records without auditing. Academic advisors and administrators must see why a student could not be scored rather than wondering why an uploaded record vanished.
- **Alternatives Rejected**:
  - Single mode dropping all invalid rows: Disastrous for web app users who lose uploaded student rows without feedback.
  - Always returning all rows in training mode: Corrupts model training with noise and invalid consent data.

---

### D-016: Two-Tiered Careless Responding Detection (Straight-Lining Heuristics)
- **Date**: 2026-10-26
- **Decision**: Flag straight-lining careless responses using a dual-criterion check across the 28 canonical indicator features: (1) Total zero variance across all available answers ($\text{std} = 0$), or (2) Zero variance across $\ge 6$ of the 7 individual indicators across all 4 periods.
- **Reason**: Students completing survey grids carelessly often pick the same column down an entire page or repeat the exact same pattern for almost every construct. Detecting both whole-survey invariance and high indicator-level invariance filters out unengaged respondents before they distort latent state inference and predictive models.
- **Alternatives Rejected**:
  - Checking only global variance: Misses respondents who alternate a single column on one question but straight-line the remaining 6 indicators.
  - Outlier Mahalanobis distance / Person-fit statistics: Computationally expensive on 8 GB RAM machines and sensitive to missing values.

---

### D-017: Three-Way Stratified Outer Split (70/15/15) with Complete Test Quarantining
- **Date**: 2026-10-26
- **Decision**: Partition the cohort into 70% Train, 15% Validation, and 15% Test sets stratified by `at_risk`. Quarantined test indices with a programmatic caller stack guard that blocks read access unless explicitly invoked by the Phase 9 final evaluation function `evaluate_final_on_test()`.
- **Reason**: Enforce Global Rule G3 (zero leakage). Peeking at test set performance during feature selection or model hyperparameter tuning causes optimistic selection bias and compromises academic generalizability.
- **Alternatives Rejected**:
  - 80/20 train/test split without dedicated validation set: Forces model hyperparameter selection on cross-validation alone without an intermediate validation barrier.
  - Unquarantined test indices: Prone to accidental evaluation leaks across developmental phases.

---

### D-018: 15-Fold Repeated Stratified Cross-Validation Scheme with Small-Cohort Extension
- **Date**: 2026-10-26
- **Decision**: Standardize all model leaderboard evaluations on a precomputed Repeated Stratified 5-Fold $\times$ 3 repeats (15 folds) cross-validation scheme fitted on Train+Validation ($85\%$ of cohort), with serialized fold indices in `artifacts/runs/<run_id>/splits/cv_folds.json`. Automatically switch to 5 splits $\times$ 5 repeats (25 folds) if cohort size $N < 500$.
- **Reason**: Fair benchmarking across diverse model families (linear, tree-based, deep learning, ensembles) requires identical validation data slices. Small cohorts suffer from high split-to-split variance, which 5 repeats effectively stabilizes.
- **Alternatives Rejected**:
  - Single 5-fold CV: Too noisy on $N \approx 2000$ to reliably detect subtle performance gains ($< 0.01$ AUC).
  - Dynamic random splits per model: Yields non-comparable testbeds across model categories.

---

### D-019: Imputation Policy: Median with Missing Indicators vs. Student Interpolation
- **Date**: 2026-10-26
- **Decision**: Select column median imputation with binary missing-indicator columns (for features with $> 3\%$ missingness) as the default pipeline imputer over within-student temporal interpolation.
- **Reason**: Empirical 15-fold cross-validation benchmarking with Logistic Regression demonstrated virtually identical predictive accuracy ($0.8717 \pm 0.0090$ AUC for median vs. $0.8716 \pm 0.0100$ AUC for within-student interpolation), while median imputation is $10\times$ faster, vectorizable across 8 GB RAM machines, and preserves discrete ordinal bounds without synthetic non-integer drift.
- **Alternatives Rejected**:
  - Within-student interpolation: Added computational overhead with no measurable AUC gain.
  - Global mean imputation: Distorts ordinal categorical boundaries.
  - KNN / MICE imputation: Severe computational cost and high risk of cross-sample distribution leakage.

---

### D-020: 68-Feature Longitudinal Dynamics and Composite Trajectory Engineering
- **Date**: 2026-10-26
- **Decision**: Engineer 56 indicator-specific dynamic features (mean, std, min, max, OLS slope, last-minus-first, last-minus-mean(first two), direction reversals across 7 indicators) plus 12 composite engagement trajectory features ($E_1..E_4$, mean, std, slope, max single-period drop, reversals, AUC, last value, drop from best to last).
- **Reason**: Self-reported survey data capture static baseline levels imperfectly due to individual subjective scales, but *temporal changes* (downward trends, sudden drops, volatility) provide powerful attrition signals. 15-fold CV confirmed that feature engineering delivers significant performance gains for both linear models (Logistic Regression AUC $+0.0060$) and tree models (LightGBM AUC $+0.0070$).
- **Alternatives Rejected**:
  - Raw 28 features only: Misses non-linear temporal dynamics and momentum.
  - Polynomial feature expansion: Creates $> 400$ features with severe multicollinearity and high risk of overfitting.

---

### D-021: Clustering Model Selection via Bootstrap Stability and Silhouette Thresholding
- **Date**: 2026-10-26
- **Decision**: Evaluate KMeans, Gaussian Mixture Models (diagonal and full covariance), and Agglomerative Ward clustering across $k \in [2, 6]$ on standardized composite trajectory features ($E_1..E_4$, slope, volatility, level). Select the model achieving the highest bootstrap stability (mean pairwise Adjusted Rand Index over 30 resamples) among candidates satisfying silhouette $\ge 0.15$ (ties broken in favor of smaller $k$). Name clusters by their level and trajectory profile.
- **Reason**: Clustering on short longitudinal sequences is susceptible to overfitting noise and finding local minima. Bootstrap stability ensures that identified clusters represent robust, generalizable student behavioral profiles rather than sample-specific artifacts.
- **Alternatives Rejected**:
  - Silhouette score alone: Systematically favors spherical $k=2$ splits that obscure distinct declining, improving, or sporadic engagement archetypes.
  - Elbow heuristic on inertia: Subjective, unautomated, and prone to ambiguous inflection points.

---

### D-022: Gaussian HMM Model Selection and Deterministic State Ordering
- **Date**: 2026-10-26
- **Decision**: Fit hmmlearn `GaussianHMM` with diagonal covariance across $K \in \{2, 3, 4\}$ with 10 random restarts on 4-period sequences of the 7 standardized indicators. Select $K$ based on BIC and validation-fold log-likelihood, applying an explicit simplicity preference rule (favoring simpler $K$ if BIC difference $< 10.0$). Deterministically reorder latent states in ascending order of standardized emission composite engagement ($\mu_1 \le \mu_2 \le \dots \le \mu_K$).
- **Reason**: Gaussian HMM parameter estimation suffers from label-switching (arbitrary permutation of latent state indices across random initializations and cross-validation folds). Enforcing deterministic sorting by composite engagement level guarantees consistent, reproducible state semantics (State 1 = Low, State $K$ = High) across all pipeline stages.
- **Alternatives Rejected**:
  - Unconstrained state indexing: Induces catastrophic label-switching across CV folds, invalidating fold-wise feature alignment.
  - Full covariance matrices: Estimating $K \times \frac{7 \times 8}{2}$ parameters on short length-4 sequences leads to singular covariance matrices and severe overfitting.

---

### D-023: Deterministic Rule-Based Trajectory Pattern and Level Classification
- **Date**: 2026-10-26
- **Decision**: Map 4-period Viterbi decoded state paths $(s_1, s_2, s_3, s_4)$ to 4 canonical behavioral patterns (Consistent, Declining, Improving, Sporadic) and 3 overall levels (High, Medium, Low) using transparent, deterministic transition rules.
- **Reason**: Institutional stakeholders, academic advisors, and retention counselors require clear, explainable behavioral descriptions (e.g., "Declining High" or "Sporadic Moderate") rather than abstract Markov state probability vectors.
- **Alternatives Rejected**:
  - Black-box sequence embeddings (e.g., LSTM / RNN latent vectors): Completely lack interpretable categorical definitions required for ethical educational interventions.
  - Secondary clustering on decoded state sequences: Introduces unnecessary stochasticity and non-deterministic groupings.

---

### D-024: Honest Reporting of Latent State Incremental Predictive Value
- **Date**: 2026-10-26
- **Decision**: Conduct an honest 15-fold cross-validation predictive ablation study comparing `agg` vs `agg_states` vs `states_only` across fixed Logistic Regression and LightGBM models, reporting paired Wilcoxon signed-rank test statistics and p-values. Plainly document any marginal incremental gains and acknowledge methodological limitations (e.g., sequences of length $T=4$ afford limited statistical power for transition estimation).
- **Reason**: Strict adherence to Global Rule G2 ("Never invent numbers"). If unsupervised latent states contribute strong standalone signal (`states_only`) but only modest incremental value over rich dynamic slope and volatility features (`agg`), this scientific finding must be reported transparently without artificial hyperparameter tuning.
- **Alternatives Rejected**:
  - Artificial metric tuning to force a large positive delta for `agg_states`.
  - Suppressing the ablation study and asserting without proof that states improve classification.




---

### D-025: Classification Model Zoo Scope and Architectural Taxonomy
- **Date**: 2026-10-26
- **Decision**: Define and benchmark 28 distinct classification architectures spanning 6 distinct families: Baselines (3 models: `single_signal_lr`, `majority_dummy`, `rule_login_imitation`), Linear Models (5 models: `logistic_regression_l2`, `logistic_regression_elasticnet`, `linear_svm_calibrated`, `ridge_calibrated`, `lda`), Classical / Shallow (6 models: `knn`, `gaussian_nb`, `decision_tree`, `svm_rbf`, `mlp_shallow`, `extra_trees`), Tree Ensembles & Boosting (7 models: `random_forest`, `gradient_boosting`, `hist_gradient_boosting`, `xgboost`, `lightgbm`, `catboost`, `adaboost`), Proposed Hybrid Models (5 models: `xgboost_hybrid`, `lightgbm_hybrid`, `catboost_hybrid`, `random_forest_hybrid`, `logistic_regression_hybrid`), and Ensembles (2 models: `voting_soft`, `stacking`). Deep learning architectures and regression models are strictly deferred to Phase 7 and Phase 8.
- **Reason**: Rigorous model benchmarking requires diverse inductive biases (linear log-odds, distance-based manifolds, recursive partitioning, second-order boosting, oblivious trees, and probability blending) to identify the optimal tradeoff between predictive discrimination, calibration, and inference efficiency on survey data.
- **Alternatives Rejected**:
  - Benchmarking only a single popular boosting model (e.g. LightGBM): Fails comparative validation standards and obscures model family strengths.
  - Including uncalibrated raw scores: Violates institutional deployment requirements for probabilistic triage.

---

### D-026: Two-Tier Hyperparameter Optimization Protocol (Optuna on Train Split with Frozen Outer CV)
- **Date**: 2026-10-26
- **Decision**: Implement a two-tier hyperparameter tuning protocol: (1) tune candidate hyperparameters using Optuna (Tree-structured Parzen Estimator, seed 42) with 5-fold cross-validation inside Outer Fold 0 (Train partition, $N=1,484$), subject to model-specific time budgets ($15 \dots 90$s); (2) freeze optimal hyperparameters $\boldsymbol{\theta}^*$ and evaluate out-of-fold generalization across the identical 15 outer cross-validation folds.
- **Reason**: Fully nested double cross-validation ($15 \times 30 = 450$ Optuna trials per model) would require $> 10$ hours of compute across 26 models, exceeding the project's 40-minute wall-clock hardware budget on standard 8 GB RAM laptop hardware. Tuning once on Outer Fold 0 introduces negligible optimism on the outer validation sets while leaving the 15% holdout test partition ($N=328$) completely quarantined until Phase 9.
- **Alternatives Rejected**:
  - Full nested double CV across all 15 outer folds: Computationally infeasible within the 40-minute budget.
  - Manual default hyperparameters without tuning: Unfair to complex algorithms (e.g. XGBoost, Elastic-Net, CatBoost) that require tuning to reach competitive performance.

---

### D-027: Out-of-Fold (OOF) Aggregation and Optimal Decision Threshold Tuning
- **Date**: 2026-10-26
- **Decision**: Collect out-of-fold predicted probabilities $\hat{p}_i^{\text{OOF}}$ across the 15 CV folds and tune the decision threshold $\theta^* \in [0.05, 0.95]$ to maximize the $F_1$ score of the minority at-risk class. Report all classification metrics at both the conventional default threshold ($\theta = 0.50$) and the empirical optimal threshold ($\theta^*$).
- **Reason**: Survey datasets exhibit moderate class imbalance ($\sim 25\%$ at-risk). Standard $\theta = 0.50$ thresholding artificially depresses recall in cost-sensitive settings where missing an at-risk student is significantly more detrimental than issuing an extra advisory check-in.
- **Alternatives Rejected**:
  - Fixed $\theta = 0.50$ only: Yields suboptimal recall for probabilistic classifiers trained with log-loss.
  - Tuning the threshold on the test set: Constitutes direct data leakage and invalidates test set independence.

---

### D-028: Leaderboard Score Composite Formula with Asymmetric Generalization Gap Penalty
- **Date**: 2026-10-26
- **Decision**: Adopt the official Leaderboard Score formula from §7 of `PROJECT_BRIEF.md`:
  $$\text{Score} = \frac{F_1(\text{optimal}) + \text{AUC-ROC} + \text{Balanced Accuracy} + \frac{\text{MCC} + 1}{2}}{4} - 2 \cdot \max(0, \text{gap} - 0.05)$$
  where $\text{gap} = \text{Train } F_1 - \text{Validation } F_1$. Categorize models into Good fit ($\text{gap} \le 0.05$), Slight overfit ($0.05 < \text{gap} \le 0.10$), and Overfit ($\text{gap} > 0.10$).
- **Reason**: A composite score balances threshold performance ($F_1$, Balanced Accuracy), ranking discrimination (AUC-ROC), and correlation robustness (MCC). The non-linear gap penalty aggressively penalizes models that memorize training noise (such as unconstrained decision trees or high-capacity RBF SVMs), ensuring that leaderboard winners truly generalize to unseen student cohorts.
- **Alternatives Rejected**:
  - Ranking solely by AUC-ROC: Ignores decision threshold calibration and precision-recall operational trade-offs.
  - Unpenalized validation $F_1$: Fails to penalize over-parameterized models on the verge of overfitting.

---

### D-029: In-Fold SMOTE vs Cost-Sensitive Class Weighting Comparison Policy
- **Date**: 2026-10-26
- **Decision**: Benchmark Synthetic Minority Over-sampling Technique (SMOTE) strictly inside training folds via `imblearn.pipeline.Pipeline` against cost-sensitive class weighting (`balanced`) for the top two gradient boosting models, recording results in `smote_comparison.json` and `FIG-CLASS-10`.
- **Reason**: Synthetic interpolation can distort real survey response manifolds by generating fractional ordinal combinations that do not correspond to realistic student behavioral patterns. Evaluating both strategies under identical 15-fold CV conditions empirically verifies whether synthetic sampling adds measurable value over class weighting on short survey sequences.
- **Alternatives Rejected**:
  - Applying SMOTE globally prior to cross-validation: Severe data leakage that artificially inflates CV metrics.
  - Dogmatically adopting SMOTE without empirical comparison to class weighting.

---

### D-030: PyTorch CPU Architecture Selection and Parity Enforcement
- **Date**: 2026-10-27
- **Decision**: Implement all eight deep learning candidate architectures in native PyTorch CPU (`torch==2.14.1`), adhering strictly to Decision D-002 and the project brief's hardware portability mandate (Windows 64-bit, 8 GB RAM, no GPU required).
- **Reason**: PyTorch CPU provides deterministic execution, lightweight memory management, direct tensor manipulation for multidimensional sequence shapes $(B, 4, 7)$, and eliminates the brittle Windows installation and transport issues associated with TensorFlow CPU.
- **Alternatives Rejected**:
  - TensorFlow / Keras CPU: Prone to large transport stalls (`tensorflow-intel` ~300 MB) and higher memory overhead on commodity 8 GB RAM machines.
  - Requiring CUDA GPU: Violates Global Rule G10 (portability across standard student and institution laptops).

---

### D-031: Bi-GRU Selection Over Bi-LSTM for Ultra-Short (4-Step) Behavioral Trajectories
- **Date**: 2026-10-27
- **Decision**: Select Bidirectional Gated Recurrent Unit (Bi-GRU) as the primary bidirectional recurrent architecture over Bidirectional LSTM for modeling 4-period longitudinal engagement sequences.
- **Reason**: On sequences of length $T=4$, LSTM cell state memory is architecturally redundant and introduces $33\%$ more gating parameters ($4(dh + h^2)$ vs $3(dh + h^2)$). Bi-GRU provides symmetrical forward-backward sequence pooling while reducing parameter complexity, mitigating parameter memorization on small survey cohorts.
- **Alternatives Rejected**:
  - Bidirectional LSTM: Unnecessary cell gating overhead and higher propensity to overfit on $N \approx 1,800$ samples.

---

### D-032: Two-Stage Training Protocol with Conditional Stage 2 Acceptance Gate
- **Date**: 2026-10-27
- **Decision**: Enforce a Two-Stage training lifecycle for all neural models:
  - Stage 1: Adam $\text{lr} = 10^{-3}$, max 150 epochs, ReduceLROnPlateau (patience 6, factor 0.5), EarlyStopping (patience 15).
  - Stage 2 (Refinement): Adam $\text{lr} = 10^{-4}$, max 60 epochs, patience 10, initialized from best Stage 1 weights.
  - Acceptance Gate: Stage 2 weights are retained **if and only if** the minimum validation loss in Stage 2 strictly improves upon Stage 1. Otherwise, weights revert to Stage 1 and the run is recorded as "Stage 2 not used".
- **Reason**: Late-stage fine-tuning with small learning rates can overfit or oscillate if learning has already converged. A strict validation loss acceptance gate guarantees monotonic validation loss improvement and prevents fine-tuning degradation.
- **Alternatives Rejected**:
  - Unconditional fine-tuning: Frequently leads to late-stage validation loss drift and overconfidence.
  - Single fixed learning rate: Fails to refine weights around flat local minima.

---

### D-033: Quarantining the Overfit Demonstration Model from Official Leaderboard
- **Date**: 2026-10-27
- **Decision**: Train an unregularized, over-parameterized multi-layer perceptron (512-256-128 units, no dropout, no weight decay, no early stopping, 300 epochs) to produce research figure `FIG-DEEP-10` ("Before vs After Regularization"), but strictly exclude it from the official classification leaderboard.
- **Reason**: The overfit demonstration model serves as an educational and viva defense artifact illustrating catastrophic failure modes. Including an intentionally crippled overfit model on the competitive leaderboard would distort ranking percentiles and category championship comparisons.
- **Alternatives Rejected**:
  - Including the overfit model on the leaderboard: Misleadingly clutters the candidate model ranking.
  - Omitting the overfit demonstration: Misses a critical viva explanation explaining why early stopping and regularization matter.

---

### D-034: Vectorized In-Epoch Training Loss and Probability Logging for CPU Efficiency
- **Date**: 2026-10-27
- **Decision**: Accumulate training loss, predicted probabilities, and ground truth labels directly during the forward-backward training pass to compute training metrics in NumPy vector space, rather than iterating through the training DataLoader a second time in evaluation mode.
- **Reason**: Eliminating the redundant second training pass cuts epoch training latency from $\approx 1.46\text{s}$ to $\approx 0.25\text{s}$ (a $6\times$ speedup on CPU), enabling all 8 deep architectures across 15 CV folds to execute comfortably within the 25-minute wall-clock budget.
- **Alternatives Rejected**:
  - Full redundant evaluation pass every epoch: Exceeds the 25-minute execution time limit.
  - Skipping training metric logging: Violates Global Rule G8 requiring honest, complete learning curve histories.

---

### D-035: Track B Continuous Intent Score Scale and Huber Loss Formulation
- **Date**: 2026-10-27
- **Decision**: Predict `attrition_intent_score` on a continuous scale of 0 to 100. For deep learning models, formulate the loss objective using Huber loss with threshold $\delta = 2.0$ rather than Mean Squared Error (MSE).
- **Reason**: Survey questionnaire ratings frequently exhibit non-Gaussian residual distributions and long-tailed outliers resulting from sporadic student responses. Huber loss behaves quadratically for small residuals ($|y - \hat{y}| \le \delta$) ensuring smooth gradient descent near the optimum, but transitions to linear penalties for large residuals ($|y - \hat{y}| > \delta$), preventing extreme individual responses from destabilizing model weights.
- **Alternatives Rejected**:
  - Pure Mean Squared Error (MSE): Excessively penalizes sporadic survey outliers, causing gradient explosion in deep architectures.
  - Mean Absolute Error (MAE / L1): Lacks continuous differentiability at zero, leading to persistent oscillations around minimum loss.

---

### D-036: Prediction Clipping Policy and In-Range Audit
- **Date**: 2026-10-27
- **Decision**: Post-process continuous model outputs by clipping predictions strictly to $[0.0, 100.0]$. Systematically track and report the clipping frequency (count and percentage) in fold-wise and out-of-fold metrics.
- **Reason**: Linear, polynomial, and neural models may mathematically extrapolate unbounded predictions ($< 0$ or $> 100$) on extreme out-of-distribution feature combinations. Clipping guarantees physical validity for staff facing dashboards, while auditing ensures models are not fundamentally ill-calibrated. In practice, champion models produced 0 clipped predictions (0.0%), confirming that regularization maintained predictions within natural domain bounds.
- **Alternatives Rejected**:
  - Sigmoid output scaling with $(100 \cdot \sigma(z))$: Saturates gradients at the tails (0 and 100), impeding learning for students with extremely low or high engagement.
  - Permitting unbounded raw predictions: Displays nonsensical negative scores (e.g., -5.2) or scores above 100 to academic staff.

---

### D-037: Composite Leaderboard Score for Continuous Intent Regression
- **Date**: 2026-10-27
- **Decision**: Define the Track B Leaderboard Score as $\text{Score} = \text{mean}(R^2_{\text{val}}, \text{Spearman}_{\text{val}}) - 2 \cdot \max(0, R^2_{\text{train}} - R^2_{\text{val}} - 0.05)$.
- **Reason**: Academic retention teams utilize continuous intent scores for two distinct purposes: (1) assessing absolute attrition risk level (quantified by variance explained, $R^2$), and (2) triaging students into a prioritized outreach queue (quantified by rank-order fidelity, Spearman $\rho$). Averaging $R^2$ and $\rho$ rewards models that excel at both tasks, while the overfit penalty aggressively penalizes models whose training $R^2$ exceeds validation $R^2$ by more than 5 percentage points.
- **Alternatives Rejected**:
  - $R^2$ alone: Fails to measure ranking fidelity; a model could have moderate $R^2$ but poor monotonic queue ordering.
  - RMSE or MAE alone: Scale-dependent and does not account for baseline target variance or ordinal triage utility.

---

### D-038: Regression Ensemble Formulation (Average Voting and Stacking Meta-Regressor)
- **Date**: 2026-10-27
- **Decision**: Formulate two ensemble architectures: (1) `voting_regressor`, an unweighted arithmetic mean of out-of-fold predictions from the top 3 individual models (`catboost_regressor`, `lstm_regressor`, `knn_regressor`); and (2) `stacking_regressor`, a meta-learner utilizing `Ridge(alpha=10.0)` trained on out-of-fold continuous predictions across 5 diverse model families (`catboost_regressor`, `lstm_regressor`, `knn_regressor`, `adaboost_regressor`, `elasticnet_regressor`).
- **Reason**: Ensembling models with orthogonal inductive biases (symmetric oblivious decision trees, recurrent memory cells, metric-space neighbors, and L1/L2 linear models) reduces uncorrelated error variance. Stacking achieved the #1 leaderboard ranking ($R^2 = 0.5495$, $\text{Score} = 0.6454$, $\text{RMSE} = 13.33$), demonstrating statistically significant superiority over individual models ($p_{\text{holm}} < 0.002$).
- **Alternatives Rejected**:
  - Uniform averaging across all 35 models: Dilutes predictions with poorly calibrated or overfit candidates.
  - Non-linear meta-learner (e.g., Random Forest / XGBoost meta-learner): Severe risk of second-level overfitting on 1,856 out-of-fold predictions.

---

### D-040: Pre-Declared Champion Selection Rule and Finalist Selection (Phase 9 Step 4)
- **Date**: 2026-10-27
- **Decision**: Pre-declare and log the formal selection rule in `DECISIONS.md` strictly PRIOR to loading or inspecting any holdout test set numbers:
  1. **Candidate Eligibility**: Only Track A models with status `Good fit` (generalization gap $\le 0.05$) or remediated to `Good fit` / `Slight overfit` are eligible for championship consideration.
  2. **Primary Selection Criterion**: The eligible candidate with the highest cross-validation Leaderboard Score wins.
  3. **Tie-Breaker Rule**: If the top two eligible candidates differ by less than one pooled cross-validation standard error ($\Delta \text{Score} < 1 \cdot \text{SE}_{\text{pooled}}$) or if a paired Wilcoxon signed-rank test across the 15 folds is not statistically significant ($p \ge 0.05$), ties are broken in favor of structural parsimony, inference speed, and explainability in this strictly pre-declared hierarchy:
     $$\text{Boosting-Family Hybrid} > \text{Linear} > \text{Other Classical} > \text{Deep Learning} > \text{Stacked Ensembles}$$
  4. **Pre-Evaluation Outcome (Track A)**:
     - Candidate #1: `knn` (k-Nearest Neighbours, classical): $\text{Score} = \mathbf{0.794875}$, Gap $= \mathbf{-0.0202}$ (`Good fit`), $F_1 = 0.6969$, $\text{AUC} = 0.8768$, $\text{MCC} = 0.5852$.
     - Candidate #2: `gru` (Gated Recurrent Unit, deep): $\text{Score} = \mathbf{0.787592}$, Gap $= \mathbf{+0.0065}$ (`Good fit`), $F_1 = 0.6804$, $\text{AUC} = 0.8807$, $\text{MCC} = 0.5620$.
     - Difference: $\Delta \text{Score} = 0.007283$, while pooled $\text{SE} \approx 0.00726$. Paired Wilcoxon test on validation $F_1$ yields $W = 12.0, p = 0.00537 < 0.05$, confirming statistically significant superiority for `knn`.
     - Furthermore, under the parsimony tie-breaker order, `Other Classical` (`knn`) strictly precedes `Deep Learning` (`gru`).
     - **Declared Champion**: `knn` (k-Nearest Neighbours).
     - **Declared Runner-Up**: `gru` (Gated Recurrent Unit).
  5. **Pre-Evaluation Outcome (Track B)**:
     - **Declared Champion**: `stacking_regressor` ($\text{Score} = \mathbf{0.645434}$, $R^2 = \mathbf{0.549519}$, $\text{RMSE} = \mathbf{13.3261}$, `Good fit`).
     - **Declared Runner-Up**: `voting_regressor` ($\text{Score} = \mathbf{0.645007}$, $R^2 = \mathbf{0.549362}$, `Good fit`).
- **Reason**: Obviates p-hacking, confirmation bias, and post-hoc rationalization. Writing the selection criteria and establishing the winners before unblinding test data guarantees institutional and scientific integrity as demanded by Global Rules G2 and G3.
- **Alternatives Rejected**:
  - Selecting the champion based on holdout test set performance: Rejected as severe target leakage and multiple hypothesis testing overfitting on the test partition.
  - Manual subjective committee selection: Rejected due to cognitive bias and lack of reproducibility.

---

### D-041: Holdout Test Set One-Time Access Guard and Cryptographic Lock File (TEST_USED.lock)
- **Date**: 2026-10-27
- **Decision**: Restrict test set evaluation strictly to `src/mei/evaluation/test_eval.py::evaluate_final_on_test()`. Upon first execution, write an immutable lock file `artifacts/runs/<run_id>/TEST_USED.lock` recording UTC timestamp, run ID, finalist IDs, test set size ($N=328$), and SHA256 digest of `test_idx.json`. Any subsequent call to evaluate the test set for this run ID MUST immediately raise a `PermissionError`.
- **Reason**: Guarantees adherence to Global Rule G3 (no leakage; test set touched exactly once). Enforces an irreversible audit trail proving the holdout partition was never re-used to tune hyperparameters, adjust decision thresholds, or reorder the leaderboard.
- **Alternatives Rejected**:
  - Unrestricted programmatic test set access: Creates risk of repeated ad-hoc test evaluations.
  - Warning without execution termination: Permits accidental re-evaluations during iterative debugging.

---

### D-046: Selection of Explainability Architecture (SHAP KernelExplainer for Champion KNN and TreeExplainer for Hybrid Tree Analysis)
- **Date**: 2026-10-27
- **Decision**: Implement a two-tiered explainability strategy:
  1. **Champion Model (`knn`)**: Use `shap.KernelExplainer` evaluating the model's output probability $f(x) = P(\text{at\_risk} = 1 \mid x)$ relative to a 25-medoid k-means summary of the 200-sample empirical background distribution (`shap_background_200.parquet`). Bounded with L1 feature selection (`num_features(10)`) to achieve exact efficiency ($\sum \phi_i = f(x) - E[f]$) at $> 3$ samples/second.
  2. **Hybrid Tree Benchmark (`xgboost_hybrid`)**: Use `shap.TreeExplainer` on the top tree-based hybrid model to compute SHAP interaction values for the top 5 features and explicitly measure the partition of total importance carried by state features vs aggregates.
- **Reason**: TreeExplainer cannot explain non-tree models like KNN; KernelExplainer is mathematically model-agnostic and satisfies the Shapley efficiency axiom. Auditing TreeExplainer on the hybrid tree model additionally answers the proposal's requirement to quantify state feature importance share.
- **Alternatives Rejected**:
  - Permutation-only explanations: Lacks additive local fairness axioms; cannot generate exact individual student waterfall charts.
  - Unconstrained KernelExplainer on all 200 background rows: Exponentially slower ($\sim 30\text{ seconds/sample}$), exceeding the 15-minute budget.

---

### D-047: Mathematical Mapping of SHAP Feature Attributions to Additive Integer MEI Points
- **Date**: 2026-10-27
- **Decision**: Define the operational translation from local probability attributions $\phi_j$ to integer MEI points:
  $$\text{points}_j = \text{round}(100 \times \phi_j)$$
  Target score delta is $\Delta \text{MEI} = \text{round}(\text{MEI} - \text{baseline\_MEI})$. To guarantee exact additivity ($\sum_{j=1}^M \text{points}_j = \Delta \text{MEI}$), any residual rounding discrepancy (typically $\pm 1$ pt) is absorbed into the feature with the largest absolute attribution.
- **Reason**: Non-technical stakeholders (academic advisors and department heads) understand integer points (e.g., "$+14$ points from declining study time") far more readily than decimal log-odds. Exact additivity guarantees that when an advisor sums up the points on a student's profile, the sum matches the difference between the student's MEI and the campus average.
- **Alternatives Rejected**:
  - Unscaled raw log-odds: Incomprehensible to educational staff and non-additive on probability.
  - Floating-point decimals: Adds visual clutter without enhancing decision utility.

---

### D-048: Empirical Multi-Fold SHAP Stability Evaluation and Interpretation Protocol
- **Date**: 2026-10-27
- **Decision**: Adopt three rigorous multi-rater consensus metrics across the 15 repeated cross-validation folds:
  1. Mean pairwise Spearman rank correlation ($\rho \ge 0.70$).
  2. Kendall's coefficient of concordance ($W \ge 0.65$).
  3. Top-10 feature Jaccard similarity ($J \ge 0.50$).
  Explicitly mandate in documentation that "high explanation stability does NOT imply causal truth".
- **Reason**: Validates the research proposal's "SHAP consistency" metric. Ensures that local explanations provided to advisors do not fluctuate wildly depending on which cross-validation fold trained the model weights. The stability $\ne$ causality caveat prevents advisors from making invalid counterfactual assumptions.
- **Alternatives Rejected**:
  - Single-fold explanation reporting: Fails to quantify explanation sensitivity to training sample perturbations.

---

### D-049: Plain-English, Non-Judgemental Reason Template Specification
- **Date**: 2026-10-27
- **Decision**: Establish `config/reason_templates.yaml` covering all 100 model input features and 23 state features. Every entry specifies `plain_name`, `risk_increasing`, and `risk_decreasing` templates. Enforce strict phrasing guidelines:
  1. **Non-Judgemental**: Never blame the student (e.g. use "Study hours were lower in recent terms" instead of "Student neglected study").
  2. **Empathetic & Objective**: Focus on observable academic pacing and routine predictability.
  3. **Protective Acknowledgment**: Always pair top risk drivers with top protective factors ($\phi < 0$).
- **Reason**: Educational interventions succeed when framed supportively rather than punitively. Stigmatizing automated labels increase student anxiety and undermine university retention initiatives.
- **Alternatives Rejected**:
  - Automated feature name concatenation (e.g., "`study_time_slope: -0.21`"): Inaccessible to students and non-technical advisors.

---

### D-050: FastAPI Backend Architecture with Dynamic RunStore and Rule G2 Adherence
- **Date**: 2026-10-27
- **Decision**: Architect the analytical backend using FastAPI with a singleton `RunStore` caching data access layer. The active run directory is dynamically resolved on-demand from `artifacts/active_run.txt`. All endpoints serve real pipeline artifacts exclusively (Rule G2); no synthetic placeholders or hard-coded metrics are permitted.
- **Reason**: Decouples API endpoints from specific model run directory names. Enables instant zero-downtime model activation (`POST /api/runs/{id}/activate`) when retrained pipelines are deployed. Guarantees mathematical honesty across all UI views.
- **Alternatives Rejected**:
  - Hardcoding run directory paths inside endpoint routers: Breaches retrainability requirements and forces manual backend modifications upon retraining.
  - Serving mock or synthetic JSON fixtures: Strictly prohibited by Rule G2.

---

### D-051: Two-Stage Upload Validation Workflow and 24-Hour File Lifecycle
- **Date**: 2026-10-27
- **Decision**: Structure student survey uploads as a two-stage transaction:
  1. `POST /api/score/validate`: Evaluates file integrity, performs Levenshtein fuzzy header matching, evaluates missing rate heuristics, and assigns a GREEN/AMBER/RED traffic light with mapping suggestions.
  2. `POST /api/score/run`: Accepts user mapping confirmations/overrides and executes the scoring pipeline.
  Uploaded files are quarantined under `artifacts/uploads/` with random UUID tokens and automatically deleted after a 24-hour expiration window.
- **Reason**: University advisors frequently upload spreadsheets with divergent header naming conventions. The validation preview prevents batch scoring failures, minimizes user frustration, and satisfies strict GDPR/FERPA storage limitation mandates.
- **Alternatives Rejected**:
  - Single-step unvalidated scoring upload: Causes silent misalignments or immediate 500 errors when headers differ.
  - Permanent retention of uploaded student survey spreadsheets: Breaches student data privacy protocols.

---

### D-053: Custom Palette Isolation and CI Enforcement Against AI Clichés
- **Date**: 2026-10-27
- **Decision**: Restrict the frontend design tokens strictly to the palette specified in `PROJECT_BRIEF.md §10`: `paper #F5F2EC`, `surface #FFFFFF` (1px `#DDD8CE` border, 6px border-radius, zero drop shadows), `ink #1A1D21`, `muted #5B6168`, `line #DDD8CE`, and primary action `forest #1F4E46`. Implement an automated pre-commit / CI script (`frontend/scripts/check-forbidden.cjs`) that scans all frontend TypeScript/TSX source files to mathematically prohibit:
  1. Forbidden colors: Purple, indigo, violet, and magenta (HSL hue range $245^\circ\text{–}335^\circ$ with saturation $> 20\%$).
  2. Forbidden marketing buzzwords: `prototype`, `demo`, `beta`, `mock`, `placeholder`, `smart`, `magic`, `powered by ai`.
  3. Emojis, sparkles, and robot icons.
- **Reason**: University analytics tools must convey institutional rigor, editorial clarity, and calm authority. Gradient cards, neon accents, floating shadows, and AI marketing badges degrade academic credibility and distract student advisors.
- **Alternatives Rejected**:
  - Relying on manual code review: Ineffective at preventing accidental Tailwind color leakage.
  - Using default Tailwind CSS colors: Introduces forbidden blue, indigo, and violet shades.

---

### D-054: Radix UI Accessible Primitives and Offline Fontsource Bundling
- **Date**: 2026-10-27
- **Decision**: Construct all interactive shell components (Tooltips, Tabs, Dialogs, Popovers, Drawers) on top of Radix UI unstyled headless primitives. Bundle typography locally via `@fontsource/source-serif-4` (headings, display numerals) and `@fontsource/public-sans` (UI text, table data) within `frontend/dist/assets/`, paired with CSS tabular numeral formatting (`font-feature-settings: 'tnum'`).
- **Reason**: Guarantees full WCAG 2.1 AA accessibility (keyboard focus, screen reader announcements, ARIA attributes) while maintaining complete visual control. Offline font bundling guarantees that the web application functions seamlessly in air-gapped or restricted university intranet environments without external Google Fonts CDN dependencies.
- **Alternatives Rejected**:
  - Heavy styled component libraries (e.g. MUI, AntD): Adds unwanted default styling, bloated bundles, and generic template appearances.
  - External CDN font links (`fonts.googleapis.com`): Fails when university firewalls block external CDN traffic or in offline deployments.

---

### D-056: TanStack Table Leaderboard Benchmark Architecture with Accessible Data Alternatives
- **Date**: 2026-10-27
- **Decision**: Construct the model leaderboard using `@tanstack/react-table` with multi-attribute benchmarking, sorting, searching, overfit status filtering, category filtering, column visibility toggles, CSV export, sticky header, row navigation, row selection checkboxes, and comparison toolbar. Every chart visualization (Score by Model, Generalisation Gap, Metric Heatmap, Time vs Score, Category Distribution) is paired with a 1-sentence caption, sample size $N$, PNG download, and an accessible data alternative (`sr-only` table plus on-demand table view toggle).
- **Reason**: Researchers and staff require transparent inspection of all candidate models without hiding or truncating metrics. Providing accessible text alternatives guarantees compliance with Section 508 / WCAG 2.1 AA and enables non-visual analysis.
- **Alternatives Rejected**:
  - Simplified summary cards without full tabular benchmark: Conceals important trade-offs (e.g. inference time vs precision).
  - Canvas-only charts without accessible table alternatives: Inaccessible to screen readers and violates institutional accessibility standards.

---

### D-057: Interactive Decision Cutoff Simulation (ThresholdSlider) for Academic Triage
- **Date**: 2026-10-27
- **Decision**: Implement the `ThresholdSlider` component on `/leaderboard/:modelId` under the `Errors & Thresholds` tab, allowing academic advisors to interactively simulate varying probability cutoffs ($0.05\text{–}0.95$). Moving the slider recalculates precision, recall, positive flags count, and flag rate in real time, with explicit visual markers for the tuned optimal threshold ($0.26$) and sensitive screening threshold ($0.21$).
- **Reason**: Fixed binary cutoffs are inappropriate for diverse university intervention capacities. When advisory resources are limited, staff need high precision; when capacity is high, staff need high recall. The interactive simulator empowers staff to understand and calibrate the trade-off empirically before deploying flags.
- **Alternatives Rejected**:
  - Hardcoding a single static threshold without exploratory simulation: Limits staff flexibility and conceals the precision-recall trade-off.
  - Allowing unconstrained manual threshold inputs without empirical grid bounds: Can lead to invalid operating points (e.g. 0 flags or 100% flags).

---

### D-058: Deterministic Plain-English Storytelling Architecture for Non-Technical University Stakeholders
- **Date**: 2026-10-28
- **Decision**: Implement deterministic sentence generation functions in `frontend/src/content/sentences.ts` (unit-tested with Vitest) and wire them to live artifact endpoints across Overview (`/overview`), Study Patterns (`/patterns`), and Risk Drivers (`/drivers`). All chart visualizations feature 1-sentence plain-language reading guides, sample size $n$, and accessible tabular data alternatives. Never use hardcoded metrics or ungrounded generative summaries; every narrative is derived directly from active run artifacts (`bands.json`, `states/summary.json`, `explain/global_importance.json`, etc.). If an ablation or test is not statistically significant (e.g. HMM state features yielding neutral incremental F1 gain over rich aggregates), state it plainly and honestly without marketing embellishment.
- **Reason**: University advisors and department heads require clear, actionable, and mathematically honest interpretations of student risk scores without technical jargon, false claims of causality, or algorithmic overpromising. Strict deterministic templating guarantees that all numbers remain 100% faithful to the underlying audited artifacts.
- **Alternatives Rejected**:
  - Generative LLM runtime summaries: Introduce hallucinations, non-deterministic phrasing, latency, and security risks in offline university deployments.
  - Purely technical statistical notation without plain interpretations: Prevents academic advisors and student support staff from acting effectively on early-warning signals.

---

### D-059: Multi-Tier Column Auto-Mapping and Data Quality Traffic Light Auditing
- **Date**: 2026-10-28
- **Decision**: Implement a 4-tier column matching pipeline (exact normalized match, alias dictionary, token cosine similarity with TF-IDF character n-grams, and confidence categorization: High $\ge 85\%$, Check this $50\text{–}84\%$, Not found $< 50\%$) combined with an automated data quality traffic light system (`GREEN`, `AMBER`, `RED`). Auto-accept only high-confidence matches; require explicit user review/confirmation for lower-confidence mappings before scoring. Provide fixable problem summaries (out-of-range Likert bounds clipping, duplicate ID deduplication, median imputation for minor missing data $< 50\%$) and an exportable problem rows report (`problem_rows.csv` / `.xlsx`).
- **Reason**: Student survey datasets from diverse LMS platforms and Google Forms exhibit irregular column headers, non-standard text options, and messy rows. A rigid parser fails on real institutional data, whereas an uninspected fuzzy mapper risks scoring against incorrect columns. The interactive review step empowers university staff to verify mappings with sample values before model execution.
- **Alternatives Rejected**:
  - Rigid schema enforcement: Rejects any file whose headers do not exactly match the internal 28 canonical names, creating severe user friction.
  - Fully uninspected auto-mapping without user review: Silently misassigns ambiguous columns (e.g., swapping Study Time and Review Time), compromising predictive validity.

---

### D-060: Vectorized High-Throughput Cohort Scoring with Longitudinal Trajectories and Waterfall Point Attribution
- **Date**: 2026-10-28
- **Decision**: Optimize the scoring pipeline using vectorized Pandas operations and chunked inference ($5,000$ rows per batch) to achieve sub-second execution on standard cohorts ($N \le 1,000$) and process institutional-scale cohorts ($N = 20,000$) in $25.1$s ($< 60$s SLA limit). Pair every scored student with a 4-period longitudinal engagement trajectory (composite and indicator-level) and a point-based waterfall attribution ($E[MEI] + \sum \Delta_{driver} = MEI$) anchored to the active run's cohort baseline ($24.6$ pts).
- **Reason**: Academic staff cannot act on a bare black-box probability. Showing the student's 4-period trajectory alongside exact additive point contributions (e.g., *Low Practice & Quizzes: +27 pts, Reduced Study Consistency: +21 pts*) provides concrete, actionable diagnostic reasons that academic advisors can reference during supportive student meetings.
- **Alternatives Rejected**:
  - Iterative row-by-row scoring: Taking $> 140$s for $20,000$ rows, exceeding acceptable web response timeouts.
  - Black-box percentage scores without waterfall points: Fails institutional explainability requirements and leaves advisors without actionable intervention targets.

---

### D-061: Trust & Privacy Enforcement with Automated 24-Hour File Purging and Non-Punitive Educational Guardrails
- **Date**: 2026-10-28
- **Decision**: Enforce strict data privacy by processing all uploaded student surveys entirely within the isolated local deployment and implementing an automated cleanup routine (`cleanup_old_uploads`) that deletes all raw files and temporary session artifacts older than 24 hours ($86,400$s). In the student detail inspection drawer, include a permanent, prominent institutional notice: *"The score describes study engagement answers. It is not a prediction about a person's character and should be used to start a supportive conversation."*
- **Reason**: Student educational records are protected by strict privacy frameworks (e.g., FERPA, GDPR). Guaranteeing automated ephemeral storage protects student data privacy. Furthermore, predictive engagement indices risk harmful stigmatization if misinterpreted as fixed character traits; explicit non-punitive framing ensures ethical institutional use.
- **Alternatives Rejected**:
  - Indefinite retention of uploaded survey spreadsheets: Violates educational data minimization principles and creates compliance risks.
  - Purely technical output without ethical guardrails: Enables inappropriate punitive usage (e.g., automatic grade penalties or disciplinary actions).




---

### D-062: Inline Implementation of `assemble_and_save_regression_leaderboard` in `retrain.py`
- **Date**: 2026-10-28
- **Decision**: Implement `assemble_and_save_regression_leaderboard()` inline inside `src/mei/registry/retrain.py` rather than importing it from `mei.models.regression_engine`, which does not expose this function.
- **Reason**: The function was referenced in the orchestrator design but was never added to `regression_engine.py`. Adding it inline avoids a circular dependency risk and keeps the orchestrator self-contained.
- **Alternatives Rejected**:
  - Patching `regression_engine.py`: Would widen the public API of an already complex module and risk breaking existing callers.

---

### D-063: `validate_retrain_dataset` Accepts `Path` Not `DataFrame`
- **Date**: 2026-10-28
- **Decision**: Changed the signature of `validate_retrain_dataset()` from `(df: pd.DataFrame)` to `(dataset_path: Path)` so the function reads and parses the file itself.
- **Reason**: The API endpoint receives a file upload (not a DataFrame). Having the function own file reading keeps the endpoint thin and ensures consistent parsing logic (CSV vs XLSX detection, encoding handling) in one place.
- **Alternatives Rejected**:
  - Pass a pre-parsed DataFrame from the endpoint: Requires the endpoint to know file format specifics and duplicates parsing logic.

---

### D-064: SmallDataGuards Traffic Light Values Are Uppercase Strings
- **Date**: 2026-10-28
- **Decision**: `SmallDataGuards.evaluate()` returns `traffic_light` as an uppercase string: `"RED"`, `"AMBER"`, or `"GREEN"`.
- **Reason**: Uppercase matches the convention used by the scoring validation endpoint (Phase 15) and allows direct comparison without case normalisation in the frontend.
- **Alternatives Rejected**:
  - Lowercase or enum: Would require extra normalisation in both the API layer and tests.

---

### D-065: Test Helper `make_test_survey_df()` Instead of `SyntheticSurveyGenerator`
- **Date**: 2026-10-28
- **Decision**: `tests/test_retraining.py` uses a local `make_test_survey_df()` helper to generate test survey DataFrames rather than importing `SyntheticSurveyGenerator` from `mei.generator`.
- **Reason**: `SyntheticSurveyGenerator` requires `config/schema.yaml` to be present. When the `config/` directory is absent from the working tree (it is gitignored in some configurations), all tests that depend on the generator fail at import time. The local helper has no external dependencies.
- **Alternatives Rejected**:
  - Fixture files on disk: Would need to be committed and kept in sync with schema changes.

---

### D-066: Cooperative Cancellation via `threading.Event` in Orchestrator
- **Date**: 2026-10-28
- **Decision**: `run_full_pipeline()` accepts an optional `cancel_event: threading.Event`. At the start of each pipeline stage, the orchestrator checks `cancel_event.is_set()` and raises `CancellationRequested` if set. The API cancel endpoint sets this event via the `JobManager`.
- **Reason**: Cooperative cancellation avoids abrupt thread termination (which can leave model artifacts in a corrupted state). Each stage completes its current unit of work before honouring the cancel request.
- **Alternatives Rejected**:
  - Daemon threads with `Thread.join(timeout)`: Cannot interrupt long-running Optuna trials cleanly.
  - `multiprocessing.Process.terminate()`: Sends SIGTERM which can corrupt open file handles on Windows.

---

### D-067: Audit Log Written to `artifacts/audit.log` on Activation and Rollback
- **Date**: 2026-10-28
- **Decision**: Every call to `activate_pipeline_run()` appends a timestamped line to `artifacts/audit.log` recording the new run id, previous run id, and the actor (user or system that triggered the change).
- **Reason**: Model governance requires a tamper-evident record of which model was active at which time and who made the change. The append-only log survives run directory deletions and can be reviewed independently.
- **Alternatives Rejected**:
  - Storing activation history only in run metadata JSON: Deleted runs would lose their history.

---

### D-068: `client.ts` and `App.tsx` Restored from Git History for Frontend Build
- **Date**: 2026-10-28
- **Decision**: `frontend/src/api/client.ts` (from commit `e7c2c7b`) and `frontend/src/App.tsx` (from commit `0af5778`) were missing from the working tree. Restored via `git checkout <commit> -- <file>`. Both files are now tracked and committed as part of the Phase 16 frontend section.
- **Reason**: These files were present in earlier phase commits but were accidentally excluded from the HEAD working tree (likely during a rebase or selective checkout). The frontend build fails without them.

