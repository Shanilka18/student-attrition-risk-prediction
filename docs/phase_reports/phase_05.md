# Phase 5 Report — Engagement States, Patterns, and Trajectory Modeling

## 1. What Was Done
- **Longitudinal Trajectory Clustering (`src/mei/states/clustering.py`)**:
  - Extracted 7-dimensional composite trajectory representations ($E_1..E_4$, slope, volatility, level) on the training partition ($N = 1,528$).
  - Evaluated KMeans, Gaussian Mixture Models (diagonal and full covariance), and Agglomerative Ward clustering across $k \in [2, 6]$.
  - Executed 30 bootstrap resamples with replacement to compute Adjusted Rand Index (ARI) stability scores.
  - Applied the formal selection rule: selected the model with the highest bootstrap stability among candidates satisfying silhouette $\ge 0.15$ (ties broken by smaller $k$).
  - Winner: **KMeans with $k=3$**, achieving a Silhouette score of **0.3744** and Bootstrap Stability of **0.9627**.
  - Formulated semantic cluster profiles: Cluster 0 (*Low Improving*), Cluster 1 (*High Consistent*), and Cluster 2 (*Moderate Declining*).
- **Gaussian Hidden Markov Modeling (`src/mei/states/hmm.py`)**:
  - Implemented `GaussianHMM` with diagonal covariance using `hmmlearn` over sequences of length $T=4$ across the 7 standardized behavioral indicators.
  - Evaluated candidate state dimensions $K \in \{2, 3, 4\}$ across 10 random restarts each.
  - Selected $K=4$ based on Bayesian Information Criterion (BIC: $73,747.09$ vs $87,285.47$ for $K=3$ and $95,814.21$ for $K=2$; $\Delta \text{BIC} = 13,538.4 \gg 10$) and held-out validation log-likelihood ($-7,918.97$ vs $-9,329.23$ for $K=3$).
  - Enforced deterministic state semantic ordering: permuted state indices in ascending order of standardized emission composite engagement so State 1 (index 0) is guaranteed to be *Low*, State 2 is *Medium-Low*, State 3 is *Medium-High*, and State 4 is *High*.
  - Computed transition probability matrix $A_{ij}$, initial state probabilities $\pi$, emission parameters $(\mu_k, \Sigma_k)$, and expected dwell times $E[T_k] = 1 / (1 - A_{kk})$.
  - Implemented Viterbi dynamic programming decoding, posterior probability computation, and sequence log-likelihood calculation.
- **Deterministic Trajectory Patterns & Longitudinal Levels (`src/mei/states/patterns.py`)**:
  - Formulated transparent, deterministic transition rules mapping 4-period Viterbi state paths into human-interpretable advisor phenotypes (*Consistent*, *Declining*, *Improving*, *Sporadic*).
  - Categorized overall semester engagement levels into *High* ($\bar{E} \ge 65$), *Medium* ($35 \le \bar{E} < 65$), and *Low* ($\bar{E} < 35$).
- **Scikit-Learn Compatible Transformer (`src/mei/states/transformers.py`)**:
  - Built `StateFeatureTransformer` encapsulating median imputation, composite trajectory extraction, clustering distance/one-hot encoding, Viterbi path decoding, transition counts, state dwell fractions, posterior probabilities, sequence log-likelihood, and pattern one-hot flags.
  - Strictly enforced Global Rule G3: fitted exclusively inside training partitions and cross-validation training folds.
  - Saved transformed feature partitions:
    - `train_states_only.parquet`, `val_states_only.parquet`, `test_states_only.parquet` (26 features)
    - `train_agg_states.parquet`, `val_agg_states.parquet`, `test_agg_states.parquet` (126 features)
- **Unsupervised Ground Truth Recovery Analysis (`src/mei/states/evaluation.py`)**:
  - Evaluated unsupervised recovery fidelity against latent generative archetypes on the synthetic training cohort (explicitly labeled *"valid only for generated benchmark data"*).
  - Measured Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI): Archetype $\times$ Cluster achieved $\text{ARI} = 0.2475, \text{NMI} = 0.3203$; Archetype $\times$ Pattern achieved $\text{ARI} = 0.1262, \text{NMI} = 0.2299$.
- **15-Fold Repeated Stratified CV Predictive Ablation (`src/mei/states/evaluation.py`)**:
  - Benchmarked `states_only` (26 features) vs `agg` (100 features) vs `agg_states` (126 features) across 15 fixed folds (Repeated Stratified 5-Fold $\times$ 3 repeats) on Train + Validation ($N = 1,856$).
  - Evaluated fixed Logistic Regression and LightGBM models across AUC-ROC, F1 At-Risk, Matthews Correlation Coefficient (MCC), and Balanced Accuracy with 95% confidence intervals.
  - Conducted paired Wilcoxon signed-rank tests across the 15 folds to test statistical significance of incremental gains ($\Delta = \text{agg\_states} - \text{agg}$).
  - Documented honest scientific conclusions: `states_only` yields powerful standalone performance (LR AUC $0.8691$, LightGBM AUC $0.8678$); incremental gains over `agg` are negligible ($\Delta \text{AUC} = -0.0025$ for LR, $-0.0008$ for LightGBM) because continuous OLS slopes and volatility in `agg` already capture longitudinal dynamics.
- **Publication Figures in Brand Palette (`src/mei/states/figures.py`)**:
  - Rendered 7 publication figures (14 files total in 300 DPI PNG and SVG) in `artifacts/runs/<run_id>/figures/`, `artifacts/figures/`, and `docs/figures/`, cataloged in `docs/figures/INDEX.md`:
    - `state_01_emission_profiles` (FIG-STATE-01)
    - `state_02_transition_matrix` (FIG-STATE-02)
    - `state_03_trajectories_by_pattern` (FIG-STATE-03)
    - `state_04_pattern_distribution` (FIG-STATE-04)
    - `state_05_at_risk_by_pattern` (FIG-STATE-05)
    - `state_06_cluster_silhouette` (FIG-STATE-06)
    - `state_07_k_selection_curves` (FIG-STATE-07)
- **CLI Implementation (`src/mei/cli.py`)**:
  - Implemented `python -m mei.cli states` executing model selection, feature transformation, ground truth recovery, 15-fold CV ablation, figure rendering, and summary serialization to `artifacts/runs/<run_id>/states/summary.json`.
  - Added robust caching (`clustering_res.joblib`, `hmm_res.joblib`, `ablation_res.joblib`) with a `--force` override option and UTF-8 console output support.
- **Documentation & Viva Defense Guide**:
  - Authored comprehensive technical documentation in `docs/06_engagement_states.md` adhering to Global Rule G8.
  - Logged architectural and methodological decisions D-021 through D-024 in `docs/DECISIONS.md`.
- **Testing & Verification**:
  - Implemented 18 comprehensive tests in `tests/test_states.py` covering clustering selection, bootstrap stability, HMM selection, state permutation invariance, transformer data-leakage guards, pattern rules, ground truth recovery, CV ablation, and publication figures.
  - Reached **93% test coverage** on `src/mei/states/` (854 statements, 63 misses). Full test suite passes across the entire workspace.
- **Git Commit Withheld**: Strictly observed user constraint not to commit to git.

---

## 2. Files Created and Modified
- `src/mei/states/clustering.py` *(Created)*: Longitudinal composite trajectory extraction, KMeans/GMM/Ward evaluations, 30-bootstrap stability ARI, and cluster profiling.
- `src/mei/states/hmm.py` *(Created)*: Sequence preparation, multi-restart GaussianHMM fitting, BIC evaluation, validation scoring, deterministic composite state sorting, dwell times, and Viterbi/posterior decoding.
- `src/mei/states/patterns.py` *(Created)*: Deterministic rule classifier for trajectory phenotypes (*Consistent*, *Declining*, *Improving*, *Sporadic*) and semester engagement levels (*High*, *Medium*, *Low*).
- `src/mei/states/transformers.py` *(Created)*: Scikit-learn compatible `StateFeatureTransformer` strictly fitted inside training partitions.
- `src/mei/states/evaluation.py` *(Created)*: Ground truth archetype recovery metrics and 15-fold CV ablation benchmark with paired Wilcoxon signed-rank tests.
- `src/mei/states/figures.py` *(Created)*: Brand-palette 300 DPI PNG and SVG generators for FIG-STATE-01 through FIG-STATE-07.
- `src/mei/states/__init__.py` *(Created)*: Public package exports.
- `config/project.yaml` *(Modified)*: Added `engagement_states:` section configuring clustering algorithms, $k$ range, $K$ range, restarts, bootstrap samples, and pattern level thresholds.
- `config/model_zoo.yaml` *(Modified)*: Activated `agg_states` feature set and registered `states_only` feature specification.
- `src/mei/cli.py` *(Modified)*: Implemented full `states` CLI command with Rich diagnostic tables and stage checkpoint caching.
- `docs/figures/INDEX.md` *(Modified)*: Cataloged the 7 newly generated Phase 5 publication figures.
- `docs/DECISIONS.md` *(Modified)*: Logged architectural decisions D-021, D-022, D-023, and D-024.
- `docs/06_engagement_states.md` *(Created)*: Technical documentation conforming to Global Rule G8.
- `docs/phase_reports/phase_05.md` *(Created)*: Phase 5 completion report.
- `tests/test_states.py` *(Created)*: 18 unit and integration tests covering clustering, HMMs, transformers, pattern rules, and figures.
- `tests/test_smoke.py` *(Modified)*: Added `test_cli_states_command` and updated remaining stubs.

---

## 3. Key Numbers Read Directly from Output Files
*Source: `artifacts/runs/run_20261026_124514_1407b3/states/summary.json` (evaluated on training cohort $N = 1,528$, val $N = 328$, test $N = 328$)*:

- **Clustering Model Selection (Training Partition, $N = 1,528$)**:
  - Selected Algorithm: `kmeans` ($k = 3$)
  - Silhouette Score: `0.3744` (exceeds $\ge 0.15$ threshold)
  - Bootstrap Stability (Mean ARI across 30 resamples): `0.9627` (Highest among all candidates)
  - Calinski-Harabasz Score: `941.8`
  - Davies-Bouldin Index: `1.055`
  - Cluster Phenotypes:
    - Cluster 0: *Low Improving* (114 students)
    - Cluster 1: *High Consistent* (831 students)
    - Cluster 2: *Moderate Declining* (583 students)
- **Gaussian HMM Model Selection ($K \in [2, 4]$)**:
  - Selected Latent States: $K = 4$
  - Model Convergence: `YES` (all 10 restarts converged)
  - Training Log-Likelihood: `-36,564.05`
  - Bayesian Information Criterion (BIC): `73,747.09` ($\Delta \text{BIC} = 13,538.4 \gg 10.0$ over $K=3$ BIC $87,285.47$)
  - Validation Log-Likelihood: `-7,918.97` (superior to $K=3$ val LL $-9,329.23$)
  - Expected State Dwell Times:
    - State 1 (*Low*): `3.50` periods ($A_{11} = 0.715$)
    - State 2 (*Medium-Low*): `3.62` periods ($A_{22} = 0.724$)
    - State 3 (*Medium-High*): `1.76` periods ($A_{33} = 0.433$)
    - State 4 (*High*): `1.92` periods ($A_{44} = 0.480$)
- **Pattern & Level Distributions (Training Cohort, $N = 1,528$)**:
  - Trajectory Patterns: *Consistent* `831` (54.4%), *Sporadic* `358` (23.4%), *Declining* `225` (14.7%), *Improving* `114` (7.5%)
  - Engagement Levels: *Medium* `953` (62.4%), *High* `298` (19.5%), *Low* `277` (18.1%)
- **Unsupervised Ground Truth Recovery (Valid Only for Generated Data)**:
  - Adjusted Rand Index (Archetype $\times$ Cluster): `0.2475`
  - Normalized Mutual Info (Archetype $\times$ Cluster): `0.3203`
  - Adjusted Rand Index (Archetype $\times$ Pattern): `0.1262`
  - Normalized Mutual Info (Archetype $\times$ Pattern): `0.2299`
- **15-Fold Repeated CV Feature Set Ablation Benchmark**:
  - **Logistic Regression**:
    - `states_only`: AUC `0.8691 ± 0.0176`, F1 `0.6446 ± 0.0385`, MCC `0.5379 ± 0.0475`, Bal Acc `0.7566 ± 0.0257`
    - `agg`: AUC `0.8711 ± 0.0183`, F1 `0.6423 ± 0.0431`, MCC `0.5374 ± 0.0547`, Bal Acc `0.7539 ± 0.0276`
    - `agg_states`: AUC `0.8686 ± 0.0182`, F1 `0.6438 ± 0.0348`, MCC `0.5360 ± 0.0448`, Bal Acc `0.7558 ± 0.0222`
  - **LightGBM**:
    - `states_only`: AUC `0.8678 ± 0.0200`, F1 `0.6235 ± 0.0380`, MCC `0.5146 ± 0.0440`, Bal Acc `0.7423 ± 0.0249`
    - `agg`: AUC `0.8722 ± 0.0197`, F1 `0.6413 ± 0.0424`, MCC `0.5346 ± 0.0548`, Bal Acc `0.7537 ± 0.0271`
    - `agg_states`: AUC `0.8713 ± 0.0201`, F1 `0.6317 ± 0.0362`, MCC `0.5198 ± 0.0480`, Bal Acc `0.7485 ± 0.0230`
- **Paired Wilcoxon Signed-Rank Tests (agg vs agg_states)**:
  - Logistic Regression AUC: Mean Diff ` -0.0025`, $W = 32.0$, $p = 0.1205$ (*Not Significant*)
  - Logistic Regression F1: Mean Diff ` +0.0015`, $W = 47.0$, $p = 0.4887$ (*Not Significant*)
  - LightGBM AUC: Mean Diff ` -0.0008`, $W = 57.0$, $p = 0.8904$ (*Not Significant*)
  - LightGBM F1: Mean Diff ` -0.0096`, $W = 30.0$, $p = 0.1578$ (*Not Significant*)
  - LightGBM MCC: Mean Diff ` -0.0149`, $W = 21.0$, $p = 0.0480$ (*Significant drop due to dimensionality*)
- **Code Coverage**:
  - `src/mei/states/`: **93%** coverage (791 / 854 statements covered, exceeding $\ge 85\%$ requirement)
- **Pytest Suite Status**:
  - Full suite passes across all workspace tests.

---

## 4. Checks Run with PASS/FAIL Status
| Check | Command / Target | Status | Details |
|---|---|---|---|
| Rule G3 Zero Data Leakage | `test_state_transformer_no_leakage` | **PASS** | StateFeatureTransformer fits strictly on training splits; val/test distributions untouched |
| Deterministic State Ordering | `test_hmm_state_ordering_deterministic` | **PASS** | States deterministically sorted in ascending order of composite emissions (State 1 = Low) |
| Clustering Selection Rule | `test_clustering_selection_rule` | **PASS** | KMeans $k=3$ selected by bootstrap stability ($0.9627$) meeting silhouette $\ge 0.15$ ($0.3744$) |
| Pattern Rule Determinism | `test_pattern_classification_rules` | **PASS** | Correctly maps synthetic Viterbi paths to Consistent, Declining, Improving, Sporadic |
| Ground Truth Recovery | `test_ground_truth_recovery_evaluation` | **PASS** | Recovery metrics computed correctly and labeled "valid only for generated data" |
| 15-Fold Repeated CV Ablation | `test_ablation_cv_execution` | **PASS** | Evaluates agg, agg_states, states_only across 15 folds with paired Wilcoxon signed-rank tests |
| Publication Figures Generation | `test_publication_figures_generation` | **PASS** | All 7 figures rendered in 300 DPI PNG and SVG in brand palette |
| CLI States Command | `test_cli_states_command` | **PASS** | Full CLI pipeline runs with Rich output, saving models and summary.json |
| Code Formatting & Linting | `ruff check src tests` & `ruff format --check` | **PASS** | Zero linting errors; 40+ files cleanly formatted |
| Full Test Suite Regression | `pytest tests/` | **PASS** | All 70+ tests pass across workspace |

---

## 5. Risks or Decisions Logged
- **D-021: Clustering Model Selection via Bootstrap Stability and Silhouette Thresholding**: Evaluated KMeans, GMM (full/diag), Ward for $k \in [2, 6]$. KMeans ($k=3$) achieved highest bootstrap stability ($0.9627$) and silhouette ($0.3744$).
- **D-022: Gaussian HMM Model Selection and Deterministic State Ordering**: Evaluated $K \in \{2, 3, 4\}$. $K=4$ yielded the lowest BIC ($73,747.09$ vs $87,285.47$ for $K=3$, $\Delta \text{BIC} \gg 10$). Reordering states by emission composite engagement guarantees deterministic semantics (State 1 = Low).
- **D-023: Deterministic Rule-Based Trajectory Pattern and Level Classification**: Formulated transparent transition rules mapping Viterbi sequences to Consistent, Declining, Improving, Sporadic and Low/Medium/High to provide human-interpretable advisor categorizations.
- **D-024: Honest Reporting of Latent State Incremental Predictive Value**: Documented that while `states_only` provides strong standalone classification signal ($\text{AUC} \approx 0.869$), incremental gains over `agg` are negligible because dynamic slope and volatility in `agg` already capture continuous longitudinal trajectories. Disclosed short sequence length ($T=4$) as an explicit methodological boundary condition.
