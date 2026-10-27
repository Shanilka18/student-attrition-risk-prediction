# Research Project Decision Log

**Project**: Multi-Factor Student Attrition Risk Prediction Using Academic, Engagement, Psychosocial, Environmental Factors  
**Component**: Academic Intelligence Framework  
**Student**: V. S. H. Gunasekara (IT23164680)  
**Supervisors**: Ms. Chathurika Koswatte, Ms. Anuththara Ekanayake  
**Cluster**: Centre of Excellence for AI (CoEAI), SLIIT  

---

## Log Format
Every decision made across all project phases that resolves ambiguity, chooses between technical trade-offs, or aligns with proposal requirements must be recorded here.

Each entry documents:
1. **Decision ID & Date**
2. **Context & Problem Statement**
3. **Alternatives Considered**
4. **Decision Made & Rationale**
5. **Supervisor / Proposal Alignment**
6. **Implications on Downstream Phases**

---

### [DEC-001] 2026-10-17: Baseline vs Sequence Model Comparison Architecture
- **Context**: The proposal's core research question is: *Does sequence-aware modelling (LSTM/GRU) beat engineered trend features + Gradient Boosting?*
- **Alternatives Considered**:
  - *Option A*: Compare LSTM against standard static snapshot models (e.g., Logistic Regression on final GPA).
  - *Option B*: Compare LSTM/GRU against Gradient Boosting (XGBoost/LightGBM) trained on hand-engineered temporal features (slope, rolling averages, variance), with static snapshot models as secondary references.
- **Decision**: Option B. To establish a rigorous, defensible scientific contribution, we must compare the sequence model against the strongest competitive baseline: Gradient Boosting trained on well-engineered longitudinal summary features (GPA slope, rolling GPA average, attendance variance, assignment submission delay trends).
- **Proposal Alignment**: Matches Proposal Section 1.3, Section 3.3, and Appendix 1.2.
- **Implications**: The feature engineering pipeline must derive genuine longitudinal features (slopes, rolling metrics) without lookahead bias.

### [DEC-002] 2026-10-17: Strict Grouped Student Splitting & Leakage Defense
- **Context**: Student records span multiple semesters (longitudinal panel data).
- **Alternatives Considered**:
  - *Option A*: Standard random k-fold / train-test split across observation rows.
  - *Option B*: Grouped stratified splitting by `student_id` / `participant_id`.
- **Decision**: Option B. Splitting by row leaks the same student's temporal signature into both train and test partitions, inflating AUC and invalidating generalization. Grouped splitting by student ID is non-negotiable. Scalers, imputers, and encoders must be fit on training folds only.
- **Proposal Alignment**: Adheres to Non-Negotiable Rule 2 and Section 3.5.
- **Implications**: Custom cross-validation strategy (`GroupKFold` or `StratifiedGroupKFold`) strictly enforced in all data splitters and pipeline transformers.

### [DEC-003] 2026-10-17: Synthetic Data Labeling and Realism Constraints
- **Context**: Institutional data may be restricted or delayed pending IRB clearance (Section 3.2, Table 3). Synthetic cohort data is utilized for pipeline validation.
- **Alternatives Considered**:
  - *Option A*: Simple independent Gaussian perturbations per semester.
  - *Option B*: Realistic autoregressive longitudinal simulation reflecting Sri Lankan higher education attrition distributions (correlated trajectories, realistic noise, semester dropout events), with explicit "SYNTHETIC" provenance tags.
- **Decision**: Option B. All synthetic outputs will be labeled `SYNTHETIC` across metadata, file naming, docstrings, and UI displays. The data generator will include non-linear noise, realistic class imbalance (15-25% at-risk base rate), and cross-semester autocorrelation. If validation AUC exceeds 0.97, the generator will be penalized and calibrated for realistic difficulty.
- **Proposal Alignment**: Directly adheres to Proposal Section 3.2, TAF Section 9, and Non-Negotiable Rule 3.
- **Implications**: Generators must output validation metrics and statistical distribution summaries comparing synthetic cohorts against published Sri Lankan benchmark ranges.

### [DEC-004] 2026-10-17: Cross-Platform Build System and Frontend Technology Stack
- **Context**: Development and viva demonstrations may take place across diverse platforms (Windows PowerShell/Command Prompt, macOS, Linux/CI). Standard Windows installations do not provide GNU `make` by default.
- **Alternatives Considered**:
  - *Option A*: Enforce POSIX GNU Make only, requiring users to install MinGW or WSL.
  - *Option B*: Provide a Python-native `run.py` CLI, complemented by a standard `Makefile` for POSIX/CI, and a lightweight `make.bat` wrapper for Windows CMD and PowerShell. For the frontend, utilize React 18 with Vite, Tailwind CSS, and Recharts to avoid React 19 peer-dependency clashes.
- **Decision**: Option B. `run.py` serves as the authoritative orchestrator for all pipeline stages (`setup`, `test`, `data`, `preprocess`, `train`, `evaluate`, `explain`, `serve`, `frontend`, `all`). Windows users can type `.\make.bat <target>` or `python run.py <target>`, while POSIX environments execute `make <target>`.
- **Proposal Alignment**: Adheres to Non-Negotiable Rule 1 ("one-command run for every stage") and Non-Functional Requirement NFR1 (rapid local invocation).
- **Implications**: All pipeline stages must hook into `run.py` targets as they are implemented.

### [DEC-005] 2026-10-17: Latent-Process Synthetic Generator Design & Calibration
- **Context**: Real student data is ethical PII subject to pending IRB clearance. A synthetic cohort is required to build and validate the machine learning pipeline.
- **Alternatives Considered**:
  - *Option A*: Independent Gaussian sampling per feature followed by logistic classification labeling.
  - *Option B*: A continuous structural latent-variable process modeling student cognitive/behavioral archetypes, life event shocks, AR(1) semester autocorrelation, and discrete-time dynamic hazard attrition.
- **Decision**: Option B. Seven latent trajectory archetypes are modeled with semester-to-semester momentum ($\alpha = 0.65$) and curriculum load effects. Crucially, non-separable conditions are introduced (resilient low-GPA persisters and sudden-drop high-GPA attriters) to avoid trivially separable toy distributions. Attrition defaults to 23.6% (calibrated within the required 15-25% range), with a high-attrition preset (~40%) reflecting SLIATE regional center extremes cited in proposal reference [17].
- **Proposal Alignment**: Directly implements Proposal Section 3.2 (p. 15), TAF Section 9 (p. 4), and Non-Negotiable Rule 3.
- **Implications**: Precludes data leakage; generates realistic missingness (MCAR and MAR); provides an uncompromised testing ground for sequence models.

### [DEC-006] 2026-10-17: Definition of Sliding Prefix Prediction Task (Next 2 Semesters)
- **Context**: An operational early-warning system must evaluate students at varying points in their academic journey, not merely after all semesters are complete.
- **Alternatives Considered**:
  - *Option A*: Single-point prediction at end of Year 1 predicting ultimate graduation/dropout.
  - *Option B*: Sliding prefix windows $t \in [2, 3, 4, 5, 6]$ predicting whether attrition occurs within the subsequent 2 semesters ($t+1$ or $t+2$).
- **Decision**: Option B. A 2-semester forward window provides university counselors with an actionable 6-to-12-month intervention runway. Prefix sampling expands the cohort into 22,071 multi-horizon evaluation samples, while strictly retaining student-level grouping (`student_id`) to ensure zero cross-semester leakage during cross-validation.
- **Proposal Alignment**: Fulfills Specific Objectives 2.2 and User Requirements FR1/FR2.
- **Implications**: Sequence models and tabular baseline pipelines must process variable-length prefixes ($t \in [2 \dots 6]$) without lookahead bias.

### [DEC-007] 2026-10-18: Student-Grouped Stratified and Temporal Splitting Protocol
- **Context**: Evaluating models on longitudinal panel data requires guaranteed independence between training, validation, and test sets.
- **Alternatives Considered**:
  - *Option A*: Standard random shuffle splitting across observation rows.
  - *Option B*: Student-level grouped split (70/15/15) stratified on student attrition outcome, accompanied by 5-fold grouped CV and an out-of-time temporal cohort split (train on 2021-2023 cohorts, test on 2024 cohort).
- **Decision**: Option B. Row-level splitting is strictly prohibited to avoid memorizing individual student baselines across semesters. All partitions are partitioned by `student_id` with zero overlap verified mathematically via automated set-intersection assertions.
- **Proposal Alignment**: Enforces Non-Negotiable Rule 2 and Section 3.5.
- **Implications**: `train_students.json`, `val_students.json`, `test_students.json`, `cv_5folds.json`, and temporal split files are frozen on disk in `/data/splits/`.

### [DEC-008] 2026-10-18: Dual-Paradigm Feature Alignment and Static Snapshot Comparator
- **Context**: To cleanly answer the core research question (LSTM/GRU vs Gradient Boosting + engineered trend features), both model paradigms must receive identical input information horizons without lookahead bias.
- **Alternatives Considered**:
  - *Option A*: Generate features ad-hoc inside each model's training script.
  - *Option B*: Centrally construct aligned representations from identical clean prefix windows: 3D sequence tensors for PyTorch, a 22-dimensional longitudinal trend feature matrix for Gradient Boosting, and a static point-in-time snapshot set (`last_gpa`, `last_attendance_pct`) for the legacy early-warning comparator.
- **Decision**: Option B. Aligned sample IDs ensure paired comparisons. A Logistic Regression model trained on static snapshot features establishes the proposal's baseline, achieving AUC of ~0.84 and Recall of ~76%, while demonstrating low precision (~18%) and motivating longitudinal trend and sequence modeling.
- **Proposal Alignment**: Matches Proposal Section 1.2, 1.3, and 3.3.
- **Implications**: Downstream training modules in Phase 4 and 5 ingest pre-aligned representations from `/data/processed/`.

### [DEC-009] 2026-10-18: Unified BaseAcademicModel Interface for the Model Zoo
- **Context**: Phase 4 requires 27+ models across 6 categories (linear, tree/bagging, boosting, MLP, sequence, static-snapshot) that must all be trainable, evaluatable, and explainable by a single training engine in Phase 5 without category-specific branching logic.
- **Alternatives Considered**:
  - *Option A*: Let each model type define its own ad-hoc fit/predict interface; wire category-specific adapters in the training engine.
  - *Option B*: Abstract base class `BaseAcademicModel` with four mandatory methods (`fit`, `predict_proba`, `save`, `load`) and a `ModelRegistry` decorator pattern so every model is discovered, built, and invoked identically regardless of whether it is sklearn, PyTorch, XGBoost, or a rule-based threshold.
- **Decision**: Option B. The `@register` decorator and `ModelRegistry.build(key)` factory make the entire zoo swappable behind a single interface. `_SklearnMixin` provides DRY sklearn integration; `BaseSequenceModel` handles PyTorch training loops, AMP, early stopping, and tensor unpacking centrally. The training engine (Phase 5) and Optuna tuner (Phase 6) will call only `model.fit()` and `model.predict_proba()` regardless of model type.
- **Proposal Alignment**: Enables the clean A/B comparison (LSTM/GRU vs Gradient Boosting) without special-casing in evaluation code. Aligns with Non-Negotiable Rule 1 (one-command run) and Rule 6 (clean code).
- **Implications**: Every new model added in future phases must register via `@register` and implement the four abstract methods. Category-specific logic is contained within each model class.

### [DEC-010] 2026-10-18: Inclusion of TCN (Temporal Convolutional Network) in Category E
- **Context**: Category E must include a broad set of sequence architectures to benchmark against LSTM/GRU. The question is whether to include a TCN, which is a non-recurrent temporal model that achieves state-of-the-art on many sequence tasks.
- **Alternatives Considered**:
  - *Option A*: Skip TCN; focus only on RNN-family models (RNN, LSTM, GRU, BiLSTM, BiGRU) and attention variants.
  - *Option B*: Include TCN as a reference point. TCNs use dilated causal convolutions with residual connections, operate in parallel (unlike sequential RNNs), and provide a strong baseline that sometimes outperforms LSTMs on short sequences.
- **Decision**: Option B. For 6-semester sequences, TCNs are competitive with LSTMs. Including TCN strengthens the viva argument: if TCN beats LSTM, the temporal ordering matters but the recurrence mechanism adds no benefit for this short horizon. If LSTM beats TCN, memory gating provides value beyond local convolution. Either outcome is scientifically interesting. Causal padding ensures no future leakage within the convolutional receptive field.
- **Proposal Alignment**: Strengthens the breadth of the model comparison (Proposal Section 3.3). Viva examiner may ask about non-LSTM temporal alternatives; TCN is the answer.
- **Implications**: TCN `_TemporalBlock` uses causal (not symmetric) padding. The `_TCN.forward()` extracts the last real timestep using lengths to avoid using padded outputs.

### [DEC-011] 2026-10-18: Attention Mechanism Variants for LSTM and GRU (Category E)
- **Context**: Standard LSTM/GRU uses only the last hidden state as the sequence summary. For student trajectories, earlier semesters (e.g., a sudden drop in Semester 2) may be highly predictive even when later semesters partially recover.
- **Alternatives Considered**:
  - *Option A*: Use only last-hidden-state readout for all LSTM/GRU models.
  - *Option B*: Add mask-aware soft attention variants (`LSTM+Attention`, `GRU+Attention`) that compute a weighted sum over all timestep outputs. The attention weights are learned, allowing the model to focus on the most diagnostically significant semester, regardless of order.
- **Decision**: Option B. Soft attention is a well-established enhancement for clinical/educational sequence data and directly tests whether temporal saliency (which semester matters most) is learnable from the data. The attention layer is mask-aware: padded positions are filled with `-inf` before softmax so they receive weight zero. SHAP values in Phase 7 can augment attention weights to give a dual explanation (which semester and which feature within that semester).
- **Implications**: `LSTMAttentionModel` and `GRUAttentionModel` in Category E use the `use_attention=True` path in `_RNNBase`. The attention linear layer (`nn.Linear(out_dim, 1)`) is trained end-to-end.

### [DEC-012] 2026-10-18: Grouped Stratified CV for Optuna Tuning (Strict Training-Fold Enclosure)
- **Context**: Hyperparameter tuning requires evaluating model configurations across multiple candidate settings. Using the validation partition to guide Optuna's Bayesian search leaks validation feedback into hyperparameter selection, invalidating the validation set as an unbiased estimator of generalization.
- **Alternatives Considered**:
  - *Option A*: Tune hyperparameters directly on the 15% validation split.
  - *Option B*: Generate grouped stratified cross-validation folds strictly partitioned from the 70% training cohort (`train_students`). Optuna maximizes the mean cross-validated AUROC across these training folds. The 15% validation partition is touched only afterwards for early stopping, post-hoc calibration, and threshold setting.
- **Decision**: Option B. To satisfy strict zero-leakage standards, Optuna never sees the validation partition during search trials. The `TrainingDataLoader.get_train_cv_folds()` method partitions only `train_students`, guaranteeing zero student overlap between CV train and CV val folds.
- **Proposal Alignment**: Complies with Non-Negotiable Rule 2 (no data leakage) and Proposal Section 3.5.
- **Implications**: Preserves the validation partition as a genuine out-of-fold generalization testbed.

### [DEC-013] 2026-10-18: Recall-Constrained Asymmetric Decision Thresholding
- **Context**: Standard classification frameworks adopt an arbitrary 0.5 decision threshold assuming symmetric misclassification penalties. In higher education, missing a student who subsequently drops out (false negative) is catastrophic, whereas flagging a stable student (false positive) results in a harmless counseling touchpoint.
- **Alternatives Considered**:
  - *Option A*: Default fixed 0.5 threshold across all models.
  - *Option B*: Standard unconstrained $F_1$ maximization.
  - *Option C*: Recall-constrained threshold optimization: find the threshold maximizing $F_1$ subject to $\text{Recall} \ge 0.80$.
- **Decision**: Option C. A fixed 0.5 threshold results in unacceptably low recall (~5-15%) under 1:4 class imbalance. The recall-constrained rule ensures that university counseling services capture at least 80% of struggling students, tuning the decision cutoff on validation probabilities to maximize precision within that non-negotiable safety constraint.
- **Proposal Alignment**: Fulfills User Requirements FR1/FR2 and addresses real-world counselor deployment constraints.
- **Implications**: Both raw and calibrated optimal cutoffs are stored in `threshold_config.json` for API serving in Phase 8.

### [DEC-014] 2026-10-18: Post-Hoc Probability Calibration and Multi-Seed Stability Reporting
- **Context**: Deep neural networks and deep gradient boosted trees produce uncalibrated probabilities that cannot be interpreted as genuine attrition odds. Furthermore, reporting single-seed numbers leaves research findings vulnerable to seed-cherrypicking criticism during viva defense.
- **Alternatives Considered**:
  - *Option A*: Report uncalibrated probabilities and single-run metrics.
  - *Option B*: Fit post-hoc calibration (Platt scaling / Isotonic regression) strictly on validation predictions, verify via Brier score reduction, and evaluate final configurations across 3 independent random seeds (`[42, 123, 999]`), reporting metric mean and standard deviation ($\mu \pm \sigma$).
- **Decision**: Option B. Probability calibration guarantees that the Academic Risk Index (0-100) represents reliable empirical risk, while multi-seed stability confirms that any observed performance differences between LSTM/GRU and Gradient Boosting are statistically significant rather than stochastic artifacts.
- **Proposal Alignment**: Directly adheres to Non-Negotiable Rule 1 (reproducibility) and Non-Negotiable Rule 6 (viva defensibility).
- **Implications**: Calibrators are persisted as `calibrator.joblib` and stability statistics are stored in `seed_metrics.json`.

### [DEC-015] 2026-10-19: Cryptographic One-Touch Test Set Provenance and Student-Grouped Bootstrap Verification
- **Context**: In academic research, repeated ad-hoc probing of the held-out test partition produces subtle information snooping and overoptimistic claims of generalization. Furthermore, calculating confidence intervals by treating multi-semester student observations as independent samples introduces severe pseudo-replication bias due to intra-student autocorrelation.
- **Alternatives Considered**:
  - *Option A*: Standard row-level bootstrap and unmonitored test-set evaluation scripts.
  - *Option B*: Enforce cryptographic SHA-256 fingerprint logging of test targets, student IDs, and sample IDs in `test_evaluation_provenance.json` with evaluation counters, alongside student-grouped bootstrap confidence intervals (1,000 resamples) and paired hypothesis testing (paired bootstrap on $\Delta \text{AUC}$, McNemar's test with continuity correction).
- **Decision**: Option B. Cryptographic provenance provides incontrovertible proof that the test set was evaluated once under frozen conditions. Student-grouped bootstrap resampling samples student entities rather than individual semester rows, ensuring that intra-student temporal dependencies are correctly preserved when estimating standard errors and 95% confidence intervals.
- **Proposal Alignment**: Directly enforces Non-Negotiable Rule 2 (zero data leakage) and proposal research methodology standards.
- **Implications**: Provenance audit logs are saved at `artifacts/metrics/test_evaluation_provenance.json` and statistical comparison artifacts in `artifacts/leaderboard/research_question_answer.json`.

### [DEC-016] 2026-10-19: Composite Weighted Leaderboard Metric Specification and Formal Research Question Adjudication
- **Context**: Selecting a production champion model based on a single scalar metric (e.g., ROC-AUC alone) ignores crucial operational trade-offs, including minority class precision-recall (PR-AUC), probability reliability (Brier Score, ECE), recall-weighted safety ($F_2$ Score), early warning lead time, and inference latency constraints (Proposal NFR1: < 2 s).
- **Alternatives Considered**:
  - *Option A*: Rank purely by test AUROC.
  - *Option B*: Rank by unconstrained $F_1$ score.
  - *Option C*: Multi-objective composite weighted rank combining PR-AUC (0.25), ROC-AUC (0.20), $F_2$ score (0.20), Brier score (0.15), Expected Calibration Error (0.10), and Early Warning Lead Time (0.10), filtered through strict eligibility constraints (latency $\le$ 2,000 ms, multi-seed std $\le$ 0.05, exclusion of reference-only rules).
- **Decision**: Option C. Imbalanced academic attrition requires prioritizing the minority class (PR-AUC 0.25) and recall-weighted safety ($F_2$ 0.20), while ensuring continuous risk index quality and lead-time prognostic utility. Models failing latency or stability gates are disqualified from champion selection. The research question comparing sequence models vs gradient boosting is formally adjudicated through statistical significance of $\Delta \text{AUC}$ and $\Delta \text{PR-AUC}$ with explicit synthetic data honesty declarations.
- **Proposal Alignment**: Fulfills Proposal Section 3.6, Proposal NFR1 (< 2 s), and Non-Negotiable Rule 3 (honesty about synthetic data).
- **Implications**: All 28 models are ranked in `artifacts/leaderboard/leaderboard.json`, `.csv`, `.md`, category champions are saved in `category_champions.json`, and the champion in `champion.json`.
### [DEC-017] 2026-10-19: Clinical Academic Risk Banding and Validation-Derived Cut-Points
- **Context**: In higher education retention programs, a raw probability or scalar index without clinical operational tiers fails to guide faculty triage. Furthermore, standard symmetric quartiles (e.g., 0–25, 25–50, 50–75, 75–100) are mathematically inappropriate under severe class imbalance ($\approx 6.4\%$ attrition prevalence) because over 90% of students would land in the first quartile.
- **Alternatives Considered**:
  - *Option A*: Uniform equal-width quartiles (0–25, 25–50, 50–75, 75–100).
  - *Option B*: Standard binary classification (At-Risk vs Not-At-Risk) without intermediate urgency tiers.
  - *Option C*: Four-tier clinical banding (Low, Moderate, High, Critical) anchored to validation set distributions and the recall-constrained operational threshold: Low ($[0.0, 5.0)$), Moderate ($[5.0, 12.0)$), High ($[12.0, 22.0)$), and Critical ($[22.0, 100.0]$).
- **Decision**: Option C. Anchoring the Low/Moderate boundary ($5.0\%$) to the validation recall-constrained cutoff ensures that students requiring monitoring align with institutional counseling capacity. The Moderate/High cutoff ($12.0\%$) represents a $2\times$ attrition odds threshold, while Critical ($\ge 22.0\%$) isolates the top 5–7% extreme-risk cohort requiring emergency dean-level review.
- **Proposal Alignment**: Fulfills User Requirements FR1/FR2 and supports shared Fusion Framework multi-tier triage.
- **Implications**: The `AcademicRiskIndexEngine` is defined in `backend/src/explain/risk_index.py` and exported in all fusion payloads.

### [DEC-018] 2026-10-19: Cooperative Game-Theoretic Tree SHAP and Deterministic Narrative Templates for Fusion Handoff
- **Context**: Faculty advisors require trustworthy local explanations indicating why a student is at risk. Generative LLMs (e.g., GPT, T5) are prone to hallucinating non-existent reasons, while perturbation surrogates like LIME suffer from sampling variance and violate the efficiency axiom.
- **Alternatives Considered**:
  - *Option A*: Generative LLM prompt completion from raw tabular features.
  - *Option B*: LIME local linear surrogates.
  - *Option C*: Exact Tree SHAP (`shap.TreeExplainer`) for feature attributions combined with Integrated Gradients for temporal sequence attribution, driving a deterministic rules-based diagnostic template compiler validated via a formal feature deletion test.
- **Decision**: Option C. Tree SHAP guarantees mathematical efficiency, missingness, symmetry, and monotonicity. Integrated Gradients provides axiomatic completeness over time. The rules-based narrative engine compiles clinical sentences strictly parameterized by the student's verified top-3 SHAP attributions, guaranteeing zero hallucinations. Faithfulness is empirically proven via monotonic risk reduction ($13.7 \rightarrow 3.4$) under feature neutralization.
- **Proposal Alignment**: Fulfills Proposal Section 3.7 (Explainability), Non-Negotiable Rule 3 (honesty), and NFR1.
- **Implications**: Pydantic schema `FusionAcademicRiskPayload` in `backend/src/explain/fusion_contract.py` guarantees strict contract compliance for teammates' Fusion Framework.

### [DEC-019] 2026-10-19: Production API Architecture, OpenAPI Specification, and Schema Validation
- **Context**: The Academic Intelligence Framework must expose an operational interface for interactive early warning, batch university cohort ingestion, and teammate integration via the shared Fusion Framework.
- **Alternatives Considered**:
  - *Option A*: Ad-hoc Flask REST server with manual JSON parsing.
  - *Option B*: Django REST framework with heavyweight ORM and database dependencies.
  - *Option C*: Asynchronous FastAPI serving engine with strict Pydantic v2 boundary validation, OpenAPI 3.1 documentation, auto-detection and conversion of wide-format tables, and row-indexed error feedback.
- **Decision**: Option C. FastAPI provides high-throughput ASGI performance, native OpenAPI 3.1 interactive documentation (`/docs`, `/redoc`), and automatic Pydantic validation. The validator supports both canonical long format and auto-detected wide format, delivering clear, row-indexed error diagnostics on malformed inputs.
- **Proposal Alignment**: Directly fulfills Specific Objective 2.4, User Requirements FR1/FR2, and Proposal NFR1 (< 2.0 s per student).
- **Implications**: Core service defined in `backend/src/api/service.py`, router in `backend/src/api/main.py`, and endpoints tested in `backend/tests/test_api.py`.

### [DEC-020] 2026-10-19: Zero-PII In-Memory Processing, Ephemeral Caching, and Security Controls
- **Context**: Handling student academic records introduces strict ethical, privacy, and regulatory constraints (e.g., FERPA, GDPR). Persistent disk storage of uploaded files risks data exposure.
- **Alternatives Considered**:
  - *Option A*: Save uploaded CSV/Excel files to local disk under `/data/uploads/` and log student IDs for audit trails.
  - *Option B*: Stream file uploads entirely in-memory (`io.BytesIO`), enforce a 10 MB file size ceiling (HTTP 413), sanitize all application logs to eliminate PII, and maintain an ephemeral, self-pruning in-memory batch download cache keyed by random UUIDs.
- **Decision**: Option B. Zero persistent disk storage of raw uploads ensures student privacy by design. Application logs record only anonymized row counts, file byte sizes, and latency metrics. Batch exports are cached in memory with a 50-session FIFO eviction policy.
- **Proposal Alignment**: Enforces Proposal Section 3.2 (Ethics and PII constraints) and Non-Negotiable Rule 3 (honesty and data integrity).
- **Implications**: Uploaded streams are never persisted to disk; batch downloads expire automatically upon process restart or cache eviction.

### [DEC-021] 2026-10-19: Interactive Research Portal Architecture, Presentation Mode, and Zero-Hardcoding Contract
- **Context**: Presenting research results to academic viva panels and potential institutional users requires a polished, accessible portal that faithfully reflects backend telemetry without hardcoding numbers.
- **Alternatives Considered**:
  - *Option A*: Static HTML reports generated from evaluation notebooks.
  - *Option B*: Standard Streamlit or Gradio dashboard with sluggish multi-page state management.
  - *Option C*: Production single-page React 18 + Vite 6 application styled with Tailwind CSS, Recharts, persistent synthetic data banners, a high-contrast projector Presentation Mode, and a strict zero-hardcoding contract consuming live FastAPI endpoints.
- **Decision**: Option C. React 18 provides instantaneous client-side interactivity, responsive drag-and-drop batch file scoring, dynamic single-student trajectory simulation, and accessible keyboard navigation. Presentation Mode scales typography and contrast for large conference halls, while the persistent synthetic banner enforces scientific integrity.
- **Proposal Alignment**: Fulfills Specific Objective 2.4, Proposal NFR1 (< 2 s), and Non-Negotiable Rule 3 (honesty about synthetic data).
- **Implications**: Component tree organized under `frontend/src/pages/`, typed API client in `frontend/src/api/client.ts`, and full-page screenshots preserved in `docs/screenshots/`.


