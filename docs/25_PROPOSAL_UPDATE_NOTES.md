> Summary: This document provides paste-ready proposal replacement text.
> It aligns the proposal with a repeated survey design and intention labels.
> It distinguishes model development from one-time held-out evaluation.
> Bracketed author prompts require real-survey details after collection.
> No unverified external research citations are added.

# Proposal Update Notes

Replace the matching proposal text with the sections below. Keep the final wording consistent with the approved questionnaire and institutional ethics conditions.

## Abstract: revise the data description

This study will investigate whether self-reported engagement patterns across a course can support early identification of students who report an intention to leave. The planned inputs are questionnaire responses collected at repeated time points. The outcome is self-reported intention, not verified course withdrawal. Generated sample data are used only to develop and exercise the pipeline before the real survey is available. After approval and collection, the analysis will report participant recruitment, consent, exclusions, missingness, and the final cohort size. Classification and continuous-score models will be compared using student-level splits, repeated stratified cross-validation, and a single held-out test evaluation. Results will be interpreted as research evidence for supportive outreach, not as an automated decision about an individual student.

**Author update after collection:** Replace the sample-data sentence with the real recruitment frame, dates, consent process, and participant flow. Insert [AUTHOR: real cohort count] and [AUTHOR: exclusions and reasons] only after checking the final data-quality report.

## Section 2.2: specific objectives

1. Define and document a consent-based repeated questionnaire for self-reported engagement and intention to leave.
2. Apply a prespecified quality-control process and describe response completeness and exclusions.
3. Construct leakage-safe engagement summaries and temporal features while excluding outcome items A1–A3 from predictors.
4. Explore interpretable engagement profiles and sequence states, then test whether those derived features add value beyond aggregate features.
5. Compare classification and regression baselines with more complex candidates using the same student-level validation design.
6. Estimate discrimination, threshold performance, error, calibration, uncertainty, and generalization without repeated use of the held-out test set.
7. Produce cautious, non-causal explanations and define a human-support-only use boundary.

## Section 3.2: data, participants, and inputs

The study will recruit [AUTHOR: eligible population and setting] using [AUTHOR: recruitment route]. Participation will be voluntary and based on informed consent. The questionnaire records self-reported engagement indicators at repeated course time points and includes outcome items A1–A3. These outcome items define the target and are excluded from the predictor matrix. The analysis file will use a study-specific pseudonymous identifier; direct identifiers and the linkage key will be stored separately with access restrictions. Report the recruitment dates, response rate, valid cohort, exclusions, missingness, and any departures from the approved protocol after data collection.

The current development run uses generated sample data. It is not a real participant cohort and must not appear as completed survey evidence. Replace all bracketed details with values read from the approved survey and final data-quality artifacts: [AUTHOR: invited count], [AUTHOR: consented count], [AUTHOR: analyzed count], [AUTHOR: exclusion counts], and [AUTHOR: missingness summary].

## Section 3.3: baselines

The analysis will compare a majority-class or prevalence reference, logistic regression, and regularized linear regression with tree-based models, ensemble methods, and temporal neural candidates where the sequence size supports them. State-based features will be evaluated as an ablation against aggregate features. Models will receive the same participant-level partitions and outcome definitions. Preprocessing, imputation, scaling, feature selection, and resampling, where justified, will be fitted using training data within each cross-validation fold. The final model choice will be based on prespecified validation criteria and practical calibration and complexity considerations, not on held-out test performance.

## Section 3.4: evaluation metrics

| Task | Metric | Purpose |
|---|---|---|
| Classification | Balanced accuracy | Gives each outcome class equal influence. |
| Classification | F1 | Summarizes precision and recall at a declared threshold. |
| Classification | ROC AUC | Measures ranking across thresholds. |
| Classification | PR AUC | Describes precision-recall trade-offs, especially for a less common positive class. |
| Classification | MCC | Uses all four confusion-matrix cells in a balanced summary. |
| Classification | Calibration and Brier score | Checks whether probabilities agree with observed frequencies and squared error. |
| Regression | R-squared | Compares fit with a mean-only reference. |
| Regression | RMSE and MAE | Report squared-error-sensitive and absolute-error magnitudes in target units. |
| Interpretation | SHAP consistency | Measures whether feature attribution rankings persist across folds; it is not a causal metric. |
| Generalization | Generalization gap | Compares training and validation performance for overfit warning. |
| Uncertainty | Confidence intervals | Shows sampling uncertainty for selected held-out metrics. |

Report [AUTHOR: final classification metric values and intervals], [AUTHOR: final regression errors and intervals], [AUTHOR: calibration results], [AUTHOR: SHAP consistency], and [AUTHOR: generalization gap] from the generated result artifacts. Mark nonsignificant comparisons explicitly; do not convert a small numerical difference into a claim of improvement.

## Section 3.5: validation strategy

Create participant-level train, validation, and held-out test partitions before fitting any data-dependent transformer. Use repeated stratified cross-validation within the training partition for model and hyperparameter comparison. Fit imputers, scalers, encoders, feature selectors, and resampling steps inside each training fold only. Use validation evidence for model selection and threshold decisions according to a prespecified rule. Freeze the selected model and analysis plan before opening the held-out test. Evaluate that test once, record the run and its metrics, and do not use it to tune thresholds or choose features. For a small cohort, report fold variability and intervals and state the resulting uncertainty plainly.

## Section 4.x: architecture

The proposed system has a data-ingestion and quality-control stage, a leakage-safe feature stage, a student-level split and cross-validation stage, separate classification and regression comparisons, and an evaluation and reporting stage. The optional state-analysis branch derives clusters and hidden-state sequence summaries from permitted predictors and returns to the same validation process. The selected model, preprocessing objects, outcome definition, threshold, feature manifest, and source-data label are versioned together. The user-facing website presents bounded analysis results and supports human review; it does not make an educational decision. Retraining requires a new quality report, a new validation cycle, governance review, and explicit activation.

## Website functional requirements

- Display the project purpose, data source, model version, and limitations clearly.
- Validate uploaded files and explain quality problems before scoring.
- Show uncertainty and the distinction between self-reported intention and actual withdrawal.
- Require authorized human review and offer a way to correct or challenge input data.
- Avoid presenting a score as a diagnosis, disciplinary flag, or decision.
- Protect identifiers, minimize retention, restrict access, and record governed model activation.
- Keep research documentation in the project documentation area, not on the website.

## Topic Assessment Form update list

Section 10 Tasks and Novelty is already updated. Review the remaining parts for consistency: title and abstract; aim and research question; objectives; participant recruitment and consent; survey inputs and outcome definition; data-protection plan; analysis and baseline models; validation strategy; metrics; risk controls; system scope; timeline and deliverables. Change any wording that describes confirmed dropout, administrative records, or a live deployment unless these are separately approved and evidenced.

## Citation handling

Use the official documentation for the implemented software libraries where implementation details need support. For research-method claims or background evidence not already verified in the proposal bibliography, write “citation to be added” and obtain an accurate source before submission.

## Common mistakes

- Leaving a bracketed author prompt in a submitted proposal.
- Describing generated records as recruited students.
- Including A1–A3 in the input features.
- Selecting a model or threshold using the held-out test.

## If the examiner asks

Explain which text describes the planned real survey and which evidence currently comes only from generated data. Show the data-quality and split artifacts before inserting final cohort values.
