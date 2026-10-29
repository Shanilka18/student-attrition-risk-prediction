# Proposal Replacement Text — Environmental & Financial Component

Paste the replacement under the matching heading in the proposal. These passages describe the survey-based component only; they do not claim that real data have been collected or evaluated.

## Abstract

This study develops an Environmental & Financial component for a multi-factor student attrition-risk research project. A voluntary, closed-ended questionnaire will capture household income, financial aid, family financial duties, study-cost difficulty, commute conditions, living situation, study space, internet and device access, mobile data, part-time work, financial shocks, money worry, and missed classes or deadlines related to these barriers. After quality checks and leakage-controlled preprocessing, four derived indices will summarize Financial Stress, Commute Burden, Digital Access Gap, and Study Environment. Clustering or latent-profile methods will describe recurring study-condition risk profiles. Gradient Boosting (XGBoost or LightGBM) will be compared with Logistic Regression and Random Forest for the shared binary at-risk label; a separately specified regression track will evaluate the continuous self-reported attrition-intention score. SHAP will explain model contributions, and group-wise fairness will be reported with small-cell suppression. The Environmental & Financial Index (EFI) is a 0–100 higher-risk score derived from calibrated risk estimates when calibration is adequate. It is not a diagnosis or a prediction of actual withdrawal, and the website is intended for aggregate research reporting rather than decisions about individual students. Findings from the real survey, including held-out performance, discovered profiles, and subgroup results: [INSERT AFTER REAL DATA]

## 2.2 Specific Objectives

1. Collect and preprocess voluntary closed-ended questionnaire responses on the environmental and financial study conditions represented by items Q23–Q39, retaining conditional and not-applicable response meanings and keeping shared outcome and administrative fields out of the predictor set.
2. Derive Financial Stress, Commute Burden, Digital Access Gap, and Study Environment indices using documented item mappings, and describe latent environmental and financial risk profiles using clustering or latent-profile methods fitted within the training folds.
3. Compare Gradient Boosting (XGBoost or LightGBM) with Logistic Regression and Random Forest for the shared binary at-risk label, and evaluate a separately specified regression model for the continuous self-reported attrition-intention score using the pre-specified classification and regression metrics.
4. Explain model behaviour with SHAP, assess the consistency of explanations across validation folds, and report group-wise performance subject to privacy-preserving small-group suppression.
5. Produce and document the Environmental & Financial Index (EFI), on a 0–100 scale where higher values indicate greater estimated study-condition risk, with calibration limitations stated wherever probability calibration is inadequate.

## 3.2 Data, Participants, and Experimental Inputs

The analysis will use one voluntary, closed-ended questionnaire administered to eligible undergraduate participants after the required ethics and institutional approvals. Environmental & Financial predictors comprise the 17 items Q23–Q39: household income band, aid status, family support duty, study-cost difficulty, commute time, commute spend, transport mode, living situation, quiet study place, home internet, main device, mobile data plan, part-time work, conditional job hours, financial shock, money worry, and missed classes or deadlines due to barriers. Conditional job-hour responses and structural not-applicable commute responses will be represented according to the approved questionnaire and data dictionary.

The shared outcome is self-reported attrition intention, not verified withdrawal. The continuous target is `attrition_intent_score` (0–100), and the shared binary label is `at_risk = 1` when `attrition_intent_score` is at least 60. The group-approved common outcome items and scoring rule must be identical across components. The predictors are the Environmental & Financial questionnaire items and their pre-specified derived indices; the outcome items, `student_id`, `timestamp`, `consent`, and `attention_check` are excluded from model features. No institutional academic or learning-platform records are part of this component's specified input.

Use the proposal's existing feasibility target of at least 300 eligible records only if it is confirmed by the supervisors and the ethics-approved sampling plan. The four components must use the same approved sample-size rule. No synthetic or generated records will be described as real participants or real findings.

## 3.3 Baseline and Benchmark Models

The single-factor baseline will use household income band (`income_band`) as its only predictor. For the binary `at_risk` outcome, fit a Logistic Regression model with income band represented as a categorical predictor. For the continuous `attrition_intent_score`, use a corresponding single-predictor linear regression baseline. These baselines test how much predictive information is available from income alone; they do not represent the full component.

Compare the binary baseline with multivariable Logistic Regression, Random Forest, and Gradient Boosting (XGBoost or LightGBM), using the same participants, outcome definition, preprocessing boundaries, validation folds, and evaluation metrics. Evaluate the continuous score in a separate regression track; do not apply classification metrics to the continuous target or regression metrics to the classifier. Preprocessing, imputation, derived-index construction, and profile discovery must be learned within training folds. Risk profiles are for descriptive analysis and are not inputs to the primary classifier. Report baseline and benchmark results only after evaluating real survey data, in the evaluation table in Section 3.4.

## 3.4 Evaluation Metrics

Report classification metrics for the shared binary `at_risk` label and regression metrics for the continuous `attrition_intent_score`. Accuracy is supplementary and must not be used alone for model selection. Choose any classification decision threshold using validation data only; do not tune it on the held-out test set. Calibration error refers to expected calibration error (ECE) and is reported with the calibration procedure and limitations. SHAP consistency summarizes agreement between global feature-importance rankings across training folds, using rank correlation and top-k overlap with k specified before final evaluation. The generalisation gap is the cross-validation-to-held-out-test difference for the same metric and model; it is descriptive and does not permit post-test tuning.

| Metric | Target and interpretation | Real-survey result |
|---|---|---|
| Accuracy | Binary `at_risk`; supplementary proportion correctly classified at the validation-selected threshold. | [INSERT AFTER REAL DATA] |
| Balanced accuracy | Binary `at_risk`; average of sensitivity and specificity. | [INSERT AFTER REAL DATA] |
| F1 | Binary `at_risk`; F1 for the pre-defined positive (`at_risk`) class. | [INSERT AFTER REAL DATA] |
| AUC-ROC | Binary `at_risk`; ranking discrimination across thresholds. | [INSERT AFTER REAL DATA] |
| PR-AUC | Binary `at_risk`; precision–recall performance for the positive class. | [INSERT AFTER REAL DATA] |
| MCC | Binary `at_risk`; Matthews correlation coefficient at the validation-selected threshold. | [INSERT AFTER REAL DATA] |
| Calibration error | Binary predicted probabilities; expected calibration error, with the calibration method stated. | [INSERT AFTER REAL DATA] |
| R2 | Continuous `attrition_intent_score`; variance explained by the regression model. | [INSERT AFTER REAL DATA] |
| RMSE | Continuous `attrition_intent_score`; root mean squared prediction error, in score units. | [INSERT AFTER REAL DATA] |
| MAE | Continuous `attrition_intent_score`; mean absolute prediction error, in score units. | [INSERT AFTER REAL DATA] |
| SHAP consistency | Agreement of global SHAP rankings across folds, reported using rank correlation and top-k overlap. | [INSERT AFTER REAL DATA] |
| Generalisation gap | Difference between repeated-CV and one-time held-out-test performance for the same metric. | [INSERT AFTER REAL DATA] |

Report group-wise performance and error patterns for the groups approved in the analysis plan. Suppress any subgroup result that does not meet the pre-approved minimum cell-size rule; do not infer or disclose small-group results. Real-survey group-wise fairness and error-pattern findings: [INSERT AFTER REAL DATA]

## 3.5 Validation Strategy

Pre-specify the outcomes, feature mappings, metrics, subgroup comparisons, and analysis procedure before inspecting evaluation results. Use repeated stratified cross-validation for model development and comparison, stratifying folds by the shared binary at-risk label and keeping all records for a participant in one fold. Fit imputers, encoders, scalers where needed, derived indices, clustering or latent-profile models, and model parameters using training data within each fold; apply the fitted transformations to that fold's validation data without refitting. Select model settings and any classification threshold using cross-validation only.

Reserve a separate held-out test set before model selection. Lock the preprocessing and selected model before opening this set, evaluate it once for the final assessment, and do not use its results to revise features, thresholds, or model settings. Report cross-validation estimates, the one-time held-out results, and their generalisation gaps separately. Include uncertainty estimates where supported by the real sample size and report any limitations caused by inadequate subgroup counts or class representation.

## 4.1 Overview of Proposed Solution

The Environmental & Financial component follows a survey-to-aggregate-reporting pipeline. Consented questionnaire responses are exported to an access-restricted research location, validated against the approved schema, and stripped of direct identifiers before analysis. A leakage-controlled preprocessing pipeline constructs the four derived indices. Clustering or latent-profile analysis describes study-condition profiles within training folds. The supervised classification track compares Gradient Boosting with Logistic Regression and Random Forest; a separate regression track models the continuous intention score. Calibration, repeated-CV evaluation, one-time held-out testing, SHAP explanation, and group-wise fairness checks are completed before results are exposed to the website.

The website presents approved cohort-level methodology, metrics, profile summaries, global SHAP explanations, and group-wise evaluation with small-group suppression. It does not expose row-level responses, student-level scores, individual explanations, or recommendations about an individual student's services, eligibility, grading, or progression. EFI values are reported as aggregate research summaries and are labelled with their calibration status and self-reported-outcome limitation.

## 5.1 Functional Requirements — Website

- **FR1 — Method and data transparency:** Display the questionnaire-based design, outcome definition, feature/target separation, known limitations, and whether displayed results use synthetic pipeline-validation data or approved real survey data.
- **FR2 — Aggregate model evaluation:** Present classification and regression metrics, repeated-CV and held-out-test status, calibration status, and model comparisons without exposing participant-level records.
- **FR3 — Risk-profile summaries:** Show descriptive cohort-level summaries of the discovered environmental and financial profiles; do not show a profile label for an identifiable person.
- **FR4 — Explainability:** Display global SHAP summaries with a clear statement that they describe model associations, not causes or individual support needs.
- **FR5 — Group-wise fairness reporting:** Present only approved subgroup metrics after applying the pre-approved small-cell suppression rule; identify when a comparison cannot be reported safely.
- **FR6 — EFI reporting:** Show aggregate EFI distributions and the calibration caveat. The website shall not provide individual EFI scores, individual SHAP explanations, or student-level decisions.

## 5.2 Non-Functional Requirements — Website

- **Privacy and data minimisation:** Do not publish direct identifiers, pseudonymous IDs, timestamps, raw questionnaire responses, or row-level predictions. Keep the linkage and analysis files in the approved restricted storage location, not in public website assets or the source repository.
- **Small-group protection:** Apply the same approved suppression rule to all component dashboards, tables, exports, and visualisations; prevent users from reconstructing suppressed values through filters or repeated queries.
- **Research integrity and auditability:** Label synthetic and real data distinctly, preserve the one-time held-out-test boundary, record the analysis version and metric definitions, and distinguish predictive SHAP summaries from causal claims.
- **Usability and accessibility:** Use plain language, accessible colour and text contrast, keyboard-operable controls, and clear explanations of the self-reported intention outcome and score limitations.
- **Reliability:** Do not display missing, invalid, or unverified metrics as successful results; show an explicit unavailable or suppressed state when evidence is insufficient.

## 3.2.1 Ethics — Data Collection and Participant Procedures

Participation is voluntary and based on informed consent. The invitation and consent text will explain the study purpose, the self-reported nature of the outcome, the intended aggregate research use, and that participation or non-participation has no effect on grades, discipline, access to services, or academic standing. Participants may stop before submitting and may decline sensitive questions. Because household income is sensitive, the final approved instrument must provide a clear way to decline that item, such as a permitted non-disclosure response or optional response, and the analysis plan must retain that response as distinct from an observed income band.

The questionnaire will not request names, contact details, or institutional student numbers. If a shared pseudonymous `student_id` is used to merge component responses, the data must be described as pseudonymous rather than fully anonymous; the ID must not be a model feature, and any linkage information must be kept separate in access-restricted storage. Collect only the minimum fields needed for the approved study. Set access, retention, deletion, and sharing procedures in the approved data-management plan before collection.

Report findings at cohort level. Apply the same ethics-approved minimum cell-size suppression rule to every subgroup table, chart, export, and website view. Neither the EFI nor any model output will be used to make decisions about an individual student, including grading, discipline, eligibility, or access to support. Do not begin real-data collection until the required ethics and institutional approvals are documented; this text does not claim that approval has already been granted.

## Real-Data Insert Inventory

Replace every marker above only after real survey data have been approved, collected, quality-checked, and evaluated:

1. **Abstract:** Summarise actual held-out performance, discovered profile patterns, and approved subgroup findings without implying prediction of observed withdrawal.
2. **Section 3.4, Accuracy:** Report the real-survey accuracy for each evaluated model.
3. **Section 3.4, Balanced accuracy:** Report the real-survey balanced accuracy for each evaluated model.
4. **Section 3.4, F1:** Report the real-survey at-risk-class F1 for each evaluated model.
5. **Section 3.4, AUC-ROC:** Report the real-survey ROC-AUC for each evaluated model.
6. **Section 3.4, PR-AUC:** Report the real-survey PR-AUC for each evaluated model.
7. **Section 3.4, MCC:** Report the real-survey MCC for each evaluated model.
8. **Section 3.4, Calibration error:** Report real-survey ECE and the calibration method for each evaluated probabilistic model.
9. **Section 3.4, R2:** Report real-survey R2 for each evaluated continuous-score regression model.
10. **Section 3.4, RMSE:** Report real-survey RMSE for each evaluated continuous-score regression model.
11. **Section 3.4, MAE:** Report real-survey MAE for each evaluated continuous-score regression model.
12. **Section 3.4, SHAP consistency:** Report real-survey fold-level rank correlation and top-k overlap.
13. **Section 3.4, Generalisation gap:** Report the real-survey cross-validation-to-held-out-test difference for each reported metric and model.
14. **Section 3.4, Group-wise fairness:** Report approved subgroup performance and error patterns, applying the pre-approved suppression rule.
