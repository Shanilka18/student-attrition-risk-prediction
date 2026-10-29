# Active Design Decisions

This log records the active methodological and engineering choices for the
Environmental & Financial Factors component. The component design is defined
in `PROJECT_BRIEF.md`; operational settings are held in `config/project.yaml`.
No model results are reported here.

## Decision 1: Cross-sectional Environmental & Financial scope

The component uses one anonymous, closed-ended questionnaire covering the
17 Environmental & Financial items and shared team outcome items. It is
cross-sectional: temporal periods, sequence models, and hidden Markov models
are out of scope.

## Decision 2: Sample and survey data sources

Generated sample data is used for pipeline validation until real survey data
is available. `project.data_source` selects `sample` or `survey`, and every run
records the selected source. Switching sources is configuration-driven; sample
results must not be presented as real findings.

## Decision 3: Outcome and feature separation

`attrition_intent_score` is the continuous outcome and represents
self-reported intention, not observed withdrawal. `at_risk` is derived using
the configured threshold. Outcomes, consent, attention checks, identifiers,
and timestamps are never model features.

## Decision 4: Environmental & Financial Index

The Environmental & Financial Index (EFI) is the rounded calibrated
probability of `at_risk`, expressed on a 0-100 scale; higher values indicate
higher predicted risk linked to study conditions. The published bands are
Low (0-34), Moderate (35-64), and High (65-100). The decision threshold for
classification is tuned on validation data. The EFI is a model-risk score,
not an individual decision or causal effect.

## Decision 5: Feature construction

Questionnaire items are rescaled into pressure-oriented features. The four
derived indices are Financial Stress, Commute Burden, Digital Access Gap, and
Study Environment Pressure. Selected ratio and interaction features may
complement these indices. Feature transformations must be fit using training
data only.

## Decision 6: Cross-sectional profile discovery

Latent profiles are discovered through clustering, Gaussian mixture modeling,
or latent-class approaches. Profile fitting occurs within validation folds to
avoid information leakage. Profiles support descriptive and group-wise
reporting; they are not inputs to the main classifier.

## Decision 7: Model evaluation protocol

The project retains its two model tracks and validation leaderboard. The
leaderboard prioritizes PR-AUC, ROC-AUC, calibration, balanced accuracy, and
macro F1 using the weights in `PROJECT_BRIEF.md` and `config/project.yaml`.
Models that exceed the configured validation-to-cross-validation overfit
tolerance are excluded. Repeated stratified cross-validation is used for
validation, and the held-out test set is accessed once for final evaluation.

## Decision 8: Explainability and fairness reporting

SHAP is used to describe predictive model attributions. These attributions
must not be described as causal effects. Performance and fairness are reported
by group where sample sizes support meaningful reporting; small groups are
suppressed to reduce disclosure risk.

## Decision 9: Website and transparency

The website product is named **Study Conditions Index**. Its pages are
Overview, Model Leaderboard, Risk Profiles, What Drives Risk, Score Students,
and Train on New Data. Sample-data views carry a transparency notice.
Predictive, SHAP, and causal content use distinct presentation conventions as
specified in the project brief.

## Decision 10: Ethics and intended use

Survey participation is voluntary. Income questions are sensitive and must
allow respondents to decline disclosure. Data storage and sharing follow the
approved ethics and team data-sharing arrangements. The ethics approval
status must not be claimed until confirmed. The EFI is for research and
aggregate support planning; it is not for individual student decisions.

## Pending confirmation

The team agreement is the shared source for consent wording, student
identifier format, timestamps, attention checks, outcome coding, export
ownership, merge key, minimum sample size, data-sharing rules, and ethics
approval status. Any unconfirmed item remains subject to supervisor and team
approval.
