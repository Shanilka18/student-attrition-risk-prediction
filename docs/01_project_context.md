# Project Context: Environmental & Financial Factors

## Purpose and scope

This repository implements the Environmental & Financial Factors component of
the student attrition risk prediction project. It studies how household
finances, travel, digital access, and the home study environment relate to
students' self-reported intention to leave their studies. The website product
is the **Study Conditions Index**; the component score is the **Environmental
& Financial Index (EFI)**.

The EFI is an explainable, calibrated model-risk score from 0 to 100. A higher
score indicates higher predicted risk associated with the measured study
conditions. It is not a diagnosis, a causal estimate, or a decision rule for
an individual student.

## Data and outcomes

The planned input is one anonymous, closed-ended questionnaire: the 17
Environmental & Financial items in `docs/questionnaire_envfin_v2.md`, together
with shared outcome and quality-control items defined by the team. The project
currently uses generated sample data to validate the pipeline. Real survey
data may be selected later through `project.data_source` in
`config/project.yaml`; the chosen source is recorded in run metadata.

The continuous outcome is `attrition_intent_score`. The binary `at_risk` label
uses the configured score threshold, and the EFI is calculated from the
calibrated probability of that label. Outcomes are never features. Consent,
attention-check responses, identifiers, and timestamps are also excluded from
the feature set.

Interpretation is limited by the cross-sectional, self-reported design. The
label is stated intention, not verified withdrawal; there is no time dimension
for measuring progression. Income questions are sensitive and must permit
non-disclosure.

## Analysis design

The pipeline preserves the stages of cleaning, feature engineering, profile
discovery, modeling, leaderboard evaluation, SHAP explanation, EFI scoring,
and website presentation. Raw questionnaire responses are rescaled into
pressure-oriented features. Four derived indices summarize Financial Stress,
Commute Burden, Digital Access Gap, and Study Environment Pressure. Selected
ratios and interactions may complement those features.

Latent profiles are discovered using cross-sectional clustering approaches
such as Gaussian mixture modeling or latent-class methods. Profile fitting
must occur inside validation folds to prevent leakage. Profiles describe
patterns in study conditions and are not features in the main classifier.
The analysis uses one questionnaire response per participant and does not
construct temporal features.

The two model tracks, validation leaderboard, overfit rule, and one-time
held-out test protocol are specified in `PROJECT_BRIEF.md` and
`config/project.yaml`. The validation decision threshold is tuned without
accessing the final test outcomes. SHAP describes model attribution; it does
not establish causation. Fairness reporting is group-wise and must suppress
small groups.

## Website and safeguards

The website provides Overview, Model Leaderboard, Risk Profiles, What Drives
Risk, Score Students, and Train on New Data pages. It must identify sample data
as pipeline-validation data until real survey data is configured. Predictive
attributions and any separately presented causal analysis must remain
distinguishable. Scores must not be used to make individual student decisions.

Participation in the survey must be voluntary. Access, retention, and sharing
of questionnaire data must follow the approved ethics and team data-sharing
arrangements. The ethics approval status must be stated only after it is
confirmed.

## Source of truth

`PROJECT_BRIEF.md` defines the component's active methodological requirements.
`config/project.yaml` provides operational settings, while
`docs/DECISIONS.md` records active design choices. Shared consent, outcome
coding, identifiers, and export procedures are governed by the team agreement.
The CLI is invoked as `python -m efi.cli`, with `profiles` as the
profile-discovery command.
