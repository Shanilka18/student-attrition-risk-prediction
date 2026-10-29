# Environmental & Financial Factors

This repository implements the Environmental & Financial component of the
student attrition risk prediction project. Its website-facing product name is
**Study Conditions Index**. The component produces an Environmental &
Financial Index (EFI) from environmental and financial study conditions.

The current data source is generated sample data for pipeline validation. The
intended real-data source is one anonymous, closed-ended questionnaire, selected
through configuration when real survey data is available. No real survey
findings or model-performance results are reported here.

## Research design

- The component uses the 17 environmental and financial items in
  `docs/questionnaire_envfin_v2.md`; shared outcome items are supplied by the
  group questionnaire.
- Survey outcomes, consent, attention checks, identifiers, and timestamps are
  excluded from model features. The primary target is self-reported
  `attrition_intent_score`, an intention proxy rather than observed withdrawal.
- The binary label `at_risk` is derived using the configured score threshold.
  The EFI is the rounded, calibrated predicted probability of that label on a
  0-100 scale; higher values indicate higher risk linked to study conditions.
- Feature engineering derives Financial Stress, Commute Burden, Digital Access
  Gap, and Study Environment Pressure indices, alongside rescaled items and
  selected ratios/interactions.
- Latent risk profiles are discovered through clustering methods and are fitted
  within validation folds. They are descriptive and are not inputs to the main
  classifier.
- Model evaluation follows the two-track design, validation leaderboard,
  overfit rule, and one-time held-out test protocol in `PROJECT_BRIEF.md`.
  Explanations use SHAP; predictive attributions are not causal effects.
- Fairness is assessed and reported by eligible groups, suppressing small
  groups. The score is a research output and is not for individual decisions.

## Data and transparency

`config/project.yaml` sets `project.data_source` to `sample` or `survey`.
The selected source is recorded in run metadata. Switching to real survey data
is intended to be configuration-driven, without changing the research design.

Sample outputs must remain clearly identified as sample data and must not be
presented as real findings. The design has important limits: responses are
self-reported, the label measures intention, the questionnaire is
cross-sectional, and household income is sensitive. Participation must be
voluntary and non-disclosure must be respected.

## Quick start

On Windows, provision the project environment and validate the survey schema:

```powershell
.\tasks.ps1 setup
.\.venv\Scripts\python.exe -m efi.cli --help
.\.venv\Scripts\python.exe -m efi.cli setup
.\.venv\Scripts\python.exe -m efi.cli profiles
```

Run the tests with:

```powershell
.\.venv\Scripts\pytest.exe tests\ -q
```

The CLI also provides `info`. The Makefile and `tasks.ps1` expose pipeline
targets for data generation, cleaning, training, evaluation, explanation,
profiles, API, and frontend tasks.

## Repository guide

- `PROJECT_BRIEF.md` — active component design and acceptance rules.
- `config/project.yaml` — project identity, data source, targets, EFI bands,
  profile method, leaderboard weights, and website page configuration.
- `config/schema_envfin.yaml` and `config/mapping_envfin.yaml` — questionnaire
  schema and response mappings.
- `src/` — data generation, cleaning, preprocessing, modeling, evaluation,
  explanation, inference, API, and related pipeline modules.
- `efi/cli.py` — setup, profiles, and project information commands.
- `frontend/` — website implementation.
- `docs/` — research context, decisions, questionnaire, ethics, and technical
  documentation.
- `data/legacy/` and `docs/legacy/` — archived engagement-era data and
  documents; these are not active inputs to the Environmental & Financial
  design.

See `docs/01_project_context.md` for the concise project context and
`docs/DECISIONS.md` for active design decisions.
