> Summary: This guide explains setup, pipeline runs, documentation, tests, and retraining.
> Use a new run for any full evaluation; never reopen the current held-out test.
> Keep real survey files out of source control and use approved consented data only.
> Generated data are for pipeline checks, not real-student evidence.
> Paths below are project-relative and outputs are preserved under their owning folders.

# Reproducibility and Run Guide

## Environment setup

Use the repository's supported Python version from [pyproject.toml](../pyproject.toml). In Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

The pinned requirements file includes the model, plotting, and test dependencies. The editable install makes `mei` available as a Python module. If only running the docs scripts from source, `python scripts/build_docs_numbers.py` can run without installing the project package; the CLI itself still needs its dependencies.

For frontend work, install the lockfile dependencies and use the declared frontend scripts:

```powershell
npm --prefix frontend ci
npm --prefix frontend run build
npm --prefix frontend run test
```

Run the API with `python -m mei.cli serve` in one terminal and the frontend with `npm --prefix frontend run dev` in another. The browser client requires the API and frontend proxy configuration to agree. Do not expose a development service publicly.

## Reproduce the analytical pipeline

The commands below are the implemented phase workflow. Full model commands can take a long time. Run them only for a new, intentionally created run; do not run `finalize` on the active run in this checkout because its one-time test marker is already present.

| Phase | Command or action | Main output |
|---|---|---|
| Environment and schema | `python -m mei.cli setup` | Validated questionnaire and configuration. |
| Generated sample | `python -m mei.cli make-sample-data` | Synthetic questionnaire exports and exploratory figures. |
| Quality control | `python -m mei.cli clean --input data/raw/survey_raw_sample.csv --mode training` | Cleaned cohort and quality report under `data/interim/` and run artifacts. |
| Features and splits | `python -m mei.cli features` | Leakage audit, feature sets, student partitions, and CV split artifacts. |
| States | `python -m mei.cli states` | Cluster and HMM summaries under the run's `states/` folder. |
| Classification | `python -m mei.cli train-classic --quick` | Quick smoke benchmark. Remove `--quick` for the full model comparison. |
| Deep models | `python -m mei.cli train-deep --quick` | Quick smoke benchmark. Remove `--quick` for the full model comparison. |
| Regression | `python -m mei.cli train-regression` | Track B regression leaderboard and diagnostics. |
| Final evaluation | `python -m mei.cli finalize` | One-time held-out evaluation, calibration, model card, and risk bands. |
| Explanations | `python -m mei.cli explain` | SHAP summaries and consistency artifacts. |
| Viva documentation | `python -m mei.cli docs` | `_numbers.json`, generated results, final figures, catalog, and documentation checks. |

`python -m mei.cli all` is currently a stub; it is not a substitute for the commands above. Check `python -m mei.cli --help` and each command's `--help` before a new experiment. The order and results of historical phases are recorded in [phase reports](./phase_reports/).

## Generate sample data

`python -m mei.cli make-sample-data` writes generated questionnaire examples and exploratory figures. It can change generated data and the active experiment context. Run it in a safe fresh workspace or preserve the current run first. Never describe these records as recruited students. The current checkout does not contain a `data/raw/` directory, so confirm the path printed by the generator before passing it to `clean`.

## Retrain on real survey data

1. Obtain the required ethics and consent approvals.
2. Save the approved, pseudonymized file outside version control, for example under `data/real/`.
3. Check its schema and quality report. Do not upload names or direct identifiers.
4. Create a candidate run without automatic activation:

   ```powershell
   python -m mei.cli retrain --data data/real/approved_survey.csv --mode quick
   ```

5. Inspect the new run's source marker, cohort flow, cleaning report, feature manifest, split, CV results, calibration, confidence intervals, limitations, and model card.
6. Confirm students were split before fitting, preprocessing was fitted inside folds, and A1–A3 were excluded from features.
7. Use a held-out test once. If it has already been used, do not repeat or tune against it.
8. Regenerate the docs with `python -m mei.cli docs`. Only consider activation after a separate governance review:

   ```powershell
   python -m mei.cli activate RUN_ID --actor "Authorized reviewer"
   ```

The quick mode is not evidence that a model is deployment-ready. A full retraining should follow the current pipeline's supported mode and compute budget, followed by independent review. The active generated-data test must not be re-evaluated.

## Test and lint commands

The documentation-specific acceptance checks are:

```powershell
python -m mei.cli docs
python scripts/check_docs.py
```

For a future full software test run, use:

```powershell
python -m pytest
ruff check src tests scripts
black --check src tests scripts
npm --prefix frontend run test
npm --prefix frontend run build
```

This guide does not claim that the full suite was run during the documentation phase. The documentation checks do not retrain models or evaluate the held-out test.

## Output locations

| Output | Location |
|---|---|
| Active run pointer | `artifacts/active_run.txt` |
| Per-run model, metric, split, state, and explanation artifacts | `artifacts/runs/RUN_ID/` |
| Generated and real input data | `data/raw/`, `data/real/` (keep private data untracked) |
| Cleaning intermediates and quality audit | `data/interim/` |
| Number snapshot | `docs/_numbers.json` |
| Generated results | `docs/RESULTS_SUMMARY.md` and `docs/19_RESULTS_SUMMARY.md` |
| Final figure exports | `docs/figures/final/` |
| Figure inventory | `docs/20_FIGURE_CATALOG.md` |
| Phase reports and decisions | `docs/phase_reports/` and `docs/DECISIONS.md` |
| Production frontend build | `frontend/dist/` |

## Troubleshooting

| Symptom | Likely cause | Safe next step |
|---|---|---|
| `No module named mei` | The package is not installed or the source path is not configured. | Activate `.venv`, install with `python -m pip install -e .`, then retry. |
| `python -m mei.cli docs` cannot import a dependency | The environment is incomplete. | Install the pinned requirements, then the editable package. |
| Active run or required artifact is missing | The run pointer is stale, or the selected run is incomplete. | Inspect `artifacts/active_run.txt`; do not replace it by guessing. Select a complete run through the project's workflow. |
| Training cleaning counts are unavailable | The active run has no matching training quality report. | Report them as unavailable and generate a matching report from an approved new run. Do not substitute the separate scoring audit. |
| `TEST_USED.lock` exists | The run's held-out set has already been evaluated. | Do not rerun final evaluation or tune on the test. Start a new governed experiment if appropriate. |
| SVG and PNG look different | The SVG source may be a separate vector export. | Compare the matched source and final files; use the PNG if presentation software renders SVG incorrectly. |
| Frontend builds but has no data | The API is stopped or the local proxy is misconfigured. | Start the API locally, check the proxy target, and use static screenshots if the service is unavailable. |
| npm reports missing packages | Frontend dependencies have not been restored. | Run `npm --prefix frontend ci` from the repository root. |

## Common mistakes

- Running final evaluation twice on one held-out test.
- Activating a candidate before governance review.
- Checking generated sample data into a real-data claim.
- Fitting preprocessing using validation or test students.
- Treating a successful build as evidence of predictive validity.

## If the examiner asks

Show the active-run pointer, the artifacts that support the claim, and the exact command used to generate them. State whether the run is generated or survey-based and whether its held-out test has already been used.
