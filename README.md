# Multimodal Student Engagement Intelligence

MEI is a research system for studying repeated, self-reported engagement patterns and reported intention to leave a course. It generates an Engagement Index from 0 to 100 and provides model comparisons, explanations, and a local web interface. It is not a confirmed-dropout predictor or an automated decision tool.

## Data status

The repository does not contain real survey responses or trained model runs. The first run generates sample survey records. Results from that data test the software workflow only; they do not establish performance on real students. The outcome items are self-reported intentions, not confirmed withdrawals.

## Five-minute quick start

Requirements: Python 3.11, Node.js 20 or later, and about 8 GB of available memory. Model training can take longer than five minutes.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools
python -m pip install -r requirements.txt
python -m pip install -e .
Set-Location frontend
npm ci
npm run build
Set-Location ..
python -m mei.cli all
.\start.ps1
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -r requirements.txt
python -m pip install -e .
(cd frontend && npm ci && npm run build)
python -m mei.cli all
./start.sh
```

Open <http://127.0.0.1:8000>. The `all` command records each stage's exit status and duration in `artifacts/pipeline_runs/`. Configured model-training budgets total no more than the hardware limit in `config/project.yaml`. The pipeline stops at the first failed stage.

## The five commands that matter

| Command | Purpose |
|---|---|
| `python -m mei.cli setup` | Validate survey schema and configuration. |
| `python -m mei.cli all` | Generate sample data, run every modeling phase, freeze the champion, explain it, and rebuild docs. |
| `python -m mei.cli serve` | Start the local API and built web application. |
| `python -m mei.cli retrain --data PATH --mode quick` | Train a candidate run from a new survey file; review it before activation. |
| `python -m mei.cli docs` | Rebuild artifact-backed result documents and verify links and figures. |

Use `python -m mei.cli --help` for all commands and options. `python -m mei.cli all` is a full, time-boxed research pipeline, not a five-minute operation.

## Retraining with real survey data

1. Obtain institutional approval and informed consent before collecting responses.
2. Keep the identity key separate; place the survey export in the ignored `data/real/` directory.
3. Validate headers and response quality with the scoring/retraining validation page or CLI.
4. Run `python -m mei.cli retrain --data data/real/survey.csv --mode quick` to build a candidate; use `--mode full` only when the configured time and hardware budget permit.
5. Review the cohort, cleaning report, feature manifest, validation results, calibration, uncertainty, subgroup evidence, and data/model cards.
6. Do not reuse a held-out test that has already been evaluated. The current saved artifacts contain a test-use provenance conflict; resolve the protocol on a new, preregistered cohort before making test claims.
7. Activate a candidate only after independent review: `python -m mei.cli activate RUN_ID --actor "reviewer"`.
8. Rebuild the docs with `python -m mei.cli docs` and confirm their source label and values before presenting results.

Current results use generated data. The label is self-reported intention, the survey spans four time points, and the model must support a conversation—not decide a student's status.

## Folders

| Path | Contents |
|---|---|
| `config/` | Survey schema, labels, model definitions, and run budgets. |
| `data/templates/` | Blank CSV and Excel survey templates. |
| `data/raw/` | Generated sample input files; recreated by the pipeline. |
| `data/real/` | Private real survey exports; ignored by Git. |
| `data/interim/` | Regenerable cleaned datasets; ignored by Git. |
| `src/mei/` | CLI, cleaning, modeling, evaluation, API, and scoring code. |
| `frontend/` | React web client and its lockfile. |
| `artifacts/runs/` | Local run outputs, models, reports, and evaluation locks; not tracked. |
| `docs/` | Technical and viva guides, decisions, reports, and figures. |
| `tests/` | Python tests, frontend tests, and upload fixtures. |

## Tests and release checks

```powershell
python -m pytest
python -m ruff check src tests scripts
python -m black --check src tests scripts
Set-Location frontend
npm ci
npm test
npm run build
npm run check:tokens
```

The repository's `make release-check` runs the configured end-to-end pipeline and test/build checks. It is intentionally time-consuming because it retrains models and evaluates a new generated run. Do not rerun evaluation against an existing run after its `TEST_USED.lock` has been created.

## Starting the product later

After setup, pipeline generation, and frontend build have completed, use `.\start.ps1` in PowerShell or `./start.sh` on Linux/macOS. Both use the local virtual environment and bind the server to `127.0.0.1` by default. Pass `--host`, `--port`, or `--reload` after the script name to change the server options.

For detailed phase commands and output locations, read [26_REPRODUCIBILITY_AND_RUN_GUIDE.md](docs/26_REPRODUCIBILITY_AND_RUN_GUIDE.md). For viva material, start at [00_README_DOCS.md](docs/00_README_DOCS.md).
