# EF-05b Tracked-Term Inventory

## Scope and method

Inventory captured on `environmental-chamodya` before the rename. Searched
tracked file contents case-insensitively with `git grep -i -n` for:
`mei`, `Multimodal`, `Engagement`, `Moodle`, `HMM`, `hmmlearn`,
`PeriodReshaper`, `period`, `states`, `GRU`, `LSTM`, `Conv1D`, and `legacy`.
Tracked paths were also checked with `git ls-files`; this catches directory and
filename hits that content search does not show. The existing CLI package is
tracked as repository-root `mei/`, not `src/mei/`.

The line numbers below refer to the pre-migration tracked files. A line may
contain more than one listed term.

## Content hits by tracked file

| File | Hit lines and terms |
|---|---|
| `Dockerfile.api` | 42 — `period` (health-check option `start-period`) |
| `Makefile` | 57 — `mei` (CLI module path) |
| `PROJECT_BRIEF.md` | 107 — `Engagement`, `states`; 109 — `HMM`, `period` |
| `README.md` | 54–56, 78 — `mei`; 82 — `Engagement`, `legacy` |
| `config/project.yaml` | 181 — `legacy` (unknown-column test value) |
| `configs/schema.yaml` | 217, 224, 238 — `Engagement` (old schema fields and descriptions) |
| `data/legacy/interim/clean.csv` | 1 — `Engagement` (old feature header) |
| `data/legacy/processed/test.csv` | 1 — `Engagement` (old feature header) |
| `data/legacy/processed/train.csv` | 1 — `Engagement` (old feature header) |
| `data/legacy/processed/val.csv` | 1 — `Engagement` (old feature header) |
| `data/legacy/samples/sample_upload.csv` | 1 — `Engagement` (old feature header) |
| `data/samples/sample_upload_messy.csv` | 1 — `legacy` (intentional unknown test column) |
| `data/synthetic/data_dictionary.csv` | 22, 24 — `Engagement` (old generated feature metadata) |
| `data/synthetic/data_dictionary_latest.csv` | 22, 24 — `Engagement` (old generated feature metadata) |
| `data/synthetic/eri_synthetic_clean_truth.csv` | 1 — `Engagement` (old feature header) |
| `data/synthetic/eri_synthetic_clean_truth_latest.csv` | 1 — `Engagement` (old feature header) |
| `data/synthetic/eri_synthetic_raw.csv` | 1 — `Engagement` (old feature header) |
| `data/synthetic/eri_synthetic_raw_latest.csv` | 1 — `Engagement` (old feature header) |
| `data/synthetic/ground_truth.json` | 41 — `Engagement` (old generated feature metadata) |
| `docker-compose.yml` | 30 — `period` (health-check option `start-period`) |
| `docs/01_project_context.md` | 50 — `period` (plain-language cross-sectional limitation) |
| `docs/03_sample_data.md` | 111 — `legacy`; 136 — `mei` |
| `docs/04_cleaning.md` | 5, 9, 125 — `legacy`; 19 — `mei` |
| `docs/DECISIONS.md` | 12 — `period` (plain-language scope) |
| `docs/data_card_sample.md` | 115 — `mei` |
| `docs/documents/supervisor_questions.md` | 8 — `period` (plain-language question) |
| `docs/envfin_audit.md` | 30 — `Engagement`; 46 — `states` |
| `docs/figures/INDEX.md` | 13 — `mei` |
| `docs/legacy/PROJECT_BRIEF_engagement.md` | 5 — `Engagement` |
| `docs/phase_reports/ef_00_repo_inspection.md` | 22, 105, 107, 109, 112–117 — historical `Engagement`; 111, 113–114, 131 — `period`; 112 — `HMM`, `GRU`, `LSTM`; 114 — `mei`; 116 — `states` |
| `docs/phase_reports/ef_01.md` | 17 — `mei`; 67 — `states` |
| `docs/phase_reports/ef_03.md` | 6, 13, 21 — historical `Engagement`; 21 — `HMM`, `period`, `legacy`; 34–36 — `mei` |
| `docs/phase_reports/ef_04.md` | 28, 43–45 — `mei` |
| `docs/phase_reports/ef_05.md` | 14, 61 — `legacy`; 31, 46, 48 — `mei` |
| `docs/questionnaire_envfin_v2.md` | 178 — `Engagement` (cross-component outcome-source note) |
| `frontend/package-lock.json` | 638, 3039 — `GRU`; 949 — `mei`; 4100 — `HMM` (dependency-lock content, to inspect before any change) |
| `frontend/src/pages/CausalAnalysisPage.tsx` | 200 — `Engagement` |
| `frontend/src/pages/DataCleaningPage.tsx` | 31 — `Engagement` |
| `frontend/src/pages/DocsVivaPage.tsx` | 30 — `Engagement` |
| `frontend/src/pages/FusionIntegrationPage.tsx` | 41 — `Engagement` |
| `frontend/src/pages/UploadPredictPage.tsx` | 51 — `Engagement` |
| `mei/__init__.py` | 1 — `mei` (package docstring) |
| `mei/cli.py` | 1, 169 — `mei` (CLI description/program name) |
| `src/causal/dag.py` | 7, 8, 39, 45, 77, 83, 85, 88, 98, 145, 153 — `Engagement` (old exposures and cross-component comments) |
| `src/causal/narrative.py` | 37 — `states` (verb in explanatory prose) |
| `src/cleaning/cleaner.py` | 148, 161, 313 — `Engagement` (legacy cleaner compatibility paths) |
| `src/cleaning/eda.py` | 155, 282 — `Engagement` (old EDA labels/columns) |
| `src/data_gen/generate.py` | 249–250, 255, 267, 276, 287, 313, 352, 398, 400, 472, 630, 632 — `Engagement` (old generator features and metadata) |
| `src/explain/narrative.py` | 33 — `Engagement` (old predictor description) |
| `src/inference/ingestion.py` | 63, 173 — `Engagement` (old feature alignment) |
| `src/preprocessing/pipeline.py` | 6, 36, 81 — `Engagement` (old feature contract) |
| `src/utils/diagnose.py` | 221, 380–381 — `Engagement` (old diagnostics and mediators) |
| `src/utils/test_scm_noise.py` | 117, 120, 130, 132, 135, 140, 163, 196 — `Engagement` (old synthetic SCM feature generation) |
| `tasks.ps1` | 58 — `legacy`; 78 — `mei` |
| `tests/test_api.py` | 231 — `Engagement`; 248 — `legacy` |
| `tests/test_data_gen.py` | 54 — `Engagement` |
| `tests/test_envfin_cleaner.py` | 233, 264 — `legacy`; 257, 272 — `mei` |
| `tests/test_envfin_retraining.py` | 14 — `legacy` |
| `tests/test_inference.py` | 101, 151 — `Engagement` |
| `tests/test_load_upload.py` | 27 — `legacy` |
| `tests/test_no_engagement_era.py` | 11 — `mei`; 26, 53 — `legacy`; 31–33 — `period`; 32 — `PeriodReshaper`; 34 — `HMM`; 36 — `GRU`, `LSTM` |
| `tests/test_smoke.py` | 147 — `Engagement` |

## Tracked path hits

- `data/legacy/` contains tracked interim, processed, and sample-upload files
  listed above, plus `.gitkeep` and `split_ids.json`.
- `docs/legacy/PROJECT_BRIEF_engagement.md` is tracked.
- `mei/` contains tracked `__init__.py` and `cli.py`.
- No tracked `src/mei/`, `src/efi/`, `pyproject.toml`, `scripts/tasks.ps1`,
  HMM module, `PeriodReshaper` implementation, or sequence-model-specific
  module was found in the initial tracked-path inventory.
- No matching engagement-questionnaire file or Antigravity prompt pack was
  tracked under `reference/` at inventory time.

## Allowed plain-language uses

- The `period` substring in `Dockerfile.api:42` and `docker-compose.yml:30` is
  part of the standard health-check option `start-period`, not a time-series
  feature.
- `period` in context, decisions, and supervisor questions refers to the
  ordinary time concept or the design's stated lack of a time dimension.
- `states` in the brief, audit notes, and phase reports is ordinary prose about
  concepts/values, not an active CLI command or module.
- Terms in `docs/phase_reports/` are historical evidence of prior migration
  steps; that directory is explicitly excluded from the later guard.

The other hits identify rename targets, old data/schema/code, deliberate
negative-test strings, or lockfile metadata. The separate Antigravity prompt
copies discovered only as untracked local files are not included in this
tracked-file inventory.
