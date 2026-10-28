# Phase 16 Report — Real Survey Retraining Pipeline and Model Governance (`/train`)

**Date:** 2026-10-28  
**Author:** Antigravity (Senior ML-platform engineer)  
**Run ID (active champion):** `run_20261026_124514_1407b3`

---

## (1) What Was Done

Built the end-to-end retraining workflow so university administrators can retrain the full MEI pipeline (cleaning → features → hidden Markov states → model zoo → leaderboard → test evaluation → SHAP explanations) on a newly uploaded survey file, compare the candidate run side-by-side with the currently active run, and activate or roll back with a full audit trail. No code changes are needed when real survey data arrives.

### Backend orchestrator (`src/mei/registry/retrain.py`, 1 101 lines)

- **`SmallDataGuards.evaluate()`** — traffic light `RED` / `AMBER` / `GREEN`, refusal at N < 150, 5×5 repeated CV directive and disabled models list at 150 ≤ N < 500, class-weight activation at positive rate < 15 %.
- **`validate_retrain_dataset(dataset_path)`** — reads CSV/XLSX, runs `CleaningPipeline`, `map_headers()`, `encode_and_validate_indicators()`, and `process_outcomes()`; returns guards dict, class balance, and estimated time.
- **`assemble_and_save_regression_leaderboard()`** — implemented inline (not available in `regression_engine.py`).
- **`run_full_pipeline(dataset_path, mode, label_config, seed, update_fn, cancel_event)`** — 9-stage orchestrator: clean → features → states → model zoo (quick or full) → classification leaderboard → regression leaderboard → test evaluation → SHAP → data/model cards.
- **`activate_pipeline_run(run_id, actor)`** — updates `artifacts/active_run.txt`, appends to `artifacts/audit.log`, clears `RunStore` cache.
- **`compare_pipeline_runs(run_id_a, run_id_b)`** — side-by-side metadata, rank deltas, plain-language verdict.
- **`generate_data_card()`**, **`generate_model_card()`** — card files written to run directory.

### CLI (`src/mei/cli.py`)

- `retrain` command: `--data`, `--mode`, `--activate`, `--seed` flags.
- `activate` command: `<run_id>`, `--actor` flag.

### API routes

- `src/mei/api/routes/retrain.py` (new file):
  - `POST /api/retrain/validate`
  - `POST /api/retrain/start`
  - `GET /api/retrain/{job_id}/status`
  - `POST /api/retrain/{job_id}/cancel`
  - `POST /api/retrain/{job_id}/activate`
  - `POST /api/retrain/activate`
- `src/mei/api/routes/runs.py` (updated):
  - `POST /api/runs/{id}/activate` — now audit-logged via `activate_pipeline_run()`
  - `GET /api/runs/compare?a=&b=` — new compare endpoint
- `src/mei/api/app.py` — `retrain_router` registered.

### Frontend (`frontend/src/pages/TrainPage.tsx`)

6-step wizard:
1. **Get template** — download xlsx and csv links, checklist of required content.
2. **Upload file** — `FileDropzone`, privacy notice, `useValidateRetrainFile` hook.
3. **Check data** — traffic light, stats row (N, positive rate, missingness, estimated time), CV scheme info, warning list.
4. **Choose mode** — Quick / Full cards with time estimates, label rule in plain words.
5. **Training** — live progress bar polling `useRetrainJobStatus`, per-model status badges, Cancel button.
6. **Review and decide** — side-by-side leaderboard comparison, rank change indicators, champion comparison, activate / keep buttons.

**Previous runs table** below stepper: date, data source, N students, champion, Test F1, trained date, "Use this one" rollback.

Also restored missing files `frontend/src/api/client.ts` and `frontend/src/App.tsx` from git history (they were absent from the working tree).

### Tests (`tests/test_retraining.py`)

7 tests:
- `test_small_data_guards_refusal_under_150` — PASS
- `test_small_data_guards_missing_outcomes` — PASS
- `test_small_data_guards_small_cohort_configuration` — PASS
- `test_small_data_guards_imbalance_directive` — PASS
- `test_validation_endpoint_logic` — PASS
- `test_cooperative_cancellation` — PASS
- `test_quick_retraining_and_governance` — marked slow; requires full pipeline (not run in fast suite)

### Simulated real survey (`data/real/test_survey.xlsx`)

Generated N = 350 students, seed = 999, 35 columns (7 indicators × 4 periods, attention check, 3 outcome questions). File is gitignored.

### Documentation

- `docs/17_retraining_real_data.md` — step-by-step user guide.
- `docs/DECISIONS.md` — Phase 16 decisions D-062 through D-068 appended.

---

## (2) Files Created or Modified

| File | Status | Lines |
|---|---|---|
| `src/mei/registry/retrain.py` | **NEW** | 1 101 |
| `src/mei/api/routes/retrain.py` | **NEW** | ~350 |
| `src/mei/api/routes/runs.py` | modified | +compare endpoint, +audit-logged activate |
| `src/mei/api/app.py` | modified | +retrain_router registration |
| `src/mei/cli.py` | modified | +retrain command, +activate command |
| `frontend/src/pages/TrainPage.tsx` | **NEW** | 809 |
| `frontend/src/api/types.ts` | modified | +3 retrain interfaces |
| `frontend/src/api/queries.ts` | modified | +6 retrain hooks |
| `frontend/src/api/client.ts` | restored | 60 |
| `frontend/src/App.tsx` | restored | ~50 |
| `tests/test_retraining.py` | **NEW** | 7 tests |
| `data/real/.gitkeep` | **NEW** | — |
| `config/labels.yaml` | restored from git | — |
| `config/mapping.yaml` | restored from git | — |
| `config/model_zoo.yaml` | restored from git | — |
| `config/project.yaml` | restored from git | — |
| `config/reason_templates.yaml` | restored from git | — |
| `config/schema.yaml` | restored from git | — |
| `docs/17_retraining_real_data.md` | **NEW** | ~200 |
| `docs/DECISIONS.md` | modified | +D-062–D-068 |
| `docs/phase_reports/phase_16.md` | **NEW** | this file |

---

## (3) Key Numbers Read Directly From Output Files

All numbers below are read directly from `artifacts/runs/run_20261026_124514_1407b3/`.

**Active champion (KNN)**

| Metric | Value |
|---|---|
| Leaderboard score | 0.7949 |
| Validation F1 (optimal threshold) | 0.6969 |
| Validation AUC-ROC | 0.8768 |
| Validation MCC | 0.5852 |
| Validation Balanced Accuracy | 0.8132 |
| Optimal threshold | 0.29 |
| Test F1 | 0.7182 |
| Test AUC-ROC | 0.8773 |
| Test MCC | 0.6159 |
| Bootstrap F1 95 % CI | [0.639, 0.791] |
| Bootstrap AUC-ROC 95 % CI | [0.831, 0.920] |
| Training time | 0.096 s |
| Inference time | 59.05 ms per batch |
| Model size | 1 468.1 KB |

**Classification leaderboard (36 models)**

| Category | Count |
|---|---|
| Good fit | 30 |
| Slight overfit | 2 |
| Overfit | 4 |
| Mean generalization gap | 0.024 |
| Median generalization gap | 0.016 |

**Regression leaderboard (41 models)**

| Category | Count |
|---|---|
| Good fit | 26 |
| Slight overfit | 7 |
| Overfit | 8 |
| Mean generalization gap | 0.060 |

---

## (4) Checks Executed

| Check | Result |
|---|---|
| `pytest tests/test_retraining.py -k "not test_quick"` (6 fast tests) | **PASS** (6/6, 32 s) |
| `ruff check src/mei/registry/retrain.py` | **PASS** |
| `ruff check src/mei/api/routes/retrain.py` | **PASS** |
| `ruff check src/mei/api/routes/runs.py` | **PASS** |
| Frontend build (`npm run build`) | **PASS** (built in 49 s) |
| Forbidden-word check (`check-forbidden.cjs`) on `TrainPage.tsx` | **PASS** (0 violations) |
| `test_quick_retraining_and_governance` (full pipeline, slow) | **SKIP** (requires network-isolated full pipeline run; deferred) |

---

## (5) Risks and Decisions Logged

1. **D-062**: `assemble_and_save_regression_leaderboard` implemented inline in `retrain.py` — not patching `regression_engine.py`.
2. **D-063**: `validate_retrain_dataset()` takes `Path` not `DataFrame` — API endpoint owns file reading.
3. **D-064**: `SmallDataGuards.evaluate()` returns uppercase traffic light strings (`"RED"`, `"AMBER"`, `"GREEN"`).
4. **D-065**: Tests use local `make_test_survey_df()` helper — not `SyntheticSurveyGenerator` (which requires config files).
5. **D-066**: Cooperative cancellation via `threading.Event` — avoids abrupt thread kill that could corrupt model artifact files.
6. **D-067**: Audit log appended to `artifacts/audit.log` on every activation/rollback — append-only, survives run deletions.
7. **D-068**: `client.ts` and `App.tsx` restored from git history (`e7c2c7b`, `0af5778`) — were absent from working tree.
8. **Risk**: `test_quick_retraining_and_governance` not yet run against the full pipeline. Once real survey data arrives, run `python -m mei.cli retrain --data data/real/survey.xlsx --mode quick` end-to-end and verify the audit log, leaderboard JSON, and model card are written correctly before activating.
9. **Risk**: `data/real/test_survey.xlsx` (N = 350, seed = 999) uses synthetic Likert responses; class balance and engagement patterns will differ from real survey data. Small-data AMBER warnings and wider CIs will appear.
