# Phase 0 Report — Project Skeleton, Environment, and Permanent Rules

## 1. What Was Done
- **Reference Material Analysis**: Read and analyzed `reference/TAF_KND_CDAP_IT_05.pdf` and `reference/vihu 1 (2).docx` using `pypdf` and `python-docx` (pandoc-free). Identified the 4-component student attrition risk project structure (Academic, Engagement, Psychosocial, Environmental). Acknowledged and adopted the critical pivot: LMS/Moodle/Zoom/mouse server telemetry described in the proposal is outdated; the sole operational truth is a closed-ended 4-period, 7-indicator student questionnaire.
- **Project Structure**: Created the full directory hierarchy using `pathlib` relative paths: `config/`, `data/` (raw, interim, processed, samples, real, templates), `src/mei/` (cli, config, io, cleaning, features, states, models, evaluation, explain, registry, api, utils), `artifacts/runs/`, `frontend/`, `docs/` (figures, phase_reports), `tests/`, and `scripts/`.
- **Environment & Dependencies**: Installed and pinned all 32 required scientific, ML, deep learning, explainability, serving, and quality dependencies in `requirements.txt` under Python 3.11.9 (`.venv`).
- **Deep Learning Framework Decision**: Per STEP 3 instructions, recorded fallback to PyTorch CPU (`torch==2.14.1`) due to large binary transport stalling with `tensorflow-intel` on Windows without GPU, ensuring 8 GB RAM compliance.
- **CLI & Task Runners**: Implemented 14 stub commands using `typer` in `src/mei/cli.py` (`setup`, `make-sample-data`, `clean`, `features`, `states`, `train-classic`, `train-deep`, `train-regression`, `finalize`, `explain`, `serve`, `retrain`, `docs`, `all`). Configured matching `Makefile` targets and a native Windows PowerShell runner `scripts/tasks.ps1`.
- **Permanent Rule Books**: Wrote verbatim `PROJECT_BRIEF.md` containing all 10 sections and `AGENTS.md` (15 lines, under 25-line cap) enforcing Global Rules G1–G13.
- **Documentation**: Authored `docs/00_environment.md`, `docs/01_project_context.md`, `docs/DECISIONS.md`, `docs/figures/INDEX.md`, and `README.md`.
- **Git Initialization & Quality Assurance**: Initialized Git repository, verified `.gitignore` exclusions, and passed all quality gates (`pytest`, `ruff`, `black`).

---

## 2. Files Created and Changed
- `PROJECT_BRIEF.md`: Verbatim 10-section permanent rule book.
- `AGENTS.md`: Autonomous agent operating constraints (15 lines).
- `README.md`: Project overview, structure, installation, and CLI reference.
- `pyproject.toml`: Project build metadata, ruff, black, and pytest configuration.
- `requirements.txt`: 32 pinned library versions.
- `Makefile`: GNU Make pipeline targets.
- `scripts/tasks.ps1`: Windows PowerShell task runner.
- `.gitignore`: Ignore rules for venv, real data, run weights, caches.
- `config/project.yaml`: Master experiment parameters, periods, indicators, and hardware limits.
- `src/mei/__init__.py`: Package initialization.
- `src/mei/cli.py`: Typer CLI entrypoint with 14 stub commands.
- `src/mei/config.py`: Pathlib paths and config loader.
- `src/mei/io/__init__.py`: Subpackage init.
- `src/mei/cleaning/__init__.py`: Subpackage init.
- `src/mei/features/__init__.py`: Subpackage init.
- `src/mei/states/__init__.py`: Subpackage init.
- `src/mei/models/__init__.py`: Subpackage init.
- `src/mei/evaluation/__init__.py`: Subpackage init.
- `src/mei/explain/__init__.py`: Subpackage init.
- `src/mei/registry/__init__.py`: Subpackage init.
- `src/mei/api/__init__.py`: Subpackage init.
- `src/mei/utils/__init__.py`: Subpackage init.
- `artifacts/active_run.txt`: Active run pointer file.
- `docs/00_environment.md`: Environment specifications and viva explanation.
- `docs/01_project_context.md`: Context, 4 components, and survey design.
- `docs/DECISIONS.md`: Decisions D-001 through D-004.
- `docs/figures/INDEX.md`: Style guide, color palette, and figure index.
- `docs/phase_reports/phase_00.md`: Phase 0 verification report.
- `tests/__init__.py`: Test package init.
- `tests/test_smoke.py`: 4 smoke tests (CLI help, stub execution, config load, leakage guard).

---

## 3. Key Numbers Read Directly from Files
- **Python version**: `3.11.9` (verified via `sys.version`)
- **Pinned libraries in requirements.txt**: `32` libraries
- **CLI commands defined**: `14` commands (`setup`, `make-sample-data`, `clean`, `features`, `states`, `train-classic`, `train-deep`, `train-regression`, `finalize`, `explain`, `serve`, `retrain`, `docs`, `all`)
- **TAF document length**: `10` pages (`reference/TAF_KND_CDAP_IT_05.pdf`)
- **Proposal document word count**: `42,447` characters (`reference/vihu 1 (2).docx`)
- **Unit test duration**: `1.62s` (`4 passed in 1.62s`)
- **Black formatting files checked**: `15` files left unchanged

---

## 4. Checks Run with PASS/FAIL Status
| Check | Command | Status | Details |
|---|---|---|---|
| CLI Help | `python -m mei.cli --help` | **PASS** | Exit code 0, all 14 subcommands listed |
| CLI Stub Output | `python -m mei.cli <command>` | **PASS** | All 14 stubs return "not implemented yet" |
| Smoke Tests | `pytest tests/` | **PASS** | 4 passed in 1.62s (CLI, stubs, config, leakage rule) |
| Linter | `ruff check src tests` | **PASS** | All checks passed cleanly |
| Formatter | `black --check src tests` | **PASS** | 15 files checked, all left unchanged |
| Leakage Prevention | `test_no_target_leakage_rule` | **PASS** | Asserted indicators $\cap$ outcome items $= \emptyset$ |
| BRIEF Integrity | Section comparison | **PASS** | All 10 sections present without truncation |
| AGENTS Line Limit | Line count check | **PASS** | 15 lines (limit $\le 25$) |

---

## 5. Risks and Architectural Decisions Logged
- **Decision D-001**: Adopted closed-ended questionnaire schema over outdated LMS/Zoom telemetry.
- **Decision D-002**: Selected PyTorch CPU (`torch==2.14.1`) as the deep learning backend on Windows 8 GB RAM without GPU.
- **Decision D-003**: Implemented `scripts/tasks.ps1` for native Windows execution without external make toolchains.
- **Decision D-004**: Automated target leakage prevention asserting $A_1\text{–}A_3$ cannot intersect with indicator features.
- **Risk**: Memory constraints during heavy model training; mitigated by time-boxing budgets ($T_{\text{max}} \le 60\text{ min}$) and lightweight CPU architectures.
