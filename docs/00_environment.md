# 00 — Environment, Tooling, and Execution Infrastructure

## What it is
The execution foundation and reproducible runtime environment for the **Multimodal Student Engagement Intelligence (MEI)** system. It pins Python 3.11.9, virtual environment management, all scientific machine learning and deep learning dependencies, quality linters (`ruff`, `black`), testing frameworks (`pytest`), and a central unified CLI (`python -m mei.cli`).

## Why we do it
Reproducibility and auditability are non-negotiable standards in academic and applied machine learning research (Rule G4). By strictly pinning dependency versions, time-boxing heavy tasks for consumer laptop hardware (8 GB RAM, no GPU on Windows), and enforcing strict formatting and smoke testing from day zero, we prevent environment drift, platform-dependent runtime failures, and silent numerical discrepancies.

## How it works step by step
1. **Python Runtime**: Python 3.11.9 64-bit installed in the user application space.
2. **Virtual Isolation**: Isolated virtual environment created at `.venv` with `pip 24.0` and `mei.pth` linking `src/` to site-packages for editable package resolution.
3. **Dependency Pinning**: All required libraries for data manipulation, tree boosting, deep learning, state sequence analysis (HMM), explainability (SHAP), and API serving are pinned in `requirements.txt`.
4. **Deep Learning Fallback**: As defined in Global Rule G10 and STEP 3, PyTorch CPU (`torch==2.14.1`) is configured for CPU-only execution without requiring CUDA or heavy GPU hardware.
5. **Quality Gates**:
   - `ruff check src tests` verifies clean imports, syntax, and style rules.
   - `black --check src tests` guarantees PEP 8 line formatting (88 chars).
   - `pytest` executes test suites, including CLI stub validation, config validation, and target leakage guards.
6. **Unified CLI**: Built with `typer`, exposing commands: `setup`, `make-sample-data`, `clean`, `features`, `states`, `train-classic`, `train-deep`, `train-regression`, `finalize`, `explain`, `serve`, `retrain`, `docs`, and `all`.

## Inputs and Outputs
- **Inputs**:
  - `requirements.txt`: Master pinned dependency list.
  - `pyproject.toml`: Build configuration, tool settings for ruff, black, and pytest.
  - `config/project.yaml`: Master experiment parameters, random seeds, and hardware constraints.
- **Outputs**:
  - Activated `.venv/` virtual environment.
  - Validated test and lint results (`4 passed`).
  - Executable entrypoint `python -m mei.cli`.

## Pinned Environment Specifications
| Package | Version | Purpose |
|---|---|---|
| Python | 3.11.9 | Target base runtime |
| numpy | 1.26.4 | Numerical routines & vector algebra |
| pandas | 2.2.3 | Tabular data manipulation |
| scipy | 1.13.1 | Scientific and statistical distributions |
| scikit-learn | 1.5.2 | Pipelines, transformers, linear & shallow models |
| statsmodels | 0.15.0 | Econometric and regression diagnostics |
| xgboost | 2.1.3 | Gradient boosted decision trees |
| lightgbm | 4.7.0 | Histogram-based gradient boosting |
| catboost | 1.2.7 | Symmetric tree gradient boosting |
| torch | 2.14.1 | CPU deep learning neural architectures |
| hmmlearn | 0.3.3 | Gaussian Hidden Markov Models for state decoding |
| shap | 0.43.0 | Tree and Kernel SHAP attribution values |
| optuna | 4.1.0 | Hyperparameter tuning and search |
| imbalanced-learn | 0.14.2 | Resampling and class balancing |
| matplotlib | 3.9.4 | 300-dpi publication figure generation |
| seaborn | 0.13.2 | Statistical distribution plotting |
| pyyaml | 6.0.2 | YAML configuration parsing |
| pydantic | 2.10.4 | Data validation and schema enforcement |
| joblib | 1.6.0 | Pipeline serialization and caching |
| openpyxl | 3.1.5 | Excel spreadsheet ingestion |
| xlsxwriter | 3.3.0 | Excel report generation |
| rapidfuzz | 3.14.6 | Fast string matching for survey columns |
| typer | 0.27.3 | CLI command building |
| rich | 15.0.0 | Terminal formatting and progress display |
| fastapi | 0.115.6 | Production inference REST API |
| uvicorn | 0.34.0 | ASGI web server |
| python-multipart | 0.0.20 | File upload parsing for FastAPI |
| pypdf | 6.21.2 | Pandoc-free PDF document parsing |
| python-docx | 1.2.0 | Pandoc-free DOCX document parsing |
| pytest | 8.3.4 | Unit and integration test runner |
| pytest-cov | 7.1.0 | Test coverage measurement |
| ruff | 0.17.2 | Fast Python linter |
| black | 26.10.1 | Deterministic code formatter |

## Formulas and Hardware Parameters
- **Global Random Seed**: Fixed at $S = 42$ across all random number generators.
- **Hardware Limits**: Target execution on standard consumer hardware: Windows OS, 8 GB RAM, 0 GPU accelerators.
- **Maximum Execution Time Budget**: $T_{\text{max}} \le 60\text{ minutes}$ for complete training runs.

## Decisions and Rejected Alternatives
- **Decision**: Used PyTorch CPU (`torch==2.14.1`) as the deep learning framework instead of TensorFlow.
  - *Reason*: TensorFlow CPU Windows distribution requires large monolithic binary packages (~300+ MB) that exhibited network stalling and compatibility issues, whereas PyTorch CPU installed cleanly, is lightweight on 8 GB RAM, and provides identical sequence and tabular neural modeling capabilities matching sibling project components.
- **Decision**: Added `scripts/tasks.ps1` in addition to `Makefile`.
  - *Reason*: Native Windows developer environments often lack GNU Make; PowerShell provides identical target execution natively without third-party installations.

## Common Mistakes
- Activating a global system Python rather than `.venv\Scripts\Activate.ps1`.
- Omitting `MPLCONFIGDIR` on Windows, causing Matplotlib to emit permission warnings when attempting to create directories in restricted user profiles.
- Unpinning package versions in `requirements.txt`, leading to unexpected API breakage across machines.

## How to Explain This in a Viva
"We established a strictly deterministic software environment configured for consumer-grade hardware. By pinning all 32 core scientific and machine learning libraries under Python 3.11 and enforcing automated linting (`ruff`), formatting (`black`), and leak-detection tests (`pytest`), we ensure that our research experiments, Hidden Markov Models, and gradient boosting benchmarks run identically across any evaluator's system without dependency drift."

## Explain Like I Am 5
"Before building a big Lego castle, you make sure you have the exact right bucket of Lego bricks, all sorted by color, with no missing pieces. That way, if your friend wants to build the exact same castle at their house, they can follow the exact instructions and get the exact same castle."
