# Multimodal Student Engagement Intelligence (MEI)

**Project ID**: `KND_CDAP_IT_05`  
**Lead Component**: Component 2 — Student Engagement Intelligence  
**Target Metric**: Multimodal Engagement Index (MEI), $0\text{–}100$ ($100 = \text{highest risk}$)

---

## 1. Overview
MEI is a machine learning research system developed as part of a four-component early-warning student attrition prevention framework. It models multidimensional engagement trajectories across 4 semester periods ($P_1\text{–}P_4$) using 7 core indicators, uncovers latent behavioral states via Clustering and Hidden Markov Models (HMM), benchmarks candidate classifiers on a leak-free leaderboard, and produces a calibrated attrition risk index ($0\text{–}100$) explained via Tree/Kernel SHAP attributions.

> **Data Truth**: All behavioral inputs originate from a structured, closed-ended student questionnaire (Google Forms style). Campus LMS server telemetry is outdated and not used. See [PROJECT_BRIEF.md](PROJECT_BRIEF.md) for full specifications.

---

## 2. Project Structure
```text
engagement-intelligence/
├── AGENTS.md                  # Autonomous agent operating rules (G1–G13)
├── PROJECT_BRIEF.md           # Authoritative permanent project brief
├── README.md                  # Project overview and quickstart
├── pyproject.toml             # Build system, Ruff, Black, and Pytest configuration
├── requirements.txt           # Pinned dependencies
├── Makefile                   # GNU Make targets
├── .gitignore                 # Version control exclusion rules
├── config/
│   └── project.yaml           # Master experiment & hardware parameters
├── data/
│   ├── raw/                   # Unprocessed questionnaire files
│   ├── interim/               # Intermediate cleaned stages
│   ├── processed/             # Modeling-ready feature sets
│   ├── samples/               # Synthetic Google Forms exports
│   ├── real/                  # Real survey exports (gitignored)
│   └── templates/             # Blank survey input templates
├── src/mei/
│   ├── __init__.py
│   ├── cli.py                 # Typer CLI entrypoint
│   ├── config.py              # Pathlib constants & config loader
│   ├── io/                    # Ingestion & file handling
│   ├── cleaning/              # Screening & validation
│   ├── features/              # Feature engineering
│   ├── states/                # Clustering & Gaussian HMM
│   ├── models/                # Model training routines
│   ├── evaluation/            # Leaderboard metrics & overfit checks
│   ├── explain/               # SHAP explainability
│   ├── registry/              # Model persistence & active run tracking
│   ├── api/                   # FastAPI serving endpoints
│   └── utils/                 # Logging & timer utilities
├── artifacts/
│   ├── runs/                  # Run artifacts (manifest, metrics, models)
│   └── active_run.txt         # Active champion run pointer
├── frontend/                  # Research web dashboard
├── docs/
│   ├── 00_environment.md      # Environment setup & dependency specs
│   ├── 01_project_context.md  # 4-component context & survey design
│   ├── DECISIONS.md           # Architectural decision log
│   ├── figures/               # 300-dpi publication figures & INDEX.md
│   └── phase_reports/         # Per-phase verification reports
├── tests/                     # Unit and integration test suite
└── scripts/
    └── tasks.ps1              # Windows PowerShell task runner
```

---

## 3. Quickstart & Installation

### Prerequisites
- Python 3.11.9 (64-bit)
- Windows / Linux / macOS (8 GB RAM, no GPU required)

### Setup Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

---

## 4. CLI Execution
The system provides a command-line interface via Typer:
```powershell
python -m mei.cli --help
```

### Available Pipeline Commands:
- `setup`: Prepare directories, environment validation, and baseline configs.
- `make-sample-data`: Generate synthetic Google Forms questionnaire responses.
- `clean`: Screen careless responders and clean responses.
- `features`: Extract period-level, slope, drop, and drift features.
- `states`: Fit clustering and Gaussian HMM to decode behavioral trajectories.
- `train-classic`: Train linear, shallow, tree, and ensemble classifiers (Track A).
- `train-deep`: Train sequence and tabular deep architectures.
- `train-regression`: Train continuous attrition intent models (Track B).
- `finalize`: Score leaderboard, compute overfit penalties, and pick champion.
- `explain`: Generate global and local SHAP explanations.
- `serve`: Launch FastAPI scoring service.
- `retrain`: Retrain champion model on new survey uploads.
- `docs`: Validate documentation and generate publication figures.
- `all`: Execute end-to-end pipeline.

Alternatively, use the PowerShell task runner or Makefile:
```powershell
.\scripts\tasks.ps1 test
.\scripts\tasks.ps1 lint
```

---

## 5. Testing & Code Quality
All commits must satisfy three quality gates:
```powershell
# Run smoke tests and leak assertions
pytest

# Style and syntax check
ruff check src tests

# Code formatting check
black --check src tests
```
