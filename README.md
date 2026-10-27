# Academic Intelligence Framework

**Project**: KND_CDAP_IT_05 — Multi-Factor Student Attrition Risk Prediction  
**Student**: V. S. H. Gunasekara · IT23164680 · SLIIT Centre of Excellence for AI (CoEAI)  
**Supervisors**: Ms. Chathurika Koswatte · Ms. Anuththara Ekanayake

> ⚠️ **SYNTHETIC DATA**: All results are from a synthetic cohort. Pending IRB clearance for real institutional data.

---

## 5-Minute Quick Start

### Prerequisites
| Requirement | Version | Check |
|-------------|---------|-------|
| Python | ≥ 3.12 | `python --version` |
| Node.js | ≥ 18 | `node --version` |
| npm | ≥ 9 | `npm --version` |
| RAM | ≥ 8 GB | — |
| Disk | ≥ 3 GB free | — |

### Step 1 — Clone & Install Python Dependencies (60 sec)
```bash
git clone <repo-url>
cd "research project"
pip install -r requirements.txt
```

### Step 2 — Install Frontend Dependencies (30 sec)
```bash
cd frontend
npm install
cd ..
```

### Step 3 — Run the Full Pipeline (≈ 5 min)
```bash
# Windows PowerShell / CMD:
python run.py all

# macOS / Linux:
make all
```
This runs: **setup → data generation → preprocessing → training → evaluation → explainability**  
All artifacts are saved under `artifacts/`.

### Step 4 — Start the Backend (in a separate terminal)
```bash
python run.py serve
# Backend: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### Step 5 — Start the Frontend (in another terminal)
```bash
cd frontend
npm run dev
# Frontend: http://localhost:5173
```

### Step 6 — Run Tests
```bash
python -m pytest backend/tests -v
# Expected: 209 passed, 0 failed
```

### Step 7 — Upload Your First Prediction
1. Download the template: `GET http://localhost:8000/predict/template`
2. Open the portal at `http://localhost:5173` → **Predict** page
3. Drag-and-drop the CSV → view per-student ARI scores and explanations

---

## Individual Pipeline Targets

| Command | What it does |
|---------|-------------|
| `python run.py setup` | Verify directories, install frontend deps |
| `python run.py data` | Generate 5,000-student synthetic cohort |
| `python run.py preprocess` | Split, engineer features, build tensors |
| `python run.py train` | Tune + train all models, calibrate, threshold |
| `python run.py evaluate` | Test-set evaluation, leaderboard, significance tests |
| `python run.py explain` | SHAP, ARI, risk bands, fusion contract |
| `python run.py serve` | Start FastAPI backend on port 8000 |
| `python run.py frontend` | Start Vite dev server on port 5173 |
| `python run.py all` | Runs setup → explain sequentially |
| `python run.py train --budget full` | Full Optuna budget (100 trials/model, slow) |
| `python run.py train --force` | Force retrain even if artifacts exist |

---

## Project Structure

```
research project/
├── run.py                  # One-command pipeline orchestrator
├── Makefile                # POSIX/CI make targets
├── make.bat                # Windows CMD wrapper
├── requirements.txt        # Pinned Python dependencies
├── config.yaml             # All tuneable parameters
│
├── backend/
│   ├── src/
│   │   ├── config.py       # Centralised config
│   │   ├── data/           # Generator, task sampler, audit
│   │   ├── preprocessing/  # Splits, features, tensors
│   │   ├── models/         # 28 models, BaseAcademicModel
│   │   ├── training/       # Optuna engine, calibration
│   │   ├── evaluation/     # Metrics, leaderboard, bootstrap
│   │   ├── explain/        # SHAP, ARI, fusion contract
│   │   └── api/            # FastAPI, validator, schemas, service
│   └── tests/              # 209 tests
│
├── frontend/               # React 18 + Vite + Tailwind (10 pages)
│
├── artifacts/
│   ├── models/             # Trained model files + summaries
│   ├── leaderboard/        # JSON/CSV/MD leaderboard
│   ├── metrics/            # Test provenance, bootstrap CIs
│   ├── explanations/       # SHAP, faithfulness, stability, fusion
│   └── plots/              # evaluation/ + explainability/ PNGs
│
├── data/
│   ├── synthetic/          # Generated CSVs
│   ├── processed/          # Feature matrices, tensors
│   ├── splits/             # Train/val/test student lists, CV folds
│   └── schema/             # schema.json, upload_template.csv
│
└── docs/                   # All phase documentation + viva pack
```

---

## Key Results

> All numbers from `artifacts/leaderboard/leaderboard.json` and model summary files.  
> **Source data: SYNTHETIC.**

| Model | ROC-AUC | PR-AUC | F2 | Brier | Lead Time |
|-------|---------|--------|-----|-------|-----------|
| **GRU** 👑 | **0.83** | **0.22** | **0.35** | **0.053** | **2.5 sem** |
| CatBoost | 0.81 | 0.20 | 0.32 | 0.055 | 2.0 sem |
| GPA Threshold | 0.51 | 0.07 | 0.18 | 0.058 | 0.5 sem |

**Research answer**: POSITIVE — GRU outperforms CatBoost (p=0.03 AUC, p=0.04 PR-AUC).  
See `docs/viva/CHEATSHEET.md` for all headline numbers.

---

## Configuration

Edit `config.yaml` to change:
- `global_seed` — reproducibility seed (default: 42)
- `training.budget` — `quick` (10 trials) / `standard` (30) / `full` (100)
- `training.n_seeds` — seeds for stability testing (default: [42, 123, 999])
- `leaderboard.ranking_weights` — composite metric weights
- `leaderboard.eligibility.max_inference_time_sec` — latency gate (default: 2.0)
- `data.n_students` — cohort size for generation (default: 5000)

---

## Documentation

| Doc | Purpose |
|-----|---------|
| `docs/00_PROJECT_OVERVIEW.md` | Architecture, phases, NFRs |
| `docs/01_DATA_GENERATION.md` | Synthetic generator design |
| `docs/02_PREPROCESSING.md` | Feature engineering, splits |
| `docs/03_MODEL_ZOO.md` | All 28 models, interfaces |
| `docs/04_TRAINING_AND_VALIDATION.md` | Optuna, calibration, thresholds |
| `docs/05_EVALUATION.md` | Metrics, leaderboard, significance |
| `docs/06_EXPLAINABILITY_AND_RISK_INDEX.md` | SHAP, ARI, fusion contract |
| `docs/07_API.md` | Full API specification |
| `docs/08_FRONTEND_GUIDE.md` | React portal user guide |
| `docs/DECISION_LOG.md` | 21 key decisions with rationale |
| `docs/viva/CHEATSHEET.md` | All headline numbers (viva use) |
| `docs/HANDOVER.md` | How to plug in real data |
| `docs/FINAL_CHECKLIST.md` | All acceptance checks: PASS/FAIL |
| `docs/AI_USE_DISCLOSURE.md` | Appendix 3 — AI usage disclosure |

---

## API Endpoints (Summary)

Base URL: `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/meta` | GET | Champion info, SYNTHETIC/REAL mode |
| `/leaderboard` | GET | All models ranked |
| `/models/{name}` | GET | Model detail + training history |
| `/research-answer` | GET | Research question adjudication |
| `/data/summary` | GET | Dataset audit stats |
| `/explain/global` | GET | Global SHAP feature importance |
| `/explain/student/{id}` | GET | Per-student ARI + explanation |
| `/predict/template` | GET | Download upload template CSV |
| `/predict/student` | POST | JSON: one student prediction |
| `/predict/upload` | POST | CSV/XLSX: batch prediction |
| `/predict/download/{token}` | GET | Export batch results (CSV/JSON) |

---

## Licence & Ethics

- Academic research only (SLIIT FYR). Not for commercial deployment without IRB clearance.
- All uploaded student data is processed in-memory and never persisted to disk.
- File size limit: 10 MB. Student IDs are never logged.
- See `docs/AI_USE_DISCLOSURE.md` for full AI tool usage transparency.
