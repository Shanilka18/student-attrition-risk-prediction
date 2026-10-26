.PHONY: help install data train evaluate explain serve test lint clean all demo audit

PYTHON   = python
UVICORN  = uvicorn
PYTEST   = pytest

# ── Colour helpers ───────────────────────────────────────────────────────────
BOLD  := \033[1m
RESET := \033[0m
GREEN := \033[32m
CYAN  := \033[36m

help:
	@echo ""
	@echo "$(BOLD)Psychosocial Intelligence Framework (PRI Component)$(RESET)"
	@echo "$(CYAN)──────────────────────────────────────────────────────────$(RESET)"
	@echo "  $(BOLD)make install$(RESET)       Install pinned Python dependencies"
	@echo "  $(BOLD)make data$(RESET)          Generate / process survey datasets"
	@echo "  $(BOLD)make train$(RESET)         Train all models (Optuna tuning, full 15 trials)"
	@echo "  $(BOLD)make evaluate$(RESET)      Evaluate models, compute CV metrics & leaderboards"
	@echo "  $(BOLD)make explain$(RESET)       Generate global & local SHAP explanations and plots"
	@echo "  $(BOLD)make serve$(RESET)         Start FastAPI server (port 8000)"
	@echo "  $(BOLD)make test$(RESET)          Run full automated test suite (pytest)"
	@echo "  $(BOLD)make lint$(RESET)          Run flake8 linting"
	@echo "  $(BOLD)make audit$(RESET)         Run Phase 10 reproducibility + upload flow audits"
	@echo "  $(BOLD)make all$(RESET)           Full pipeline: install → data → train → evaluate → explain"
	@echo "  $(BOLD)make demo$(RESET)          Demo pipeline: quick train + serve (ports 8000 & 5173)"
	@echo "  $(BOLD)make clean$(RESET)         Remove __pycache__, .pytest_cache, coverage artefacts"
	@echo ""

# ── Core pipeline steps ──────────────────────────────────────────────────────

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.data.make_dataset

train:
	$(PYTHON) -m src.models.train --track both --trials 15

evaluate:
	$(PYTHON) -m src.evaluation.run_evaluation

explain:
	$(PYTHON) -m src.explain.shap_explainer

serve:
	$(UVICORN) src.api.main:app --host 127.0.0.1 --port 8000 --reload

test:
	$(PYTEST) tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	$(PYTHON) -m flake8 src tests --max-line-length=120 || true

# ── Phase 10 audit targets ───────────────────────────────────────────────────

audit:
	@echo "$(BOLD)[Phase 10] Running reproducibility audit...$(RESET)"
	$(PYTHON) -m scripts.reproducibility_check
	@echo ""
	@echo "$(BOLD)[Phase 10] Running upload flow integration tests...$(RESET)"
	@echo "NOTE: Requires API running on port 8000. Start with: make serve"
	$(PYTHON) -m scripts.upload_flow_test

# ── Composite pipeline targets ───────────────────────────────────────────────

all: install data train evaluate explain
	@echo ""
	@echo "$(GREEN)$(BOLD)Full pipeline complete!$(RESET)"
	@echo "  Leaderboards : artifacts/leaderboards/"
	@echo "  Metrics      : artifacts/metrics/"
	@echo "  SHAP Plots   : artifacts/shap/"
	@echo "  Models       : artifacts/models/"
	@echo "  Start server : make serve"

demo:
	@echo "$(BOLD)[Demo] Quick-mode pipeline (fast smoke test)...$(RESET)"
	$(PYTHON) -m src.data.make_dataset
	$(PYTHON) -m src.models.train --track both --quick
	$(PYTHON) -m src.evaluation.run_evaluation
	$(PYTHON) -m src.explain.shap_explainer
	@echo ""
	@echo "$(BOLD)[Demo] Starting API server + Frontend...$(RESET)"
	@echo "  Backend  -> http://127.0.0.1:8000/docs"
	@echo "  Frontend -> http://localhost:5173  (run: cd frontend && npm run dev)"
	$(UVICORN) src.api.main:app --host 127.0.0.1 --port 8000

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('htmlcov') if p.is_dir()]"
	@echo "$(GREEN)Clean complete.$(RESET)"
