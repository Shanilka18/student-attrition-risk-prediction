# Makefile for ERI Research Repository (SLIIT CDAP Project KND-CDAP-IT-05)
# Component: Environmental & Financial Factors — Study Conditions Index
# Cross-Platform / POSIX & Windows nmake/gnu-make compatibility
# Usage: make <target>  (e.g., make all, make data, make train, make test)

PYTHON = ./.venv/bin/python
PYTEST = ./.venv/bin/pytest

# Windows override detection
ifeq ($(OS),Windows_NT)
    PYTHON = ./.venv/Scripts/python.exe
    PYTEST = ./.venv/Scripts/pytest.exe
endif

.PHONY: setup data clean train leaderboard explain causal profiles api frontend test all help

help:
	@echo "Available targets:"
	@echo "  setup       - Create virtual environment and install pinned requirements"
	@echo "  data        - Generate synthetic validation dataset (N=2000, seed=42)"
	@echo "  clean       - Run data cleaning, EDA, and leakage-proof split"
	@echo "  train       - Train model zoo across 4 categories (Track C)"
	@echo "  leaderboard - Compute validation metrics, significance tests, calibration, and audit"
	@echo "  explain     - Compute SHAP values, waterfall plots, and fold stability diagnostics"
	@echo "  causal      - Estimate causal risk differences with sIPW, AIPW, and refutations"
	@echo "  profiles    - Show latent risk profile configuration"
	@echo "  api         - Start FastAPI backend microservice on port 8000"
	@echo "  frontend    - Start React dashboard development server on port 5173"
	@echo "  test        - Execute automated pytest test suite"
	@echo "  all         - Execute full pipeline end-to-end with seed 42"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.data_gen.generate --n 2000 --seed 42

clean:
	$(PYTHON) -m src.cleaning.cleaner
	$(PYTHON) -m src.cleaning.eda
	$(PYTHON) -m src.preprocessing.split

train:
	$(PYTHON) -m src.training.run_all --track C --models all --quick --no-tune

leaderboard:
	$(PYTHON) -c "from src.evaluation.evaluate_phase5 import run_phase5_evaluation; run_phase5_evaluation()"

explain:
	$(PYTHON) -c "from src.explain.run_phase6 import run_phase6_explainability; run_phase6_explainability()"

causal:
	$(PYTHON) -m src.causal.run_phase7

profiles:
	$(PYTHON) -m efi.cli profiles

api:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && npm run dev

test:
	$(PYTEST) tests/ -v

all: data clean train leaderboard explain causal
