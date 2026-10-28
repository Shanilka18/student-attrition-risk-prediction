.PHONY: help setup make-sample-data clean features states train-classic train-deep train-regression finalize explain serve retrain docs all test lint format

PYTHON = ./.venv/Scripts/python.exe

help:
	@echo "Multimodal Student Engagement Intelligence (MEI) - Make Targets"
	@echo "Available commands:"
	@echo "  make setup"
	@echo "  make make-sample-data"
	@echo "  make clean"
	@echo "  make features"
	@echo "  make states"
	@echo "  make train-classic"
	@echo "  make train-deep"
	@echo "  make train-regression"
	@echo "  make finalize"
	@echo "  make explain"
	@echo "  make serve"
	@echo "  make retrain"
	@echo "  make docs"
	@echo "  make all"
	@echo "  make test"
	@echo "  make lint"
	@echo "  make format"

setup:
	$(PYTHON) -m mei.cli setup

make-sample-data:
	$(PYTHON) -m mei.cli make-sample-data

clean:
	$(PYTHON) -m mei.cli clean

features:
	$(PYTHON) -m mei.cli features

states:
	$(PYTHON) -m mei.cli states

train-classic:
	$(PYTHON) -m mei.cli train-classic

train-deep:
	$(PYTHON) -m mei.cli train-deep

train-regression:
	$(PYTHON) -m mei.cli train-regression

finalize:
	$(PYTHON) -m mei.cli finalize

explain:
	$(PYTHON) -m mei.cli explain

serve:
	$(PYTHON) -m mei.cli serve

retrain:
	$(PYTHON) -m mei.cli retrain

docs:
	$(PYTHON) -m mei.cli docs

all:
	$(PYTHON) -m mei.cli all

test:
	$(PYTHON) -m pytest tests/

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m black --check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m black src tests
