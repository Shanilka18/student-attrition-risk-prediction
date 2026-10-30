# Academic Intelligence Framework Makefile
# Project: Multi-Factor Student Attrition Risk Prediction
# Author: V. S. H. Gunasekara (IT23164680)

.PHONY: setup data preprocess train evaluate explain serve frontend test all

PYTHON ?= python

setup:
	$(PYTHON) run.py setup

test:
	$(PYTHON) run.py test

data:
	$(PYTHON) run.py data

preprocess:
	$(PYTHON) run.py preprocess

train:
	$(PYTHON) run.py train

evaluate:
	$(PYTHON) run.py evaluate

explain:
	$(PYTHON) run.py explain

serve:
	$(PYTHON) run.py serve

frontend:
	$(PYTHON) run.py frontend

all:
	$(PYTHON) run.py all
