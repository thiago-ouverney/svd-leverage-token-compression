.PHONY: help install install-experiment test experiment

PYTHON ?= python3
PIP ?= pip

help:
	@echo "Targets:"
	@echo "  make install             deps minimas (numpy, matplotlib)"
	@echo "  make install-experiment  deps completas para IMDb"
	@echo "  make test                testes unitarios"
	@echo "  make experiment          grade experimental IMDb"

install:
	$(PIP) install -r requirements.txt

install-experiment: install
	$(PIP) install datasets transformers torch

test: install
	$(PYTHON) testes.py

experiment: install-experiment
	$(PYTHON) experimento.py --max-amostras 400 --max-tokens 256
