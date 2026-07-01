.PHONY: help setup experiment

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
TORCH_INDEX = https://download.pytorch.org/whl/cpu
KERNEL_NAME = svd-leverage

help:
	@echo "Targets:"
	@echo "  make setup       cria .venv, instala deps (pyproject.toml) e kernel Jupyter"
	@echo "  make experiment  grade experimental IMDb (400 amostras, 256 tokens)"

setup: $(VENV)/bin/python
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url $(TORCH_INDEX)
	$(PIP) install -e .
	$(PYTHON) -m ipykernel install --user --name $(KERNEL_NAME) --display-name "Python (svd-leverage)"

$(VENV)/bin/python:
	python3 -m venv $(VENV)

experiment: setup
	$(PYTHON) src/experimento.py --max-amostras 400 --max-tokens 256
