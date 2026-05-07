# jacopy — development convenience targets.
#
# Quick start:
#   make setup      # one-shot: create .venv, install editable + all deps,
#                   # register Jupyter kernel.
#   make test       # run the full test suite inside .venv.
#   make notebooks  # smoke-execute every tutorial notebook.
#   make clean      # remove .venv and caches.

PYTHON ?= /opt/homebrew/opt/python@3.14/bin/python3.14
VENV   := .venv
VBIN   := $(VENV)/bin

.PHONY: setup test notebooks clean kernel deps

# One-shot dev environment.
setup: $(VENV)/.installed kernel
	@echo "✓ Setup complete. Activate with:  source $(VBIN)/activate"
	@echo "  Or open a notebook and pick the 'Python (jacopy .venv)' kernel."

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(VBIN)/pip install --upgrade pip

$(VENV)/.installed: $(VENV)/bin/python pyproject.toml
	$(VBIN)/pip install -e ".[dev,parallel]"
	@touch $@

deps: $(VENV)/.installed

kernel: $(VENV)/.installed
	$(VBIN)/python -m ipykernel install --user --name=jacopy \
		--display-name="Python (jacopy .venv)"

test: $(VENV)/.installed
	$(VBIN)/python -m pytest tests/ -q --ignore=tests/test_docs

notebooks: $(VENV)/.installed
	$(VBIN)/python -m pytest tests/test_docs -q

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__ .mypy_cache .ruff_cache
