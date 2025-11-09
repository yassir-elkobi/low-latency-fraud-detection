SHELL := /usr/bin/bash
PY := python3

ENV_FILE ?= .env

.PHONY: install train serve stream test lint fmt

install:
	$(PY) -m pip install --upgrade pip
	pip install -r requirements.txt

train:
	$(PY) -m scripts.train_baseline

serve:
	uvicorn app.main:create_app --factory --host 0.0.0.0 --port $${PORT:-8000}

stream:
	$(PY) -m scripts.simulate_stream

test:
	pytest -q

lint:
	@echo "Add your linter (e.g., ruff/flake8) if desired"

fmt:
	@echo "Add your formatter (e.g., black) if desired"

.PHONY: site
site:
	$(PY) -m scripts.build_static_site
