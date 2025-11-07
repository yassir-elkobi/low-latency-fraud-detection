SHELL := /usr/bin/bash
PY := python3

ENV_FILE ?= .env

.PHONY: install train serve stream test lint fmt

install:
	$(PY) -m pip install --upgrade pip
	pip install -r requirements.txt

train:
	$(PY) scripts/train_baseline.py

serve:
	@echo "Implement FastAPI startup before using 'make serve' (uvicorn)."
	@echo "Example (once app factory exists): uvicorn app.main:create_app --factory --reload"

stream:
	$(PY) scripts/simulate_stream.py

test:
	pytest -q

lint:
	@echo "Add your linter (e.g., ruff/flake8) if desired"

fmt:
	@echo "Add your formatter (e.g., black) if desired"
