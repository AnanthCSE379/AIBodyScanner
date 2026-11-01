.DEFAULT_GOAL := lint
.PHONY: install format lint dashboard

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

format:
	$(PYTHON) -m ruff format web tests

lint:
	$(PYTHON) -m ruff check web tests
	$(PYTHON) -m ruff format --check web tests

dashboard:
	$(PYTHON) -m uvicorn web.server:app --host 127.0.0.1 --port 8000
