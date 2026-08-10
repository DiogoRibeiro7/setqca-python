.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test cov check docs docs-build build clean

POETRY ?= poetry
RUN := $(POETRY) run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependency groups and the pre-commit hooks
	$(POETRY) install --with dev,docs
	$(RUN) pre-commit install

lint: ## Run the linter
	$(RUN) ruff check .

format: ## Apply formatting and autofixes
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

typecheck: ## Run mypy in strict mode
	$(RUN) mypy

test: ## Run the test suite
	$(RUN) pytest

cov: ## Run the test suite with a coverage report
	$(RUN) pytest --cov=setqca --cov-report=term-missing --cov-report=html

check: ## Run the full quality gate, as CI does
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy
	$(RUN) pytest --cov=setqca

docs: ## Serve the documentation locally with live reload
	$(RUN) mkdocs serve

docs-build: ## Build the documentation as CI does
	$(RUN) mkdocs build --strict

build: ## Build the sdist and wheel
	$(POETRY) build

clean: ## Remove build, test and cache artefacts
	rm -rf dist build site htmlcov .coverage .coverage.* coverage.xml
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
