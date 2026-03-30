SHELL := /bin/sh

UV ?= uv
PYTHON ?= python
MODULE ?= src
VENV_DIR ?= /goinfre/$(USER)/call_me_maybe_venv

FUNCTIONS_DEFINITION ?= data/input/functions_definition.json
INPUT ?= data/input/function_calling_tests.json
OUTPUT ?= data/output/function_calling_results.json

FLAKE8_EXCLUDE := llm_sdk,.venv,__pycache__,.mypy_cache,.pytest_cache
MYPY_EXCLUDE := llm_sdk|\.venv|__pycache__|\.mypy_cache|\.pytest_cache

MYPY_FLAGS := --warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs

OK   = \033[0;32m✓\033[0m
INFO = \033[0;34m>\033[0m
ERR  = \033[0;31m✗\033[0m

.PHONY: help check-uv install run debug lint lint-strict test clean distclean

help:
	@printf "Targets:\n"
	@printf "  install      Install dependencies with uv sync\n"
	@printf "  run          Run project\n"
	@printf "  debug        Run project with pdb\n"
	@printf "  lint         Run flake8 . + mypy . excluding llm_sdk and virtualenvs\n"
	@printf "  lint-strict  Run flake8 . + mypy . --strict excluding llm_sdk and virtualenvs\n"
	@printf "  test         Run pytest\n"
	@printf "  clean        Remove caches and generated files\n"
	@printf "  distclean    clean + remove project .venv\n"

check-uv:
	@command -v $(UV) >/dev/null 2>&1 || { \
		printf "$(ERR) uv is not installed or not in PATH.\n"; \
		exit 1; \
	}

install: check-uv
	@printf "$(INFO) Syncing dependencies with uv into $(VENV_DIR)\n"
	@mkdir -p $(VENV_DIR)
	@mkdir -p /goinfre/$(USER)/.hf
	@HF_HOME=/goinfre/$(USER)/.hf \
	HUGGINGFACE_HUB_CACHE=/goinfre/$(USER)/.hf/hub \
	TRANSFORMERS_CACHE=/goinfre/$(USER)/.hf/transformers \
	XDG_CACHE_HOME=/goinfre/$(USER)/.hf \
	UV_PROJECT_ENVIRONMENT=$(VENV_DIR) \
	UV_LINK_MODE=copy \
	$(UV) sync
	@printf "$(OK) Dependencies installed\n"

run: check-uv
	@printf "$(INFO) Running project\n"
	@mkdir -p /goinfre/$(USER)/.hf
	@HF_HOME=/goinfre/$(USER)/.hf \
	HUGGINGFACE_HUB_CACHE=/goinfre/$(USER)/.hf/hub \
	TRANSFORMERS_CACHE=/goinfre/$(USER)/.hf/transformers \
	XDG_CACHE_HOME=/goinfre/$(USER)/.hf \
	UV_PROJECT_ENVIRONMENT=$(VENV_DIR) \
	UV_LINK_MODE=copy \
	$(UV) run $(PYTHON) -m $(MODULE) \
		--functions_definition $(FUNCTIONS_DEFINITION) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug: check-uv
	@printf "$(INFO) Debugging with pdb\n"
	@mkdir -p data/output
	@mkdir -p /goinfre/$(USER)/.hf
	@HF_HOME=/goinfre/$(USER)/.hf \
	HUGGINGFACE_HUB_CACHE=/goinfre/$(USER)/.hf/hub \
	TRANSFORMERS_CACHE=/goinfre/$(USER)/.hf/transformers \
	XDG_CACHE_HOME=/goinfre/$(USER)/.hf \
	UV_PROJECT_ENVIRONMENT=$(VENV_DIR) \
	UV_LINK_MODE=copy \
	$(UV) run $(PYTHON) -m pdb -m $(MODULE) \
		--functions_definition $(FUNCTIONS_DEFINITION) \
		--input $(INPUT) \
		--output $(OUTPUT)

lint: check-uv
	@printf "$(INFO) Running flake8\n"
	@UV_PROJECT_ENVIRONMENT=$(VENV_DIR) UV_LINK_MODE=copy \
	$(UV) run flake8 . --exclude=$(FLAKE8_EXCLUDE)
	@printf "$(INFO) Running mypy\n"
	@UV_PROJECT_ENVIRONMENT=$(VENV_DIR) UV_LINK_MODE=copy \
	$(UV) run mypy . --exclude '$(MYPY_EXCLUDE)' $(MYPY_FLAGS)
	@printf "$(OK) Lint OK\n"

lint-strict: check-uv
	@printf "$(INFO) Running flake8\n"
	@UV_PROJECT_ENVIRONMENT=$(VENV_DIR) UV_LINK_MODE=copy \
	$(UV) run flake8 . --exclude=$(FLAKE8_EXCLUDE)
	@printf "$(INFO) Running mypy --strict\n"
	@UV_PROJECT_ENVIRONMENT=$(VENV_DIR) UV_LINK_MODE=copy \
	$(UV) run mypy . --exclude '$(MYPY_EXCLUDE)' --strict
	@printf "$(OK) Strict lint OK\n"

test: check-uv
	@printf "$(INFO) Running pytest\n"
	@UV_PROJECT_ENVIRONMENT=$(VENV_DIR) UV_LINK_MODE=copy \
	$(UV) run pytest
	@printf "$(OK) Tests passed\n"

clean:
	@printf "$(INFO) Cleaning caches and generated artifacts\n"
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .mypy_cache .pytest_cache .ruff_cache dist build *.egg-info 2>/dev/null || true
	@rm -f data/output/*.json 2>/dev/null || true
	@printf "$(OK) Clean done\n"

distclean: clean
	@printf "$(INFO) Removing project-local virtual environment only\n"
	@rm -rf .venv
	@printf "$(OK) distclean done\n"
