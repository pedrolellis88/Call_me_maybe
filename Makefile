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

MOULINETTE_DIR ?= moulinette
MOULINETTE_SET ?= private
MOULINETTE_PROJECT_VENV ?= /goinfre/$(USER)/call_me_maybe_venv
MOULINETTE_VENV ?= /goinfre/$(USER)/call_me_maybe_moulinette_venv
MOULINETTE_OUTPUT ?= data/output/function_calling_results.json
MOULINETTE_GRADE_CMD ?= grade_student_answers

OK   = \033[0;32m✓\033[0m
INFO = \033[0;34m>\033[0m
ERR  = \033[0;31m✗\033[0m

.PHONY: help check-uv install run debug lint lint-strict test clean distclean \
	check-moulinette-dir moulinette-prepare moulinette-run moulinette-grade moulinette

help:
	@printf "Targets:\n"
	@printf "  install           Install dependencies with uv sync\n"
	@printf "  run               Run project\n"
	@printf "  debug             Run project with pdb\n"
	@printf "  lint              Run flake8 . + mypy . excluding llm_sdk and virtualenvs\n"
	@printf "  lint-strict       Run flake8 . + mypy . --strict excluding llm_sdk and virtualenvs\n"
	@printf "  test              Run pytest\n"
	@printf "  moulinette        Full moulinette flow: prepare + run + grade\n"
	@printf "  moulinette-prepare Prepare moulinette private set\n"
	@printf "  moulinette-run    Run project against moulinette inputs using subject schema\n"
	@printf "  moulinette-grade  Grade generated output with moulinette\n"
	@printf "  clean             Remove caches and generated files\n"
	@printf "  distclean         clean + remove virtual environment\n"

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

check-moulinette-dir:
	@if [ ! -d "$(MOULINETTE_DIR)" ]; then \
		printf "$(ERR) Moulinette directory not found: $(MOULINETTE_DIR)\n"; \
		printf "    Add the moulinette folder to the project root before running this target.\n"; \
		exit 1; \
	fi
	@if [ ! -f "$(MOULINETTE_DIR)/pyproject.toml" ]; then \
		printf "$(ERR) Invalid moulinette directory: $(MOULINETTE_DIR)\n"; \
		printf "    Missing pyproject.toml inside the moulinette folder.\n"; \
		exit 1; \
	fi

moulinette-prepare: check-uv check-moulinette-dir
	@printf "$(INFO) Preparing moulinette exercise set ($(MOULINETTE_SET))\n"
	@mkdir -p /goinfre/$(USER)/.hf
	@mkdir -p $(MOULINETTE_VENV)
	@cd $(MOULINETTE_DIR) && \
	HF_HOME=/goinfre/$(USER)/.hf \
	HUGGINGFACE_HUB_CACHE=/goinfre/$(USER)/.hf/hub \
	TRANSFORMERS_CACHE=/goinfre/$(USER)/.hf/transformers \
	XDG_CACHE_HOME=/goinfre/$(USER)/.hf \
	UV_PROJECT_ENVIRONMENT=$(MOULINETTE_VENV) \
	UV_LINK_MODE=copy \
	$(UV) sync && \
	HF_HOME=/goinfre/$(USER)/.hf \
	HUGGINGFACE_HUB_CACHE=/goinfre/$(USER)/.hf/hub \
	TRANSFORMERS_CACHE=/goinfre/$(USER)/.hf/transformers \
	XDG_CACHE_HOME=/goinfre/$(USER)/.hf \
	UV_PROJECT_ENVIRONMENT=$(MOULINETTE_VENV) \
	UV_LINK_MODE=copy \
	$(UV) run $(PYTHON) -m moulinette prepare_exercises --set $(MOULINETTE_SET)
	@printf "$(OK) Moulinette inputs prepared\n"

moulinette-run: check-uv check-moulinette-dir
	@printf "$(INFO) Running project against moulinette inputs using subject schema\n"
	@mkdir -p /goinfre/$(USER)/.hf
	@mkdir -p $(MOULINETTE_PROJECT_VENV)
	@mkdir -p data/output
	@HF_HOME=/goinfre/$(USER)/.hf \
	HUGGINGFACE_HUB_CACHE=/goinfre/$(USER)/.hf/hub \
	TRANSFORMERS_CACHE=/goinfre/$(USER)/.hf/transformers \
	XDG_CACHE_HOME=/goinfre/$(USER)/.hf \
	CALL_ME_MAYBE_OUTPUT_SCHEMA=subject \
	UV_PROJECT_ENVIRONMENT=$(MOULINETTE_PROJECT_VENV) \
	UV_LINK_MODE=copy \
	$(UV) run $(PYTHON) -m $(MODULE) \
		--functions_definition $(MOULINETTE_DIR)/data/input/functions_definition.json \
		--input $(MOULINETTE_DIR)/data/input/function_calling_tests.json \
		--output $(MOULINETTE_OUTPUT)
	@printf "$(OK) Project output generated for moulinette\n"

moulinette-grade: check-uv check-moulinette-dir
	@printf "$(INFO) Grading output with moulinette using command: $(MOULINETTE_GRADE_CMD)\n"
	@mkdir -p /goinfre/$(USER)/.hf
	@mkdir -p $(MOULINETTE_VENV)
	@cd $(MOULINETTE_DIR) && \
	HF_HOME=/goinfre/$(USER)/.hf \
	HUGGINGFACE_HUB_CACHE=/goinfre/$(USER)/.hf/hub \
	TRANSFORMERS_CACHE=/goinfre/$(USER)/.hf/transformers \
	XDG_CACHE_HOME=/goinfre/$(USER)/.hf \
	UV_PROJECT_ENVIRONMENT=$(MOULINETTE_VENV) \
	UV_LINK_MODE=copy \
	$(UV) run $(PYTHON) -m moulinette $(MOULINETTE_GRADE_CMD) \
		--set $(MOULINETTE_SET) \
		--student_answer_path ../$(MOULINETTE_OUTPUT)

moulinette: moulinette-prepare moulinette-run moulinette-grade
	@printf "$(OK) Full moulinette flow completed\n"

clean:
	@printf "$(INFO) Cleaning caches and generated artifacts\n"
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .mypy_cache .pytest_cache .ruff_cache dist build *.egg-info 2>/dev/null || true
	@rm -f data/output/*.json 2>/dev/null || true
	@printf "$(OK) Clean done\n"

distclean: clean
	@printf "$(INFO) Removing virtual environment at $(VENV_DIR)\n"
	@rm -rf $(VENV_DIR)
	@rm -rf $(MOULINETTE_VENV)
	@rm -rf .venv
	@printf "$(OK) distclean done\n"
