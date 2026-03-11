SHELL := /bin/sh

# ====== Config ======
UV ?= uv
PYTHON ?= python
MODULE ?= src

FUNCTIONS_DEFINITION ?= data/input/functions_definition.json
INPUT ?= data/input/function_calling_tests.json
OUTPUT ?= data/output/function_calling_results.json

EXCLUDES := .venv,__pycache__,.mypy_cache,.pytest_cache,build,dist,llm_sdk
MYPY_FLAGS := --warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs

# ====== UX helpers ======
OK    = \033[0;32m✓\033[0m
WARN  = \033[0;33m!\033[0m
INFO  = \033[0;34m>\033[0m
ERR   = \033[0;31m✗\033[0m

.PHONY: help install run debug lint lint-strict clean distclean test format

help:
	@printf "Targets:\n"
	@printf "  install      Install dependencies with uv sync\n"
	@printf "  run          Run project with default input/output paths\n"
	@printf "  debug        Run project with pdb\n"
	@printf "  lint         Run flake8 + mypy (subject flags)\n"
	@printf "  lint-strict  Run flake8 + mypy --strict\n"
	@printf "  test         Run pytest\n"
	@printf "  clean        Remove caches and generated artifacts\n"
	@printf "  distclean    clean + remove .venv\n"

install:
	@printf "$(INFO) Syncing dependencies with uv\n"
	@$(UV) sync || ( \
		printf "$(ERR) uv sync failed.\n"; \
		exit 1 )
	@printf "$(OK) Dependencies installed\n"

run: install
	@printf "$(INFO) Running project\n"
	@$(UV) run $(PYTHON) -m $(MODULE) \
		--functions_definition $(FUNCTIONS_DEFINITION) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug: install
	@printf "$(INFO) Debugging with pdb\n"
	@$(UV) run $(PYTHON) -m pdb -m $(MODULE) \
		--functions_definition $(FUNCTIONS_DEFINITION) \
		--input $(INPUT) \
		--output $(OUTPUT)

lint: install
	@printf "$(INFO) Running flake8\n"
	@$(UV) run flake8 src --exclude $(EXCLUDES) || ( \
		printf "$(ERR) flake8 failed.\n"; \
		exit 1 )
	@printf "$(OK) flake8 passed\n"
	@printf "$(INFO) Running mypy\n"
	@$(UV) run mypy --explicit-package-bases src $(MYPY_FLAGS) || ( \
		printf "$(ERR) mypy failed.\n"; \
		exit 1 )
	@printf "$(OK) mypy passed\n"
	@printf "$(OK) Lint OK\n"

lint-strict: install
	@printf "$(INFO) Running flake8\n"
	@$(UV) run flake8 src --exclude $(EXCLUDES) || ( \
		printf "$(ERR) flake8 failed.\n"; \
		exit 1 )
	@printf "$(INFO) Running mypy --strict\n"
	@$(UV) run mypy --explicit-package-bases src --strict || ( \
		printf "$(ERR) mypy --strict failed.\n"; \
		exit 1 )
	@printf "$(OK) Strict lint OK\n"

test: install
	@printf "$(INFO) Running pytest\n"
	@$(UV) run pytest || ( \
		printf "$(ERR) Tests failed.\n"; \
		exit 1 )
	@printf "$(OK) Tests passed\n"

clean:
	@printf "$(INFO) Cleaning caches and generated artifacts\n"
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .mypy_cache .pytest_cache .ruff_cache 2>/dev/null || true
	@rm -rf dist build *.egg-info 2>/dev/null || true
	@rm -f data/output/*.json 2>/dev/null || true
	@printf "$(OK) Clean done\n"

distclean: clean
	@printf "$(INFO) Removing virtual environment\n"
	@rm -rf .venv
	@printf "$(OK) distclean done\n"
