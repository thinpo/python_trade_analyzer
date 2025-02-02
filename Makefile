.PHONY: venv clean install update-deps test lint format help run

PYTHON := python3
VENV_NAME := venv
VENV_BIN := $(VENV_NAME)/bin
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip

help:
	@echo "Available commands:"
	@echo "  make venv          - Create virtual environment and install dependencies"
	@echo "  make install       - Install dependencies"
	@echo "  make update-deps   - Update dependencies to latest versions"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters (mypy, black --check, isort --check)"
	@echo "  make format        - Format code (black, isort)"
	@echo "  make clean         - Remove virtual environment and cache files"
	@echo "  make run          - Run the trade analyzer (use ARGS='...' to pass arguments)"

$(VENV_NAME):
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV_NAME)
	@$(VENV_PIP) install --upgrade pip setuptools wheel
	@echo "Installing dependencies..."
	@$(VENV_PIP) install -r requirements.txt
	@$(VENV_PIP) install -e .

venv: $(VENV_NAME)

install: venv

update-deps: venv
	@echo "Updating dependencies..."
	@$(VENV_PYTHON) scripts/update_dependencies.py

test: venv
	@echo "Running tests..."
	@$(VENV_BIN)/pytest

lint: venv
	@echo "Running linters..."
	@$(VENV_BIN)/mypy src tests
	@$(VENV_BIN)/black --check src tests
	@$(VENV_BIN)/isort --check src tests

format: venv
	@echo "Formatting code..."
	@$(VENV_BIN)/black src tests
	@$(VENV_BIN)/isort src tests

run: venv
	@echo "Running trade analyzer..."
	@$(VENV_PYTHON) -m trade_analyzer.cli $(ARGS)

clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV_NAME)
	@rm -rf build dist *.egg-info
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type f -name ".coverage" -delete
	@find . -type f -name "coverage.xml" -delete 