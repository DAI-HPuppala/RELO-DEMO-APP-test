# Makefile for Barcode Detector

# Variables
PYTHON := python3
VENV := barcode-env
VENV_BIN := $(VENV)/bin
PACKAGE := barcode_detector
SRC_DIR := src
TEST_DIR := tests

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

.PHONY: help install install-dev setup test test-unit test-integration test-performance \
        lint format type-check coverage clean benchmark examples docs build \
        install-deps venv activate

# Default target
help: ## Show this help message
	@echo "$(BLUE)Barcode Detector - Makefile Help$(RESET)"
	@echo "=================================="
	@echo ""
	@echo "$(GREEN)Setup Commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?## .*$$/ {printf "  $(YELLOW)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Usage Examples:$(RESET)"
	@echo "  make setup          # Complete setup including venv and dependencies"
	@echo "  make test           # Run all tests"
	@echo "  make lint           # Run linting and formatting checks"
	@echo "  make benchmark      # Run performance benchmarks"
	@echo "  make examples       # Run example scripts"

# Setup targets
venv: ## Create virtual environment
	@echo "$(BLUE)Creating virtual environment: $(VENV)$(RESET)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)Virtual environment created successfully$(RESET)"
	@echo "$(YELLOW)To activate: source $(VENV)/bin/activate$(RESET)"

install-deps: ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(RESET)"
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt
	@echo "$(GREEN)Dependencies installed successfully$(RESET)"

install: venv install-deps ## Install package in production mode
	@echo "$(BLUE)Installing package...$(RESET)"
	$(VENV_BIN)/pip install -e .
	@echo "$(GREEN)Package installed successfully$(RESET)"

install-dev: venv install-deps ## Install package in development mode
	@echo "$(BLUE)Installing package in development mode...$(RESET)"
	$(VENV_BIN)/pip install -e ".[dev]"
	@echo "$(GREEN)Development package installed successfully$(RESET)"

setup: install-dev ## Complete development setup
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Creating .env from template...$(RESET)"; \
		cp .env.example .env; \
	fi
	@echo "$(GREEN)Development setup complete!$(RESET)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(RESET)"
	@echo "  1. Activate virtual environment: source $(VENV)/bin/activate"
	@echo "  2. Run tests: make test"
	@echo "  3. Try examples: make examples"

# Testing targets
test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(RESET)"
	$(VENV_BIN)/pytest $(TEST_DIR) -v

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(VENV_BIN)/pytest $(TEST_DIR)/unit -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(VENV_BIN)/pytest $(TEST_DIR)/integration -v -m integration

test-performance: ## Run performance tests only
	@echo "$(BLUE)Running performance tests...$(RESET)"
	$(VENV_BIN)/pytest $(TEST_DIR)/performance -v -m performance --benchmark-only

coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(VENV_BIN)/pytest $(TEST_DIR) --cov=$(SRC_DIR)/$(PACKAGE) --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/$(RESET)"

# Code quality targets
lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(RESET)"
	$(VENV_BIN)/flake8 $(SRC_DIR) $(TEST_DIR) examples/
	$(VENV_BIN)/mypy $(SRC_DIR)
	@echo "$(GREEN)Linting checks passed$(RESET)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(RESET)"
	$(VENV_BIN)/black $(SRC_DIR) $(TEST_DIR) examples/
	@echo "$(GREEN)Code formatted successfully$(RESET)"

format-check: ## Check code formatting without modifying files
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	$(VENV_BIN)/black --check $(SRC_DIR) $(TEST_DIR) examples/

type-check: ## Run type checking
	@echo "$(BLUE)Running type checks...$(RESET)"
	$(VENV_BIN)/mypy $(SRC_DIR)

# Application targets
run: ## Run the application with default settings
	@echo "$(BLUE)Running barcode detector...$(RESET)"
	$(VENV_BIN)/python -m $(PACKAGE).main

run-camera: ## Run with camera input
	@echo "$(BLUE)Running barcode detector with camera...$(RESET)"
	$(VENV_BIN)/python -m $(PACKAGE).main --source camera

run-image: ## Run with image input (requires IMAGE_PATH)
	@echo "$(BLUE)Running barcode detector with images...$(RESET)"
	@if [ -z "$(IMAGE_PATH)" ]; then \
		echo "$(RED)Error: IMAGE_PATH not specified$(RESET)"; \
		echo "$(YELLOW)Usage: make run-image IMAGE_PATH=/path/to/images$(RESET)"; \
		exit 1; \
	fi
	$(VENV_BIN)/python -m $(PACKAGE).main --source image --path "$(IMAGE_PATH)"

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	$(VENV_BIN)/python -m $(PACKAGE).main --benchmark
	@echo "$(GREEN)Benchmark results saved to benchmark_report.txt$(RESET)"

validate-config: ## Validate configuration
	@echo "$(BLUE)Validating configuration...$(RESET)"
	$(VENV_BIN)/python -m $(PACKAGE).main --validate-config

# Example targets
examples: ## Run all example scripts
	@echo "$(BLUE)Running example scripts...$(RESET)"
	@echo "$(YELLOW)Running basic examples...$(RESET)"
	$(VENV_BIN)/python examples/basic_usage.py
	@echo ""
	@echo "$(YELLOW)Running advanced examples...$(RESET)"
	$(VENV_BIN)/python examples/advanced_usage.py

examples-basic: ## Run basic examples only
	@echo "$(BLUE)Running basic examples...$(RESET)"
	$(VENV_BIN)/python examples/basic_usage.py

examples-advanced: ## Run advanced examples only
	@echo "$(BLUE)Running advanced examples...$(RESET)"
	$(VENV_BIN)/python examples/advanced_usage.py

# Build and distribution targets
build: ## Build package for distribution
	@echo "$(BLUE)Building package...$(RESET)"
	$(VENV_BIN)/python -m build
	@echo "$(GREEN)Package built in dist/$(RESET)"

clean-build: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

# Documentation targets
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(RESET)"
	@echo "$(YELLOW)Documentation available in README.md and examples/$(RESET)"

# Development helpers
dev-shell: ## Start development shell with environment loaded
	@echo "$(BLUE)Starting development shell...$(RESET)"
	@echo "$(YELLOW)Virtual environment: $(VENV)$(RESET)"
	@echo "$(YELLOW)To exit: type 'exit' or press Ctrl+D$(RESET)"
	$(VENV_BIN)/python

jupyter: ## Start Jupyter notebook for development
	@echo "$(BLUE)Starting Jupyter notebook...$(RESET)"
	$(VENV_BIN)/pip install jupyter
	$(VENV_BIN)/jupyter notebook

# Maintenance targets
clean: clean-build ## Clean all generated files
	@echo "$(BLUE)Cleaning generated files...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf logs/*.log 2>/dev/null || true
	rm -rf benchmark_*.txt benchmark_*.json 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete$(RESET)"

clean-all: clean ## Clean everything including virtual environment
	@echo "$(BLUE)Cleaning everything including virtual environment...$(RESET)"
	rm -rf $(VENV)/
	@echo "$(GREEN)Complete cleanup finished$(RESET)"

update-deps: ## Update dependencies to latest versions
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	$(VENV_BIN)/pip install --upgrade -r requirements.txt
	@echo "$(GREEN)Dependencies updated$(RESET)"

# Utility targets
check-system: ## Check system requirements
	@echo "$(BLUE)Checking system requirements...$(RESET)"
	@echo "$(YELLOW)Python version:$(RESET)"
	$(PYTHON) --version
	@echo "$(YELLOW)OpenCV version:$(RESET)"
	$(PYTHON) -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "$(RED)OpenCV not installed$(RESET)"
	@echo "$(YELLOW)Camera availability:$(RESET)"
	$(PYTHON) -c "import cv2; cap=cv2.VideoCapture(0); print('Available' if cap.isOpened() else 'Not available'); cap.release()" 2>/dev/null || echo "$(RED)Cannot check camera$(RESET)"

list-configs: ## List current configuration
	@echo "$(BLUE)Current configuration:$(RESET)"
	@if [ -f .env ]; then \
		echo "$(YELLOW)Configuration from .env:$(RESET)"; \
		cat .env | grep -v "^#" | grep -v "^$$"; \
	else \
		echo "$(YELLOW)No .env file found, using defaults$(RESET)"; \
	fi

# Docker targets (if needed)
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t barcode-detector .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(RESET)"
	docker run -it --rm barcode-detector

# Help for specific commands
help-test: ## Show testing help
	@echo "$(BLUE)Testing Commands:$(RESET)"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make coverage       - Run tests with coverage report"
	@echo ""
	@echo "$(YELLOW)Test Options:$(RESET)"
	@echo "  Test specific file: $(VENV_BIN)/pytest tests/unit/test_config.py -v"
	@echo "  Test with markers:  $(VENV_BIN)/pytest -m integration"
	@echo "  Verbose output:     $(VENV_BIN)/pytest -v -s"

help-run: ## Show run command help
	@echo "$(BLUE)Run Commands:$(RESET)"
	@echo "  make run                                    - Default settings"
	@echo "  make run-camera                             - Camera input"
	@echo "  make run-image IMAGE_PATH=/path/to/images   - Image input"
	@echo ""
	@echo "$(YELLOW)Command Line Options:$(RESET)"
	@echo "  --mode custom|library|hybrid    - Detection mode"
	@echo "  --config /path/to/.env          - Custom config file"
	@echo "  --verbose                       - Enable debug logging"
	@echo "  --benchmark                     - Run performance test"

# Version info
version: ## Show version information
	@echo "$(BLUE)Version Information:$(RESET)"
	@$(VENV_BIN)/python -c "from src.$(PACKAGE) import __version__; print(f'Package version: {__version__}')" 2>/dev/null || echo "Package not installed"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Make version: $$(make --version | head -1)"