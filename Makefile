.PHONY: install test lint format clean help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*'  | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", \1, \2}'

install:  ## Install dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:  ## Run tests
	pytest tests/ -v

lint:  ## Run linters
	flake8 .
	black --check .
	isort --check-only .

format:  ## Format code
	black .
	isort .

clean:  ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov