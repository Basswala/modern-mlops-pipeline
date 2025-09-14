.PHONY: help install test lint format clean build run docker-build docker-run pipeline api docs

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt
	pre-commit install

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=src --cov-report=html --cov-report=term-missing

lint: ## Run linting
	flake8 src/ tests/
	black --check src/ tests/

format: ## Format code
	black src/ tests/
	isort src/ tests/

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/

build: clean ## Build the package
	python -m build

run: ## Run the pipeline
	dvc repro

api: ## Start the API server
	python -m src.api

docker-build: ## Build Docker image
	docker build -t diabetes-ml-pipeline .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --env-file .env diabetes-ml-pipeline

docker-compose-up: ## Start with Docker Compose
	docker-compose up -d

docker-compose-down: ## Stop Docker Compose
	docker-compose down

pipeline: ## Run the complete ML pipeline
	dvc repro

setup-env: ## Setup environment file
	cp .env.example .env
	@echo "Please edit .env file with your credentials"

setup-dev: install setup-env ## Setup development environment
	@echo "Development environment setup complete!"

ci: lint test ## Run CI checks locally

docs: ## Generate documentation
	@echo "API docs available at http://localhost:8000/docs when server is running"

# Development workflow
dev-setup: setup-dev ## Complete development setup
	@echo "Ready for development! Run 'make api' to start the server."

# Production deployment
deploy: docker-build ## Deploy to production
	@echo "Deployment would happen here"
	# Add your deployment commands

# Data operations
data-pull: ## Pull data using DVC
	dvc pull

data-push: ## Push data using DVC
	dvc push

# Model operations
model-train: ## Train the model
	python src/train.py

model-evaluate: ## Evaluate the model
	python src/evaluate.py

# Utility commands
logs: ## Show application logs
	docker-compose logs -f ml-pipeline

status: ## Show system status
	@echo "=== Git Status ==="
	git status
	@echo "\n=== DVC Status ==="
	dvc status
	@echo "\n=== Docker Status ==="
	docker-compose ps
