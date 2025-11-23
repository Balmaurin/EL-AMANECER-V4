.PHONY: help install test lint format docker-up migrate clean

# Colors for output
BLUE := \033[36m
NC := \033[0m

help:  ## Show this help message
	@echo "$(BLUE)EL-AMANECERV3 - Enterprise AI System$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(BLUE)%-25s$(NC) %s\n", $$1, $$2}'

# =====================================
# INSTALLATION
# =====================================

install:  ## Install all dependencies
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	pip install -r requirements.txt
	pip install -r config/requirements-dev.txt
	@echo "$(BLUE)Installing Frontend dependencies...$(NC)"
	cd Frontend && npm install
	@echo "✅ Installation complete"

install-dev:  ## Install development dependencies
	pip install -r config/requirements-dev.txt
	pip install -e .
	pre-commit install

# =====================================
# TESTING
# =====================================

test:  ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	pytest tests/ -v --cov --cov-report=html --cov-report=term

test-unit:  ## Run unit tests only
	pytest tests/ -v -m "not integration and not e2e"

test-integration:  ## Run integration tests
	pytest tests/integration/ -v

test-e2e:  ## Run end-to-end tests
	pytest tests/e2e/ -v

test-watch:  ## Run tests in watch mode
	pytest-watch tests/

coverage:  ## Generate coverage report
	pytest --cov --cov-report=html --cov-report=term
	@echo "Open htmlcov/index.html to view coverage report"

# =====================================
# CODE QUALITY
# =====================================

lint:  ## Run all linters
	@echo "$(BLUE)Running linters...$(NC)"
	ruff check .
	mypy backend/ sheily_core/
	cd Frontend && npm run lint

lint-fix:  ## Fix linting issues
	ruff check --fix .
	cd Frontend && npm run lint -- --fix

format:  ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	black backend/ sheily_core/ scripts/ tools/
	ruff check --fix .
	cd Frontend && npm run format

type-check:  ## Run type checking
	mypy backend/ sheily_core/ --strict

# =====================================
# DOCKER & SERVICES
# =====================================

docker-build:  ## Build Docker images
	docker-compose build

docker-up:  ## Start all services with Docker
	@echo "$(BLUE)Starting services...$(NC)"
	docker-compose up -d
	@echo "✅ Services started:"
	@echo "   - Backend:  http://localhost:8000"
	@echo "   - Frontend: http://localhost:3000"
	@echo "   - Postgres: localhost:5432"
	@echo "   - Redis:    localhost:6379"

docker-down:  ## Stop all Docker services
	docker-compose down

docker-logs:  ## Show Docker logs
	docker-compose logs -f

docker-clean:  ## Clean Docker resources
	docker-compose down -v --remove-orphans
	docker system prune -f

# =====================================
# DATABASE
# =====================================

db-init:  ## Initialize database
	python backend/init_database.py

db-migrate:  ## Run database migrations
	cd backend && alembic upgrade head

db-rollback:  ## Rollback last migration
	cd backend && alembic downgrade -1

db-reset:  ## Reset database (DANGER!)
	@echo "⚠️  This will delete all data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -f backend/*.db; \
		python backend/init_database.py; \
		echo "✅ Database reset complete"; \
	fi

# =====================================
# DEVELOPMENT
# =====================================

dev-backend:  ## Run backend in development mode
	cd backend && uvicorn main_api:app --reload --host 0.0.0.0 --port 8000

dev-frontend:  ## Run frontend in development mode
	cd Frontend && npm run dev

dev-all:  ## Run both backend and frontend in dev mode
	@echo "$(BLUE)Starting development servers...$(NC)"
	make dev-backend & make dev-frontend

# =====================================
# TRAINING
# =====================================

train:  ## Start training pipeline
	python -m scripts.training.train_real_neural_network

train-gpu:  ## Train with GPU
	python -m scripts.training.gpu.real_transformers_training

train-distributed:  ## Distributed training
	python -m packages.training_system.src.distributed.launch

# =====================================
# DEPLOYMENT
# =====================================

deploy-staging:  ## Deploy to staging
	@echo "$(BLUE)Deploying to staging...$(NC)"
	./scripts/deployment/deploy-staging.sh

deploy-prod:  ## Deploy to production
	@echo "⚠️  Deploying to PRODUCTION"
	./scripts/deployment/deploy-prod.sh

# =====================================
# UTILITIES
# =====================================

clean:  ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage
	cd Frontend && rm -rf .next out
	@echo "✅ Clean complete"

audit:  ## Run security audit
	pip-audit
	cd Frontend && npm audit

docs:  ## Generate documentation
	cd docs && mkdocs build

docs-serve:  ## Serve documentation locally
	cd docs && mkdocs serve

migrate-structure:  ## Migrate to new project structure
	@echo "$(BLUE)Migrating to enterprise structure...$(NC)"
	python scripts/migrate_structure.py
	@echo "✅ Migration complete"

health-check:  ## Check system health
	@echo "$(BLUE)System Health Check$(NC)"
	@echo "Python version:"
	@python --version
	@echo "\nNode version:"
	@node --version 2>/dev/null || echo "Node not installed"
	@echo "\nDocker version:"
	@docker --version 2>/dev/null || echo "Docker not installed"
	@echo "\nPostgreSQL:"
	@pg_isready 2>/dev/null && echo "✅ PostgreSQL ready" || echo "❌ PostgreSQL not available"
	@echo "\nRedis:"
	@redis-cli ping 2>/dev/null && echo "✅ Redis ready" || echo "❌ Redis not available"

.DEFAULT_GOAL := help
