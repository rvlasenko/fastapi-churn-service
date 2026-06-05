.PHONY: install run test lint lint-fix typecheck docker-build docker-run

install:
	uv sync --all-extras

run:
	uv run uvicorn churn_service.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

lint-fix:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

typecheck:
	uv run pyright src/

docker-build:
	docker build -t churn-service .

docker-run:
	docker run --rm -p 8000:8000 churn-service
