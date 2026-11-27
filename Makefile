.PHONY: up down migrate seed index cli lint typecheck test eval-mock eval-live verify-all

up:
	docker compose up -d --wait

down:
	docker compose down

migrate:
	uv run alembic upgrade head

seed:
	uv run python -m src.db.seed

index:
	uv run python -m src.rag.indexer

cli:
	uv run python -m src.cli.chat

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src tests evals

test:
	uv run pytest -v

eval-mock:
	uv run python -m evals.run_evals --mode mock

eval-live:
	uv run python -m evals.run_evals --mode live

verify-all: up migrate seed index lint typecheck test eval-mock
