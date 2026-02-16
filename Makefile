PYTHON_VERSIONS ?= 3.10 3.11 3.12

.PHONY: install lint format test test-eval test-all check clean leaderboard

install:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check invert tests
	uv run mypy invert tests

format:
	uv run ruff format invert tests

test:
	uv run --extra dev pytest -m "not slow"

test-eval:
	uv run --extra dev pytest -m slow

test-all:
	@for v in $(PYTHON_VERSIONS); do \
		echo "=== Python $$v ==="; \
		uv run --isolated --extra dev --python $$v pytest || exit 1; \
	done

check: lint test-all
	uv run pre-commit run --all-files

leaderboard:
	uv run python scripts/eval_all_release.py

docs:
	uv sync --all-extras
	uv run mkdocs serve
	uv run mkdocs build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
