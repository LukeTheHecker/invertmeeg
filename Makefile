.PHONY: install lint format test test-eval test-all check audit build docs clean leaderboard

install:
	uv sync --locked --all-extras
	uv run --locked pre-commit install

lint:
	uv run --locked --all-extras ruff check .
	uv run --locked --all-extras ruff format --check .
	uv run --locked --all-extras mypy invert tests

format:
	uv run --locked --all-extras ruff check --fix .
	uv run --locked --all-extras ruff format .

test:
	uv run --locked pytest -m "not slow" --timeout=300

test-eval:
	uv run --locked pytest -m slow --timeout=600

test-all:
	uv sync --locked --all-extras
	uv run --locked --all-extras pytest --timeout=600

check: lint test-all audit build docs
	uv run --locked pre-commit run --all-files

audit:
	@requirements=$$(mktemp /tmp/invertmeeg-audit.XXXXXX); \
	trap 'rm -f "$$requirements"' EXIT; \
	uv export --locked --all-extras --no-emit-project --no-hashes --output-file "$$requirements" >/dev/null && \
	uv run --locked pip-audit --no-deps --disable-pip -r "$$requirements"

build:
	uv build
	uv run --locked twine check dist/*

leaderboard:
	uv run --locked python scripts/eval_all_release.py

docs:
	uv run --locked --extra docs mkdocs build --strict

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -not -path "./.venv/*" -exec rm -rf {} +
