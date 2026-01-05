install: install-hooks
	uv sync --all-extras

start:
	uv run -m src.main run

jupyter:
	uv run jupyter lab

format:
	uv run ruff format .

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix . --unsafe-fixes

check:
	uv run ruff check .
	uv run ruff format --check .

# Git hooks
install-hooks:
	mkdir -p .git/hooks
	cp -f scripts/pre-commit.sh .git/hooks/
	chmod +x .git/hooks/pre-commit.sh
	@echo "Git pre-commit hook installed successfully."

.PHONY: install start jupyter format lint lint-fix check install-hooks
