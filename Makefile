install: install-hooks
	uv sync --all-extras

FUNC ?= eva
# luna train eva llm img

start:
	uv run -m src.main $(FUNC)

inspect:
	uv run -m src.inspect_ppo_model run

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

.PHONY: install start inspect format lint lint-fix check install-hooks
