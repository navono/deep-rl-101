install: install-hooks
	uv sync --all-extras

FUNC ?= gridworld_train
# llm img 
# unit_0_luna unit_0_train unit_0_eva unit_0_play unit_0_api
# basic_rl rl_spaces
# blackjack_train blackjack_eva blackjack_analyze
# gridworld_train gridworld_eva

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
