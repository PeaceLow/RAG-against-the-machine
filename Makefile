install:
	uv sync

run:
	uv run python -m src.main --help

debug:
	uv run python -m pdb -m src.main --help

lint:
	uv run flake8 --exclude .venv,build,dist,vllm-0.10.1 src
	uv run mypy src --explicit-package-bases --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs


lint-strict:
	uv run flake8 --exclude .venv,build,dist,vllm-0.10.1 src
	uv run mypy src --strict --explicit-package-bases --ignore-missing-imports

clean:
	rm -rf build dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "data" -exec rm -rf {} +

fclean: clean
	rm -rf .venv

.PHONY: install run debug lint lint-strict clean fclean