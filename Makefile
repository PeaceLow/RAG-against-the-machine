install:
	uv sync

lint:
	uv run flake8 --exclude .venv,build,dist,vllm-0.10.1
	uv run mypy . --explicit-package-bases --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs


lint-strict:
	uv lint --strict

clean:
	rm -rf build dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

fclean: clean
	rm -rf .venv

.PHONY: install lint lint-strict fclean