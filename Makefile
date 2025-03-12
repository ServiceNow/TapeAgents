.PHONY: setup lint lint-check test test-slow test-all audit clean update-intro update-clean-intro clear-clean-intro build

setup:
	@uv sync --all-extras

lint:
	@uv run ruff format .
	@uv run ruff check . --fix

lint-check:
	@uv run ruff check . --output-format github

test:
	@uv run --all-extras pytest -s --color=yes -m "not slow" tests/

coverage:
	@uv run --all-extras pytest --cov=tapeagents  -s --color=yes -m "not slow" tests/

test-core:
	@uv run pytest -s --color=yes tests/ --ignore-glob="tests/*/*"

test-slow:
	@uv run --all-extras pytest -m "slow" tests/

test-all:
	@uv run --all-extras pytest tests/

audit:
	@uv export --all-extras --format requirements-txt --no-emit-project > requirements.txt
	@uv run pip-audit -r requirements.txt --disable-pip --desc --aliases; \
	rm requirements.txt

clean:
	@uv cache clean
	@rm -rf .venv/

update-intro:
	@cp examples/intro_clean.ipynb intro.ipynb
	@uv run jupyter execute --inplace intro.ipynb

update-clean-intro:
	@uv run jupyter nbconvert intro.ipynb --output=examples/intro_clean.ipynb --to notebook --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True

clear-clean-intro:
	@uv run jupyter nbconvert --inplace examples/intro_clean.ipynb --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True

build:
	@rm -rf dist/
	@mkdir dist
	@uv build
