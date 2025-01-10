.PHONY: setup env install lint test test-slow test-all clean update-intro update-clean-intro clear-clean-intro build

ENV_NAME=tapeagents
PYTHON_VERSION=3.10
CONDA := $(shell which conda)
ifeq ($(CONDA),)
CONDA := $(CONDA_EXE)
endif
ifeq ($(CONDA),)
$(error "Conda not found. Please install Conda and try again.")
endif

setup: env install

env: 
	@$(CONDA) create --name $(ENV_NAME) python=$(PYTHON_VERSION) --yes

install:
	@$(CONDA) run --no-capture-output --name $(ENV_NAME) pip install -r ./requirements.txt -r ./requirements.dev.txt -r ./requirements.finetune.txt -r ./requirements.converters.txt
	@$(CONDA) run --no-capture-output --name $(ENV_NAME) pip install -e .

lint:
	@$(CONDA) run --no-capture-output --name ${ENV_NAME} ruff format .
	@$(CONDA) run --no-capture-output --name ${ENV_NAME} ruff check . --fix

test:
	@$(CONDA) run --no-capture-output --name ${ENV_NAME} pytest -s --color=yes -m "not slow" tests/

test-slow:
	@$(CONDA) run --no-capture-output --name ${ENV_NAME} pytest -m "slow" tests/

test-all:
	@$(CONDA) run --no-capture-output --name ${ENV_NAME} pytest tests/

clean:
	@$(CONDA) env remove --name $(ENV_NAME) --yes
	@$(CONDA) clean --all --yes

update-intro:
	@cp examples/intro_clean.ipynb intro.ipynb
	@$(CONDA) run --no-capture-output --name ${ENV_NAME} jupyter execute --inplace intro.ipynb

update-clean-intro:
	@$(CONDA) run --no-capture-output --name ${ENV_NAME} jupyter nbconvert intro.ipynb --output=examples/intro_clean.ipynb --to notebook --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True

clear-clean-intro:
	@$(CONDA) run --no-capture-output --name ${ENV_NAME} jupyter nbconvert --inplace examples/intro_clean.ipynb --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True

build:
	@mkdir -p dist
	@rm dist/*
	@python3 -m build --outdir dist/