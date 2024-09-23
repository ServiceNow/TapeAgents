.PHONY: setup install lint test test-slow clean

ENV_NAME=tapeagents
PYTHON_VERSION=3.10

CONDA := $(shell which conda)

setup: 
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) --yes
	$(CONDA) run --name $(ENV_NAME) pip install -r ./requirements.txt -r ./requirements-dev.txt -r ./requirements-finetune.txt -r ./requirements-converters.txt
	$(CONDA) run --name $(ENV_NAME) pip install -e .

install:
	$(CONDA) run --name $(ENV_NAME) pip install -r ./requirements.txt -r ./requirements-dev.txt -r ./requirements-finetune.txt -r ./requirements-converters.txt
	$(CONDA) run --name $(ENV_NAME) pip install -e .

lint:
	$(CONDA) run --name ${ENV_NAME} ruff format .

test:
	$(CONDA) run --name ${ENV_NAME} pytest -m "not slow"

test-slow:
	conda activate $(ENV_NAME)
	pytest -m "slow"

clean:
	conda env remove --name $(ENV_NAME) --yes
	conda clean --all --yes
