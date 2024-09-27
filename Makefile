.PHONY: setup env install lint test test-slow test-all clean

ENV_NAME=tapeagents
PYTHON_VERSION=3.10
CONDA := $(shell which conda)

ifeq ($(CONDA),)
$(error "Conda not found. Please install Conda and try again.")
endif

setup: env install

env: 
	$(CONDA) create --name $(ENV_NAME) python=$(PYTHON_VERSION) --yes

install:
	$(CONDA) run --name $(ENV_NAME) pip install -r ./requirements.txt -r ./requirements.dev.txt -r ./requirements.finetune.txt -r ./requirements.converters.txt
	$(CONDA) run --name $(ENV_NAME) pip install -e .

lint:
	$(CONDA) run --name ${ENV_NAME} ruff format .

test:
	$(CONDA) run --name ${ENV_NAME} pytest -m "not slow" tests/

test-slow:
	$(CONDA) run --name ${ENV_NAME} pytest -m "slow" tests/

test-all:
	$(CONDA) run --name ${ENV_NAME} pytest tests/

clean:
	$(CONDA) env remove --name $(ENV_NAME) --yes
	$(CONDA) clean --all --yes
