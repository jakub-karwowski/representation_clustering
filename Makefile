DEVICE = cpu
PORT = 8888
GPU = all

# Set GPU_FLAG based on DEVICE
GPU_FLAG = $(if $(filter-out cpu,$(DEVICE)),--gpus=$(GPU),)

install:
	pip install -r requirements-$(DEVICE).txt

build_image:
	docker build -t representation_learning -f $(DEVICE).dockerfile .

run_container:
	docker run --rm $(GPU_FLAG) -p $(PORT):8888 -v `pwd`:/assignment --name representation_learning representation_learning

SHELL:=/bin/bash

vcreate:
	python3 -m venv .venv

vinstall:
	source ./.venv/bin/activate && \
	pip3 install -r requirements-${DEVICE}.txt

vrun:
	source ./.venv/bin/activate && \
	jupyter lab --no-browser

# CREATE VENV
# python -m venv .venv

# START VENV
# source ./.venv/bin/activate

# STOP VENV
# deactivate
