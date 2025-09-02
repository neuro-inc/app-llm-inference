.PHONY: all clean test lint format
all clean test lint format:

SHELL := /bin/sh -e
IMAGE_NAME ?= app-llm-inference
IMAGE_TAG ?= latest

.PHONY: install setup
install setup:
	poetry config virtualenvs.in-project true
	poetry install --with dev
	poetry run pre-commit install;

.PHONY: lint format
lint format:
ifdef CI
	pre-commit run --all-files --show-diff-on-failure
else
	# automatically fix the formatting issues and rerun again
	pre-commit run --all-files || pre-commit run --all-files
endif

.PHONY: test
test:

.PHONY: clean
clean:

.PHONY: build-hook-image
build-hook-image:
	docker build \
		-t $(IMAGE_NAME):latest \
		-f hooks.Dockerfile \
		.;

.PHONY: push-hook-image
push-hook-image:
	docker tag $(IMAGE_NAME):latest ghcr.io/neuro-inc/$(IMAGE_NAME):$(IMAGE_TAG)
	docker push ghcr.io/neuro-inc/$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: gen-types-schemas
gen-types-schemas:
	app-types dump-types-schema .apolo/src/apolo_apps_llm_inference llm-inference VLLMInferenceInputs .apolo/src/apolo_apps_llm_inference/schemas/VLLMInferenceInputs.json
	app-types dump-types-schema .apolo/src/apolo_apps_llm_inference llm-inference VLLMInferenceOutputs .apolo/src/apolo_apps_llm_inference/schemas/VLLMInferenceOutputs.json
