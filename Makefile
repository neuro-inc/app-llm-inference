SHELL := /bin/sh -e
IMAGE_NAME ?= app-llm-inference
IMAGE_TAG ?= latest

.PHONY: all
all: lint test

.PHONY: test
test: test-unit

.PHONY: install setup
install setup:
	poetry config virtualenvs.in-project true
	poetry install --with dev
	poetry run pre-commit install;

.PHONY: format
format:
ifdef CI
	poetry run pre-commit run --all-files --show-diff-on-failure
else
	# automatically fix the formatting issues and rerun again
	poetry run pre-commit run --all-files || poetry run pre-commit run --all-files
endif

.PHONY: lint
lint: format
	poetry run mypy .apolo

.PHONY: test-unit
test-unit:
	poetry run pytest -vvs --cov=.apolo --cov-report xml:.coverage.unit.xml .apolo/tests/unit

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
	for schema in VLLMInferenceInputs VLLMInferenceOutputs LLama4Inputs DeepSeekInputs MistralInputs GptOssInputs Kimi2Inputs; do \
		app-types dump-types-schema .apolo/src/apolo_apps_llm_inference "$$schema" ".apolo/src/apolo_apps_llm_inference/schemas/$$schema.json"; \
	done
