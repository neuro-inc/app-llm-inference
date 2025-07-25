.PHONY: all clean test lint format
all clean test lint format:

SHELL := /bin/sh -e

.PHONY: install
install:
	pip install pre-commit;
	pre-commit install;

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
