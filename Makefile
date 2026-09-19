UV ?= uv
TEST = $(UV) run --group test python -m unittest discover -s tests -v

.PHONY: build test coverage

build:
	$(UV) build

test:
	$(TEST)

coverage:
	$(UV) run --group test coverage erase
	$(UV) run --group test coverage run -m unittest discover -s tests -v
	$(UV) run --group test coverage report -m
