.PHONY: test test-fast test-integration test-unit test-cov clean test-install help

help:
	@echo "LLaCIE Test Suite Commands:"
	@echo "  make install        Install package with test dependencies"
	@echo "  make test          Run all tests"
	@echo "  make test-fast     Run only fast tests (skip slow LLM-dependent tests)"
	@echo "  make test-integration  Run integration tests only"
	@echo "  make test-unit     Run unit tests only"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make clean         Remove test artifacts and cache"

test-install:
	pip install -e ".[test]"

test:
	pytest 

test-fast:
	pytest -m "not slow"

test-integration:
	pytest tests/integration/

test-unit:
	pytest tests/unit/

test-cov:
	# Note that this doesn't work with the integration tests since llacie is run in a subprocess.
	pytest --cov=llacie --cov-report=term-missing --cov-report=html

clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
