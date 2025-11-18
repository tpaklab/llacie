# LLaCIE Test Suite

This directory contains the test suite for the LLaCIE package. Tests validate the complete pipeline from the README Quickstart section.

## Installation

Install test dependencies:

```bash
pip install -e ".[test]"
```

This installs:
- `pytest` - test framework
- `pytest-cov` - coverage reporting
- `pytest-timeout` - test timeouts

## Running Tests

### Quick tests only (skip slow LLM tests)

```bash
pytest -m "not slow"
```

### Run all tests including slow integration tests

```bash
pytest
```

### Run specific test file

```bash
pytest tests/integration/test_quickstart_with_validation.py
```

### Run specific test

```bash
pytest tests/integration/test_quickstart_with_validation.py::TestQuickstartWithValidation::test_01_init_db
```

### Run with verbose output

```bash
pytest -v
```

### Run with coverage report

```bash
pytest --cov=llacie --cov-report=html
```

## Test Organization

### Integration Tests (`tests/integration/`)

- `test_quickstart.py` - Integration tests with output and database validation

Integration tests validate the complete workflow from the README Quickstart:
1. `llacie init-db` - Initialize database
2. `llacie import-notes text` - Import 100 example notes
3. `llacie sections extract -s regex` - Extract sections
4. `llacie features extract -s llama3_8b` - Extract features with LLM
5. `llacie episode-labels extract` - Create episode labels
6. `llacie episode-labels import` - Import gold standard labels
7. `llacie episode-labels evaluate` - Evaluate performance

### Unit Tests (`tests/unit/`)

Unit tests for individual components (to be added).

### Test Fixtures (`tests/fixtures/`)

Reference outputs and expected values for regression testing.

## Test Markers

Tests are marked with pytest markers for selective execution:

- `@pytest.mark.slow` - Tests that take >30 seconds (LLM inference)
- `@pytest.mark.integration` - Integration tests of full pipeline
- `@pytest.mark.unit` - Fast unit tests of individual functions
- `@pytest.mark.requires_db` - Tests requiring database connection
- `@pytest.mark.requires_llm` - Tests requiring LLM model files

### Running specific test types

```bash
# Only fast tests
pytest -m "not slow"

# Only unit tests
pytest -m unit

# Only integration tests
pytest -m integration

# Skip tests requiring LLM
pytest -m "not requires_llm"
```

## Test Output Validation

Each test validates commands based on:

1. **Command output messages** - Expected messages from CLI (e.g., "Successfully imported X notes")
2. **Database table checks** - Verifying row counts and data presence
3. **Exit codes** - Ensuring commands complete successfully

Test runs save command outputs to `output_XX_*.txt` files in the test directory for manual inspection and comparison with known good values.

## Generating Reference Outputs

After confirming the pipeline produces correct results, generate reference outputs:

1. Run the full test suite:
   ```bash
   pytest tests/integration/test_quickstart_with_validation.py
   ```

2. The test saves output files to the temp directory. To preserve them as fixtures:
   ```bash
   # Copy output files from a successful test run
   cp /tmp/pytest-*/output_*.txt tests/fixtures/
   ```

3. Add validation against these reference files in tests:
   ```python
   # Compare current output with known good output
   expected = Path("tests/fixtures/output_07_episode_labels_evaluate.txt").read_text()
   # Add assertions comparing key metrics
   ```

## Database Requirements

Tests that validate database contents require:
- PostgreSQL database connection
- `DATABASE_URL` environment variable set

Example `.env` for testing:
```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/llacie_test
```

Tests will skip database validation if connection fails, allowing basic CLI testing without full database setup.

## Continuous Integration

For CI/CD pipelines, run fast tests only:

```bash
# Run tests that complete in <30 seconds
pytest -m "not slow" --tb=short

# Or use a custom timeout
pytest --timeout=30
```

## Adding New Tests

When adding new features:

1. Add unit tests for individual functions in `tests/unit/`
2. Add integration tests for new CLI commands in `tests/integration/`
3. Mark slow tests with `@pytest.mark.slow`
4. Update this README with new test descriptions

## Troubleshooting

### Tests fail with database connection errors

- Ensure PostgreSQL is running
- Check `DATABASE_URL` in `.env`
- Tests will skip database checks if connection fails

### Tests timeout

- Increase timeout in `pytest.ini` or use `--timeout=N`
- Skip slow tests with `-m "not slow"`

### LLM model download fails

- Ensure HuggingFace authentication: `huggingface-cli login`
- Check model access permissions
- Tests requiring models are marked with `@pytest.mark.requires_llm`
