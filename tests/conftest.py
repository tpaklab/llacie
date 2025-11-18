"""Pytest configuration and shared fixtures."""
import os
import shutil
import subprocess
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
import pytest
from llacie.config import Config


@contextmanager
def _create_temp_database():
    """Context manager to create and cleanup a temporary PostgreSQL database.

    Yields the llacie Config object for the test database.
    """
    db_name = f"llacie_test_{uuid.uuid4().hex[:8]}"
    old_config = Config()

    # Store original environment variables
    original_db_name = os.environ.get("PG_DBNAME")
    original_pgpassword = os.environ.get("PGPASSWORD")
    os.environ["PGPASSWORD"] = old_config["PG_PASS"]
    cmd_args = ["-h", old_config["PG_HOST"], "-U", old_config["PG_USER"]]

    # Create the database
    try:
        subprocess.run(
            ["createdb", *cmd_args, db_name],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        raise Exception(f"Could not create test database: {e.stderr}")

    # Set PG_DBNAME to the test database
    os.environ["PG_DBNAME"] = db_name
    new_config = Config()

    try:
        yield new_config
    finally:
        # Cleanup: restore original PG_DBNAME
        if original_db_name:
            os.environ["PG_DBNAME"] = original_db_name
        else:
            os.environ.pop("PG_DBNAME", None)

        # Drop the test database
        try:
            subprocess.run(
                ["dropdb", *cmd_args, "--if-exists", db_name],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError:
            warnings.warn(f"Failed to cleanup test Postgres database '{db_name}'", UserWarning)
        finally:
            # Restore original PGPASSWORD
            if original_pgpassword:
                os.environ["PGPASSWORD"] = original_pgpassword
            else:
                os.environ.pop("PGPASSWORD", None)


def _setup_isolated_env(tmp_path):
    """Set up an isolated test environment with examples and config files.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Path to the temporary directory
    """
    # Copy test data
    test_data = Path(__file__).parent.parent / "examples"
    if test_data.exists():
        shutil.copytree(test_data, tmp_path / "examples")

    # Copy dotenv files
    for config_file in [".env.example", ".env"]:
        config_file = Path(__file__).parent.parent / config_file
        if config_file.exists():
            shutil.copy(config_file, tmp_path)

    return tmp_path


@pytest.fixture
def temp_db():
    """Create a temporary PostgreSQL test database (function-scoped).

    Uses existing PG_HOST, PG_USER, and PG_PASS environment variables
    and creates a uniquely named test database.
    """
    with _create_temp_database() as config:
        yield config


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Create an isolated environment for testing (function-scoped)."""
    monkeypatch.chdir(tmp_path)
    return _setup_isolated_env(tmp_path)


@pytest.fixture(scope="class")
def class_temp_db():
    """Class-scoped temporary PostgreSQL database.

    Shared across all tests in a test class to avoid recomputing expensive operations.
    """
    with _create_temp_database() as config:
        yield config


@pytest.fixture(scope="class")
def class_isolated_env(tmp_path_factory):
    """Class-scoped isolated environment.

    Shared across all tests in a test class to avoid recomputing expensive operations.
    Changes to the temporary directory for the duration of the test class.
    """
    tmp_path = tmp_path_factory.mktemp("llacie_class")
    original_cwd = os.getcwd()

    # Setup environment and change directory
    os.chdir(tmp_path)
    result = _setup_isolated_env(tmp_path)

    yield result

    # Restore original directory
    os.chdir(original_cwd)
