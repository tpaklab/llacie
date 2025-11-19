"""Integration tests for Quickstart workflow with output and database validation.

This validates each step based on:
1. Command output messages (as shown in app.py)
2. Database table checks where applicable
"""
import subprocess
import pytest
import getpass
import gzip

from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import create_engine, text

# Timeout values, in seconds
DEFAULT_TIMEOUT = 60
LONG_TIMEOUT = 600 

# Expected counts after the various steps of the pipeline are completed
TABLE_COUNT = 9
NOTE_COUNT = 100
SECTION_COUNT = 100
SHORT_FEATURE_COUNT = 2
HUMAN_LABELS_COUNT = 145

# A fixture that contains SQL data to support tests of steps after the LLM runs
SKIPTO_EPISODE_LABELS_SQL = "SKIPTO_episode-labels_extract.sql.gz"


def run_cmd(cmd, timeout=None, capture_output=True):
    if timeout is None: timeout = DEFAULT_TIMEOUT
    kwargs = {"check": True}
    if capture_output: 
        kwargs["check"] = False
        kwargs["text"] = True
    return subprocess.run(cmd, capture_output=capture_output, timeout=timeout, **kwargs)


@contextmanager
def get_db_connection(llacie_config):
    """Get database connection as a context manager.

    Ensures the connection is properly closed when the context exits.
    """
    engine = create_engine(llacie_config["PG_URI"])
    engine.llacie_config = llacie_config
    try:
        yield engine
    finally:
        engine.dispose()


def count_tables(engine):
    """Count the number of tables in the database."""
    query = """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
    """
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return result.scalar()


def count_rows(engine, table_name, where_clause=None):
    """Count rows in a table."""
    prefix = engine.llacie_config["PG_TABLE_PREFIX"]
    query = f"SELECT COUNT(*) FROM {prefix}{table_name}"
    if where_clause:
        query += f" WHERE {where_clause}"
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return result.scalar()


def open_possible_gzip(filepath, mode='rt'):
    """Open file, detecting gzip by magic number."""
    with open(filepath, 'rb') as f:
        magic = f.read(2)
    if magic == b'\x1f\x8b':  # gzip magic number
        return gzip.open(filepath, mode)
    return open(filepath, mode)


def load_sql_file(llacie_config, sql_file):
    """Loads an SQL file (optionally gzipped) into the database."""
    with get_db_connection(llacie_config) as engine:
        conn = engine.raw_connection()
        cursor = conn.cursor()
        with open_possible_gzip(sql_file, 'rt') as f:
            cursor.execute(f.read())
        conn.commit()
        conn.close()


class TestQuickstart:
    """Test the Quickstart workflow with proper validation at each step."""

    @pytest.mark.dependency()
    def test_01_init_db(self, isolated_env, temp_db):
        """Test: llacie init-db

        Expected output: A message for each created table
        Validation: Command succeeds and the expected number of tables exist
        """
        result = run_cmd(["llacie", "init-db"])

        # Validate command return code and outputted messages to stderr
        assert result.returncode == 0, f"init-db failed: {result.stderr}"
        message_count = result.stderr.count("Creating table:")
        assert message_count == TABLE_COUNT, \
            f"Expected {TABLE_COUNT} messages, got {message_count}"

        # Validate database
        with get_db_connection(temp_db) as engine:
            table_count = count_tables(engine)
            assert table_count == TABLE_COUNT, \
                f"Expected {TABLE_COUNT} tables in database, found {table_count}"


    @pytest.mark.dependency()
    def test_02_import_notes(self, isolated_env, temp_db):
        """Test: llacie import-notes text examples/admission-100.txt

        Expected output: "Successfully imported X notes"
        Validation: Check that the notes table has the expected number of rows
        """
        # Setup: init db first
        run_cmd(["llacie", "init-db"])

        notes_path = isolated_env / "examples" / "admission-100.txt"
        result = run_cmd(["llacie", "import-notes", "text", str(notes_path)])

        # Validate command return code
        assert result.returncode == 0, f"import-notes failed: {result.stderr}"

        # Validate output message
        assert "Successfully imported" in result.stderr, \
            f"Expected success message not found in output: {result.stderr}"
        assert f"{NOTE_COUNT} notes" in result.stderr, \
            f"Expected {NOTE_COUNT} notes to be imported, got: {result.stderr}"

        # Validate database
        with get_db_connection(temp_db) as engine:
            note_count = count_rows(engine, "notes")
            assert note_count == NOTE_COUNT, \
                f"Expected {NOTE_COUNT} note rows in database, found {note_count}"

        # Store result
        output_file = isolated_env / "output_02_import_notes.txt"
        output_file.write_text(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")


    @pytest.mark.dependency()
    def test_03_sections_extract(self, isolated_env, temp_db):
        """Test: llacie sections extract -s regex

        Expected output: "X notes will undergo section extraction"
        Validation: Check that the note_sections table has the expected number of rows
        """
        # Setup
        run_cmd(["llacie", "init-db"])
        notes_path = isolated_env / "examples" / "admission-100.txt"
        run_cmd(["llacie", "import-notes", "text", str(notes_path)])

        result = run_cmd(["llacie", "sections", "extract", "-s", "regex"])

        # Validate command return code
        assert result.returncode == 0, f"sections extract failed: {result.stderr}"

        # Validate output message
        assert "notes will undergo section extraction" in result.stderr, \
            f"Expected extraction message not found: {result.stderr}"

        # Validate database
        with get_db_connection(temp_db) as engine:
            section_count = count_rows(engine, "note_sections")
            assert section_count == SECTION_COUNT, \
                f"Expected {SECTION_COUNT} note_section rows, found {section_count}"

        # Store result
        output_file = isolated_env / "output_03_sections_extract.txt"
        output_file.write_text(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")


    @pytest.mark.dependency(depends=["TestQuickstart::test_03_sections_extract"])
    @pytest.mark.slow
    @pytest.mark.timeout(LONG_TIMEOUT)
    @pytest.mark.requires_llm
    def test_04_features_extract(self, isolated_env, temp_db):
        """Test: llacie features extract -s llama3_8b

        Expected output: "X notes will undergo feature extraction"
        Validation: Check that the note_features table has entries
        """
        # Setup - only run if test_03 hasn't already done this
        # Since we're using class fixtures, we need to ensure the pipeline is run
        run_cmd(["llacie", "init-db"])
        notes_path = isolated_env / "examples" / "admission-100.txt"
        run_cmd(["llacie", "import-notes", "text", str(notes_path)])
        run_cmd(["llacie", "sections", "extract", "-s", "regex"])

        result = run_cmd(
            ["llacie", "features", "extract", "-s", "llama3_8b", "-M", str(SHORT_FEATURE_COUNT)],
            timeout=LONG_TIMEOUT
        )

        # Validate command succeeded
        assert result.returncode == 0, f"features extract failed: {result.stderr}"

        # Validate output message
        assert "notes will undergo feature extraction" in result.stderr, \
            f"Expected extraction message not found: {result.stderr}"

        # Validate database
        with get_db_connection(temp_db) as engine:
            feature_count = count_rows(engine, "note_features")
            assert feature_count > 0, \
                f"Expected >0 note_features in database, found {feature_count}"

        # Store result
        output_file = isolated_env / "output_04_features_extract.txt"
        output_file.write_text(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")


    @pytest.mark.dependency()
    def test_05_episode_labels_extract(self, class_isolated_env, class_temp_db):
        """Test: llacie episode-labels extract -s pres_sx_eplab2.llama3_8b

        Expected output: "X episodes will undergo label creation"
        Validation: Check that episode_labels table has entries
        """
        # This test loads a cached copy of the database from after test_04_... completed
        sql_file = Path(__file__).parent.parent / "fixtures" / SKIPTO_EPISODE_LABELS_SQL
        load_sql_file(class_temp_db, sql_file)

        # Run the next step of the llacie pipeline: extracting episode labels
        result = run_cmd(
            ["llacie", "episode-labels", "extract", "-s", "pres_sx_eplab2.llama3_8b"]
        )

        # Validate command succeeded
        assert result.returncode == 0, f"episode-labels extract failed: {result.stderr}"

        # Validate output message (from app.py line 218)
        assert "episodes will undergo label creation" in result.stderr, \
            f"Expected label creation message not found: {result.stderr}"

        # Validate database
        with get_db_connection(class_temp_db) as engine:
            label_count = count_rows(engine, "episode_labels")
            assert label_count > 0, f"Expected >0 episode labels in database, found {label_count}"


    @pytest.mark.dependency(depends=["TestQuickstart::test_05_episode_labels_extract"])
    def test_06_episode_labels_import(self, class_isolated_env, class_temp_db):
        """Test: llacie episode-labels import pres_sx_eplab2 examples/admission-100-labels.xlsx

        Expected output: Import completes successfully
        Validation: Check that human annotations are in database

        NOTE: Uses class-scoped fixtures to reuse database from test_05
        """
        # No setup needed - test_05 already has the full pipeline

        labels_path = class_isolated_env / "examples" / "admission-100-labels.xlsx"
        result = run_cmd(
            ["llacie", "episode-labels", "import", "pres_sx_eplab2", str(labels_path)]
        )

        # Validate command succeeded
        assert result.returncode == 0, f"episode-labels extract failed: {result.stderr}"

        # Validate database
        username = getpass.getuser()
        with get_db_connection(class_temp_db) as engine:
            label_count = count_rows(engine, "episode_labels", f"\"FK_human_annotator\"='{username}'")
            assert label_count == HUMAN_LABELS_COUNT, \
                f"Expected {HUMAN_LABELS_COUNT} human-created episode_labels, found {label_count}"


    @pytest.mark.dependency(depends=["TestQuickstart::test_06_episode_labels_import"])
    def test_07_episode_labels_evaluate(self, class_isolated_env, class_temp_db):
        """Test: llacie episode-labels evaluate

        Expected output: Confusion matrices with accuracy, precision, recall, F1
        Validation: Check that evaluation metrics are present in output

        NOTE: Uses class-scoped fixtures to reuse database from test_06
        """
        # No setup needed - test_06 already has the full pipeline with imported labels!

        result = run_cmd(["llacie", "episode-labels", "evaluate"])

        # Validate command succeeded
        assert result.returncode == 0, f"episode-labels evaluate failed: {result.stderr}"

        # Validate that the output contains evaluation metrics
        output_lower = result.stdout.lower()
        metrics = ["accuracy", "precision", "recall", "f1"]
        found_metrics = [m for m in metrics if m in output_lower]
        assert len(found_metrics) == 4, \
            f"Expected evaluation metrics in output, found only {found_metrics}: {result.stdout}"
