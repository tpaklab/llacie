import os
import tempfile
import pandas as pd
import time

from os import path
from sqlalchemy import create_engine, text
from contextlib import contextmanager
from datetime import datetime
from glob import glob
from collections import defaultdict

from .abstract import AbstractFeatureWorkerCache
from ..batch.slurm import SlurmJobManager
from ..tasks import TaskOutputType as out_t
from ..utils import echo_warn


class SqliteFeatureWorkerCache(AbstractFeatureWorkerCache):
    """
    Implements a WorkerCache for generating features from note sections by saving the inputs
    and outputs in a series of SQLite files (each file is one shard for one worker).
    """

    WORKER_CACHE_SUBDIR = "cache"
    WORKER_CACHE_EXTENSION = "cache.db"
    WORKER_CACHE_DONE_EXTENSION = "done.db"

    WORKER_CACHE_DIRECTORY_RETRIES = 10
    WORKER_CACHE_DIRECTORY_DELAY = 60

    # Change this to True to leave all files on disk, even after being pushed to the upstream DB
    WORKER_CACHE_KEEP_AFTER_SWEEP = False  

    def __init__(self, upstream_db, config, feature_strategy, **options):
        super().__init__(upstream_db, config, **options)

        self._dir = config.get("LLACIE_WORKER_CACHE_DIR", 
            f"{SlurmJobManager.SLURM_IO_DIR}/{self.WORKER_CACHE_SUBDIR}")
        self._ext = config.get("LLACIE_WORKER_CACHE_EXTENSION", self.WORKER_CACHE_EXTENSION)
        self._done_ext = config.get("WORKER_CACHE_DONE_EXTENSION", 
            self.WORKER_CACHE_DONE_EXTENSION)
        self._keep_after_sweep = config.get("LLACIE_WORKER_CACHE_KEEP_AFTER_SWEEP",
            self.WORKER_CACHE_KEEP_AFTER_SWEEP)
        self._retries = int(config.get("LLACIE_WORKER_CACHE_DIRECTORY_RETRIES",
            self.WORKER_CACHE_DIRECTORY_RETRIES))
        self._retry_delay = int(config.get("LLACIE_WORKER_CACHE_DIRECTORY_DELAY",
            self.WORKER_CACHE_DIRECTORY_DELAY))
        
        self.strategy = feature_strategy

        self._name = None
        self._default_shard = None
        self._current_shard = None
        self.prefix = ""  # For table names; they're less important here than in LlacieDatabase

        # This tracks the status of shards. Note that it is lazy and does not necessarily reflect
        # changes that other processes may have made to the filesystem. These can only be detected
        # and updated into this variable using the `self.find_finalized_shards()` generator.
        self._is_done = defaultdict(lambda: False)

        # If LLACIE_WORKER_CACHE_PATH is specified in the environment or the config at startup,
        # the cache directory, cache name, and current shard are parsed from this path.
        # This is how individual workers get automatically pointed to the right shard.
        prespecified_path = config.get("LLACIE_WORKER_CACHE_PATH")
        if prespecified_path is not None: 
            self._name, shard, is_done = self._parse_path(prespecified_path)
            if is_done is None: 
                raise RuntimeError("An invalid LLACIE_WORKER_CACHE_PATH was provided")
            self._current_shard = self._default_shard = shard
            self._is_done[shard] = is_done

        if not self._autocreate_directory():
            raise RuntimeError("Could not autocreate the directory for the Llacie worker cache")


    def _parse_path(self, shard_path):
        """
        Parse a given `shard_path` to a shard file in the cache into the corresponding cache name, 
        shard, and shard status.

        If there is a problem with parsing, this will return None for all components.
        """
        filename_parts = path.basename(shard_path).split(".", 1)
        shard = filename_parts[0]
        if not shard.isdigit(): return (None, None, None)
        name = path.basename(path.dirname(shard_path))
        is_done = filename_parts[1] == self._done_ext
        if not is_done and filename_parts[1] != self._ext: return (None, None, None)
        return (name, shard, is_done)
    

    def _check_if_done(self, shard):
        is_done = path.exists(self.get_path(for_shard=shard, is_done=True))
        self._is_done[shard] = is_done
        return is_done


    def _autocreate_directory(self):
        """
        Ensure that the directories required by the cache exist.
        
        If the cache does not yet have a name (they are typically named by Slurm job ID, but this
        may not be available before the job is submitted), a random one is created using mkdtemp.
        The name can be updated later with `self.rename()`.
        """
        os.makedirs(self._dir, exist_ok = True)
        if self._name is None:
            self._name = path.basename(tempfile.mkdtemp(prefix="unsubmitted-", dir=self._dir))
            os.chmod(path.join(self._dir, self._name), 0o700)
            return True

        dir_path = path.join(self._dir, self._name)
        # Workers can step on each other in accessing this directory for the first time (on NFS)
        # Therefore we allow a few retries
        for attempt in range(self._retries):
            try:
                os.makedirs(dir_path, exist_ok = True)
                return True
            except PermissionError:
                echo_warn(f"Permissions error in opening worker cache directory {dir_path}.")
                if attempt == self._retries - 1: return False
                echo_warn(f"Attempt {attempt + 1}, retrying in {self._retry_delay} seconds...")
                time.sleep(self._retry_delay)
        
        return False

    @contextmanager
    def conn(self, shard=None):
        """
        Provides a connection to a shard, by default the current shard. The current shard can be 
        changed by providing a `shard` argument, which can be any number or numeric string, but 
        typically these are sequential integers corresponding to Slurm job array task IDs.
        
        Intended to be used as a context manager, e.g., `with self.conn() as conn:` to ensure that
        the shard's corresponding SQLite database file is closed after any operations.
        
        Note that `conn.commit()` MUST be called by the user after any query that makes changes
        to the database, otherwise the changes will be rolled back!
        """
        if shard is not None:
            if not str(shard).isdigit(): raise RuntimeError("shard must be a numeric string")
            self._current_shard = str(shard)
        if self._current_shard is None: self._current_shard = self._default_shard
        if self._current_shard is None: raise RuntimeError("Unable to select a WorkerCache shard")
        mode = "ro" if self._check_if_done(self._current_shard) else "rwc"
        uri = f"sqlite:///file:{self.get_path()}?mode={mode}&uri=true"
        conn = None

        try:
            conn = create_engine(uri).connect()
            self._autocreate_tables(conn)
            yield conn
        finally:
            if conn is not None: conn.close()


    def get_path(self, for_name=None, for_shard=None, is_done=None):
        """Returns a full path to the current cache file being used, OR for the specific cache
        name, shard, and shard status given, when those are passed in as the optional arguments.
        If not given, these default to whatever the current values are."""
        if for_name is None: for_name = self._name
        if for_shard is None: for_shard = self._current_shard
        if is_done is None: is_done = self._is_done[for_shard]
        ext = self._done_ext if is_done else self._ext
        return os.path.join(self._dir, for_name, f"{for_shard}.{ext}")
    

    def _autocreate_tables(self, conn):
        create_table_sqls = [
            text(f"""\
                CREATE TABLE IF NOT EXISTS "{self.prefix}strategies" (
                    "id" BIGINT UNIQUE NOT NULL,
                    "FK_task_id" BIGINT,
                    "name" VARCHAR(255) NOT NULL,
                    "desc" TEXT,
                    "version" VARCHAR(255) NOT NULL,
                    "last_updated" DATETIME NOT NULL
                );"""),
            text(f"""\
                CREATE TABLE IF NOT EXISTS "{self.prefix}note_sections" (
                    "id" BIGINT UNIQUE NOT NULL,
                    "FK_note_id" BIGINT,
                    "section_name" VARCHAR(40),
                    "section_value" TEXT
                );"""),
            text(f"""\
                CREATE TABLE IF NOT EXISTS "{self.prefix}note_features" (
                    "FK_note_id" BIGINT NOT NULL,
                    "feature_name" VARCHAR(255) NOT NULL,
                    "FK_note_section_id" BIGINT,
                    "FK_strategy_id" BIGINT NOT NULL,
                    "llm_output_raw" TEXT NOT NULL,
                    "feature_value" TEXT,
                    "feature_updated" DATETIME NOT NULL,
                    "strategy_runtime" DECIMAL(12,5),
                    UNIQUE ("FK_note_id", "feature_name", "FK_strategy_id"),
                    FOREIGN KEY ("FK_strategy_id") REFERENCES "{self.prefix}strategies"("id"),
                    FOREIGN KEY ("FK_note_section_id") REFERENCES "{self.prefix}note_sections"("id")
                );""")
        ]
        for sql in create_table_sqls: conn.execute(sql)
        conn.commit()


    def rename(self, new_name):
        """
        Renames the entire cache. Intended to be used when the Slurm array job ID is known and
        you want to move the cache to a path that includes this array job ID so that the workers
        can find it.
        """
        new_name = str(new_name)
        old_dir = os.path.join(self._dir, self._name)
        os.rename(old_dir, os.path.join(self._dir, new_name))
        self._name = new_name
    

    def find_finalized_shards(self):
        """
        A generator that iterates through all of the shards that are currently finalized but not
        yet cleared from the filesystem.
        """
        finalized_shard_files = glob(self.get_path(for_shard='*', is_done=True))
        for shard_file in finalized_shard_files:
            _, shard, _ = self._parse_path(shard_file)
            self._is_done[shard] = True
            yield shard
    

    def _delete_shard(self, shard):
        os.unlink(self.get_path(for_shard=shard))


    ##############
    # These methods are used by the worker process
    ##############

    def get_note_section(self, note_id, strategy):
        if strategy.task.output_type != out_t.SECTION:
            raise RuntimeError("Only strategies for creating sections are allowed.")

        params = {
            "note_id": int(note_id), 
            "section_name": strategy.task.name
        }
        select_section_sql = text(f"""
            SELECT "id", "FK_note_id", "section_name", "section_value"
                FROM "{self.prefix}note_sections"
                WHERE "FK_note_id" = :note_id
                    AND "section_name" = :section_name
                ORDER BY "id" ASC
            """)

        with self.conn() as conn:
            return conn.execute(select_section_sql, params).fetchone()


    def upsert_note_feature(self, note_id, strategy, llm_output_raw, feature_value=None, 
            note_section_id=None, runtime=None):
        if strategy.name != self.strategy.name:
            raise RuntimeError("Inconsistent strategy supplied to upsert_note_feature!")
        if runtime is None:
            try: runtime = strategy.stop_timer()
            except RuntimeError: runtime = None

        get_strategy_params = {"strategy_name": self.strategy.name}
        get_strategy_sql = text(f"""
            SELECT "id", "name" from "{self.prefix}strategies"
                WHERE "name" = :strategy_name
            """)
        update_params = {
            "note_id": int(note_id), 
            "feature_name": strategy.task.name,
            "note_section_id": int(note_section_id),
            "llm_output_raw": llm_output_raw, 
            "feature_value": feature_value,
            "start_time": datetime.now(),
            "strategy_runtime": runtime
        }
        update_nf_sql = text(f"""
            INSERT INTO "{self.prefix}note_features" ("FK_note_id", "feature_name",
                    "FK_strategy_id", "FK_note_section_id", "llm_output_raw", "feature_value",
                    "feature_updated", "strategy_runtime")
                VALUES (:note_id, :feature_name, :strategy_id, :note_section_id, 
                    :llm_output_raw, :feature_value, :start_time, :strategy_runtime)
                ON CONFLICT ("FK_note_id", "feature_name", "FK_strategy_id")
                DO UPDATE SET "FK_note_section_id" = :note_section_id, 
                    "llm_output_raw" = :llm_output_raw,
                    "feature_value" = :feature_value,
                    "feature_updated" = :start_time,
                    "strategy_runtime" = :strategy_runtime
            """)
        
        with self.conn() as conn:
            strat_row = conn.execute(get_strategy_sql, get_strategy_params).fetchone()
            if strat_row is None:
                raise RuntimeError("Could not retrieve the strategy from the WorkerCache!")
            update_params["strategy_id"] = strat_row.id
            conn.execute(update_nf_sql, update_params)
            conn.commit()
    

    def finalize(self): 
        """
        Takes the current shard and marks it as done.

        In practice, this renames the file to end with ".done.db" which signals to other
        processes that it won't be written to any more (and is ready to be saved upstream). From
        this point onward, that shard can only be opened in read-only mode.
        """
        if self._is_done[self._current_shard] is True: 
            raise RuntimeError("finalize() can be called only once per shard!")
        os.rename(self.get_path(), self.get_path(is_done=True))
        self._is_done[self._current_shard] = True


    ##############
    # These methods are used by the parent process, which does have access to `self.upstream_db`
    ##############

    def cache_note_sections(self, id_chunks):
        """
        Saves all of the note sections specified for the note IDs in `id_chunks` into separate
        shards within this cache.
        
        We also cache the single row for the `feature_strategy` that was given to this WorkerCache.
        This is because its database row ID is also needed by workers, to preserve foreign keys.
        """
        strat_df = pd.DataFrame([dict(self.strategy.db_row._mapping)])
        for i, note_ids in enumerate(id_chunks):
            shard = i + 1
            sections_df = self.upstream_db.get_note_sections(note_ids, self.strategy.sec_strat)
            with self.conn(shard) as conn:
                strat_df.to_sql(f"{self.prefix}strategies", conn, if_exists='replace')
                sections_df.to_sql(f"{self.prefix}note_sections", conn, if_exists='replace')
                conn.commit()
    

    def sweep_into_upstream_db(self):
        count = 0
        for shard in self.find_finalized_shards():
            with self.conn(shard) as conn:
                # Dump all the completed note features into a DataFrame
                get_nf_sql = text(f"""
                    SELECT "FK_note_id", "llm_output_raw", "feature_value", "FK_note_section_id",
                        "strategy_runtime"
                        FROM "{self.prefix}note_features"
                    """)
                df = pd.read_sql_query(get_nf_sql, conn)
                # Push the contents of this DataFrame to the upstream database
                for i, row in df.iterrows():
                    self.upstream_db.upsert_note_feature(row['FK_note_id'], self.strategy,
                        row['llm_output_raw'], row['feature_value'], row['FK_note_section_id'],
                        row['strategy_runtime'])
            if not self._keep_after_sweep:
                self._delete_shard(shard)
            count += 1
        return count
