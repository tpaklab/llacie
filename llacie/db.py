import os
import re
import getpass
import pandas as pd
import numpy as np

from collections import OrderedDict
from click import UsageError
from datetime import datetime
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.engine.reflection import Inspector
from contextlib import contextmanager

from .config import ConfigError
from .tasks import TaskOutputType as out_t
from .tasks.abstract import AbstractTask
from .strategies import find_strategies
from .strategies.abstract import AbstractStrategy
from .utils import echo_info, echo_warn, generate_unique_id, create_row_like


HUNDRED_YEARS_IN_SECONDS = 60 * 60 * 24 * 365 * 100
LAST_UPDATED_COLUMNS = {
    out_t.SECTION: "last_updated",
    out_t.FEATURE: "feature_updated"
}

class LlacieDatabase(object):
    def __init__(self, config):
        if config['PG_URI'] is None:
            raise ConfigError("Could not build postgres connection string")

        self._conn = None
        self.config = config
        self.prefix = config.get('PG_TABLE_PREFIX', '')
        self.ep_table = config.get('PG_EPISODE_TABLE', 'llacie_episodes')
        self.cohort_table = config.get('PG_COHORT_TABLE', 'llacie_cohorts')
        self._metadata = None

    @property
    def conn(self):
        # Lazily initialize the connection to the database only when needed
        if self._conn is None:
            self._conn = create_engine(self.config['PG_URI']).connect()
        return self._conn

    @property
    def metadata(self):
        """A property that accesses reflected table MetaData created by SQLAlchemy."""
        if self._metadata is not None: return self._metadata
        self._metadata = MetaData()
        self._metadata.reflect(bind=self.conn)
        return self._metadata

    @property
    def tables(self):
        """A property that reflects all of the SQLAlchemy table objects.
        See: https://docs.sqlalchemy.org/en/20/core/reflection.html#reflecting-all-tables-at-once"""
        return self.metadata.tables
    
    def table(self, table_name):
        return Table(f"{self.prefix}{table_name}", self.metadata, autoload_with=self.conn)

    def interpolate_config_vars(self, sql):
        sql = re.sub('{{prefix}}', self.prefix, sql)
        sql = re.sub('{{episode_table}}', self.ep_table, sql)
        return re.sub('{{cohort_table}}', self.cohort_table, sql)

    def close(self):
        if self._conn is not None:
            self._conn.close()

    #############################
    # INITIALIZING THE DATABASE
    #############################

    def create_tables(self, overwrite = False, missing_tables_only = False):
        """Creates all of the database tables needed to run LLaCIE."""
        inspect = Inspector.from_engine(self.conn)

        sql_file = os.path.abspath(os.path.join(__file__, "../sql/schema.sql"))
        with open(sql_file) as f:
            sql = self.interpolate_config_vars(f.read())
            sql_chunks = re.split(r'^\s*CREATE TABLE\s+', sql, flags = re.MULTILINE)
            
            tbl_chunks = OrderedDict()
            llacie_tables = []
            for sql_chunk in sql_chunks:
                if len(sql_chunk.strip()) == 0: continue
                match = re.match(r'(IF +NOT +EXISTS +)?([a-zA-Z0-9_"]+) +[(]', sql_chunk)
                if match is None: 
                    raise ConfigError("Invalid schema in sql/schema.sql")

                table_name = match.group(2).strip('"')
                tbl_chunks[table_name] = f"CREATE TABLE {sql_chunk}"
                if match.group(1) is None: llacie_tables.append(table_name)

            existing_tables = [tbl for tbl in llacie_tables if inspect.has_table(tbl)]
            if not missing_tables_only and len(existing_tables) > 0:
                if not overwrite:
                    raise UsageError(f"Tables {', '.join(existing_tables)} already exist. If "
                        "you want to drop and create new blank tables, use --overwrite. "
                        "If you only want to add missing tables, use --missing-tables-only.")
                else:
                    # Existing tables are deleted in reverse order of creation
                    for existing_table in reversed(existing_tables):
                        echo_info(f"Dropping existing table: {existing_table}")
                        self.conn.execute(text(f"DROP TABLE \"{existing_table}\""))

            # Finally, create each table and its indexes/constraints
            for table_name, sql_chunk in tbl_chunks.items():
                if missing_tables_only and table_name in existing_tables: continue
                echo_info(f"Creating table: {table_name}")
                self.conn.execute(text(sql_chunk))
                self.conn.commit()


    def warn_or_truncate(self, table_name, overwrite=False):
        res = self.conn.execute(text(f"""
            SELECT COUNT(*) FROM "{self.prefix}{table_name}"
            """)).fetchone()
        
        if res[0] == 0:
            return
        elif not overwrite:
            raise UsageError(f"Table {self.prefix}{table_name} contains rows. If "
                "you want to truncate and recreate this table, use --overwrite.")
        
        self.conn.execute(
            text(f"TRUNCATE \"{self.prefix}{table_name}\" RESTART IDENTITY"))

    #############################
    # BASIC FUNCTIONS FOR APPENDING TO, UPDATING, AND RETRIEVING FROM TABLES
    #############################

    def get_row_by_id(self, table_name, id):
        tbl = self.table(table_name)
        return self.conn.execute(tbl.select().where(tbl.c.id == id)).fetchone()

    def _gather_column_values_into_set(self, table_and_colnames):
        """Gather all distinct values in multiple columns into a set.
        This is a helper function that aids in avoiding ID collisions when inserting new rows."""
        existing_ids = set()
        for table_name, col_name in table_and_colnames:
            existing_ids_sql = text(f'SELECT DISTINCT "{col_name}" FROM "{table_name}"')
            existing_ids.update(pd.read_sql_query(existing_ids_sql, self.conn)[col_name])
        return existing_ids

    def append_df_to_table(self, df, table_name):
        df.to_sql(f"{self.prefix}{table_name}", self.conn, if_exists='append', 
            index=False)
        self.conn.commit()

    def update_table_from_df(self, df, table_name, key_col):
        tbl = self.table(table_name)
        for _, row in df.iterrows():
            key_val = row.pop(key_col)
            self.conn.execute(
                tbl.update()
                    .where(tbl.columns[key_col] == key_val)
                    .values(**row)
            )
        self.conn.commit()

    @contextmanager
    def foreign_keys_temporarily_dropped(self, table_name, fk_names=None):
        """Perform an action on the database with some foreign keys temporarily dropped.
        This is a context manager, so it's intended to be used in a `with` clause.
        
        IMPORTANT: This won't work with composite (multi-column) foreign keys."""
        tbl = self.table(table_name)
        fks = [fk for fk in tbl.foreign_key_constraints]
        if fk_names is not None:
            fks = [fk for fk in fks if fk.name in fk_names]
        
        for fk in fks:
            self.conn.execute(text(f"""
                ALTER TABLE "{self.prefix}{table_name}" 
                DROP CONSTRAINT "{fk.name}" 
                """))
        try:
            yield
        except:
            self.conn.rollback()
            raise
        finally:
            echo_info("Rebuilding foreign key constraints...")
            for fk in fks:
                self.conn.execute(text(f"""
                    ALTER TABLE "{self.prefix}{table_name}" 
                    ADD CONSTRAINT "{fk.name}" 
                    FOREIGN KEY ("{fk.columns[0].name}")
                    REFERENCES "{fk.referred_table.name}"("{fk.elements[0].column.name}")
                    ON DELETE {fk.ondelete}
                    ON UPDATE {fk.onupdate}
                    """))
                self.conn.commit()
            echo_info("Done.")


    #############################
    # EPISODES AND NOTES
    #############################

    def get_all_episodes(self):
        """Get all episodes that we will import notes for.
        Returns a pandas DataFrame with all rows."""
        sql_file = os.path.abspath(os.path.join(__file__, "../sql/episodes.sql"))
        with open(sql_file) as f:
            sql = self.interpolate_config_vars(f.read())
            eps_df = pd.read_sql_query(text(sql), self.conn)
        return eps_df

    def get_notes_without_text(self):
        """Get all notes and `NoteID`s that do not have corresponding `note_text`.
        Returns a pandas DataFrame with all rows."""
        notes_without_text_sql = text(f"""
            SELECT "id", "NoteID"
                FROM "{self.prefix}notes"
                WHERE "note_text" IS NULL
            """)
        return pd.read_sql_query(notes_without_text_sql, self.conn)

    def get_full_text_notes(self, note_ids):
        """Get all notes with `full_text` for a given iterable of `note`.`id`s.
        Returns a pandas DataFrame with all rows."""
        note_ids = [int(id) for id in note_ids]
        notes_with_full_text_sql = text(f"""
            SELECT "id", "note_text"
                FROM "{self.prefix}notes"
                WHERE "id" = ANY(:note_ids)
                ORDER BY "id" ASC
            """)
        return pd.read_sql_query(notes_with_full_text_sql, self.conn, 
            params={"note_ids": note_ids})
    
    def get_earliest_notes_with_feature(self, ep_ids, feat_strat, 
            time_limit=HUNDRED_YEARS_IN_SECONDS):
        """For the given episodes with `ep_ids` and a feature strategy `feat_strat`, 
        get the earliest non-empty note for each episode where that feature strategy has 
        successfully produced a `feature_value`.
        
        Returns a row including ..."""
        if feat_strat.task.output_type != out_t.FEATURE:
            raise RuntimeError("Only strategies for creating features are allowed.")

        params = {
            "ep_ids": [int(ep_id) for ep_id in ep_ids], 
            "feat_name": feat_strat.task.name,
            "strat_id": feat_strat.id,
            "time_limit": time_limit
        }
        select_earliest_notes_sql = text(f"""
            SELECT DISTINCT ON (n."FK_episode_id")
                    n."id", "feature_name", "feature_value",
                    e."id" AS "episode_id",
                    nf."id" AS "note_feature_id",
                    EXTRACT(EPOCH FROM ("DateOfServiceDTS" - "startDTS")) 
                        AS "relative_note_service_seconds"
                FROM "{self.prefix}notes" as n
                LEFT OUTER JOIN "{self.prefix}note_features" as nf
                    ON nf."FK_note_id" = n."id"
                        AND "feature_name" = :feat_name
                        AND "FK_strategy_id" = :strat_id
                LEFT JOIN "{self.ep_table}" as e
                    ON n."FK_episode_id" = e."id"
                WHERE "note_text" IS NOT NULL 
                    AND ("feature_name" = :feat_name AND "feature_value" IS NOT NULL)
                    AND n."FK_episode_id" = ANY(:ep_ids)
                    AND EXTRACT(EPOCH FROM ("DateOfServiceDTS" - "startDTS")) < :time_limit
                ORDER BY n."FK_episode_id",
                    "relative_note_service_seconds" ASC
            """)

        return pd.read_sql_query(select_earliest_notes_sql, self.conn, params=params)

    def get_notes_for_human_annotator(self, task, ep_or_pt_id):
        """Retrieves notes that would be relevant to a human creating annotations for a particular
        task regarding labeling of either episodes or patients. 
        
        Provide a `task` and specify the episode or patient with `ep_or_pt_id`.
        Returns a DataFrame."""
        if not isinstance(task, type) or not issubclass(task, AbstractTask):
            raise RuntimeError("First argument must be a task")
        
        if task.output_type == out_t.EPISODE_LABEL:
            relevant_notes_sql = text(f"""
                SELECT n."id", n."note_text"
                    FROM "{self.prefix}notes" AS n
                    LEFT JOIN "{self.ep_table}" as e
                        ON n."FK_episode_id" = e."id"
                    WHERE
                        n."FK_episode_id" = :ep_id
                        AND n."note_text" IS NOT NULL
                        AND {self.interpolate_config_vars(task.note_where_sql)}
                    ORDER BY "DateOfServiceDTS"
                """)
            df = pd.read_sql_query(relevant_notes_sql, self.conn, params={"ep_id": ep_or_pt_id})            
        else:
            raise NotImplementedError
        return df


    #############################
    # IMPORTING NOTES
    #############################

    def import_notes(self, notes):
        """Imports notes from a list containing the full text for each note.
        In the absence of any metadata, we generate reasonable default values.
        All notes are assumed to have a unique new episode they are associated with."""
        ids_to_collect = [(f"{self.prefix}notes", "NoteID"), (f"{self.prefix}notes", "PatientEncounterID"), 
            (self.ep_table, "admitPatientEncounterID"), (self.ep_table, "dcPatientEncounterID")]
        existing_ids = self._gather_column_values_into_set(ids_to_collect)
        current_timestamp = datetime.now()

        # Prepare data for bulk insert
        cohort_records = []
        episode_records = []
        note_records = []
        for note_text in notes:
            patient_encounter_id = note_id = generate_unique_id(existing_ids)
            patient_id = f"T{patient_encounter_id}"
            # FK_episode_id's will be generated after the episodes are inserted
            cohort_records.append({
                'infectionCriteria': True,
                'excl_ST0_combined': False
            })
            episode_records.append({
                'patientID': patient_id,
                'admitPatientEncounterID': patient_encounter_id,
                'dcPatientEncounterID': patient_encounter_id,
                'startDTS': current_timestamp
            })
            note_records.append({
                'NoteID': note_id,
                'PatientEncounterID': patient_encounter_id,
                'DateOfServiceDTS': current_timestamp,
                'note_text': note_text
            })

        # Insert episodes first to get their IDs for the FK_episode_id's
        episodes_df = pd.DataFrame(episode_records)
        episodes_df.to_sql(self.ep_table, self.conn, if_exists='append', index=False)
        self.conn.commit()

        # Get the newly inserted episode IDs
        # We need to query back the episodes we just inserted by their unique PatientEncounterID
        ep_ids_sql = text(f'''
            SELECT "id", "admitPatientEncounterID"
            FROM "{self.ep_table}"
            WHERE "admitPatientEncounterID" = ANY(:encounter_ids)
        ''')
        inserted_episodes = pd.read_sql_query(
            ep_ids_sql,
            self.conn,
            params={'encounter_ids': [rec['admitPatientEncounterID'] for rec in episode_records]}
        )

        # Create a mapping from PatientEncounterID to episode ID
        encounter_to_episode = dict(zip(
            inserted_episodes['admitPatientEncounterID'],
            inserted_episodes['id']
        ))

        # Now insert cohort records with the correct FK_episode_id
        for i, cohort_rec in enumerate(cohort_records):
            cohort_rec['FK_episode_id'] = encounter_to_episode[episode_records[i]['admitPatientEncounterID']]

        cohorts_df = pd.DataFrame(cohort_records)
        cohorts_df.to_sql(self.cohort_table, self.conn, if_exists='append', index=False)
        self.conn.commit()

        # Finally, insert notes with the correct FK_episode_id
        for i, note_rec in enumerate(note_records):
            note_rec['FK_episode_id'] = encounter_to_episode[note_rec['PatientEncounterID']]

        notes_df = pd.DataFrame(note_records)
        self.append_df_to_table(notes_df, 'notes')


    #############################
    # STRATEGIES AND TASKS
    #############################

    def get_or_register_task(self, task):
        """Gets the database row corresponding to a task (creating it if necessary)."""
        as_row = task.as_row()
        res = self.conn.execute(text(f"""
            SELECT * FROM "{self.prefix}tasks"
                WHERE "output_type" = :output_type
                AND "name" = :name
            """), as_row).fetchone()      
        if res is None:
            tbl = self.table("tasks")
            insert_res = self.conn.execute(
                tbl.insert().values(**as_row).returning(tbl.c.id)
            )
            self.conn.commit()
            res = self.get_row_by_id("tasks", insert_res.fetchone()[0])
        return res

    def get_or_register_strategy(self, strategy):
        """Gets the database row corresponding to a strategy (creating it if necessary)."""
        as_row = strategy.as_row()
        res = self.conn.execute(text(f"""
            SELECT * FROM "{self.prefix}strategies"
                WHERE "FK_task_id" = :FK_task_id
                AND "name" = :name
                AND "version" = :version
            """), as_row).fetchone()
        if res is None:
            tbl = self.table("strategies")
            insert_res = self.conn.execute(
                tbl.insert().values(**as_row).returning(tbl.c.id)
            )
            self.conn.commit()
            res = self.get_row_by_id("strategies", insert_res.fetchone()[0])
        return res

    def get_unfinished_ids_for_strategy_or_task(
            self, strategy_or_task, not_annot_by=None, having_notes=False, newer_than=None):
        """Gets the IDs for notes, episodes, or patients for which a particular strategy or
        task has not yet been completed.
        
        If `not_annot_by` is provided, IDs for entities that specifically have not been annotated 
        by the given username are returned.
        
        If `having_notes` is True, will further filter to only IDs for episodes/patients that 
        have at least one relevant note (specified by the task's `note_where_sql` attribute).
        
        If `newer_than` is provided, IDs"""

        strategy_sql_clause = having_notes_join_sql = having_notes_where_sql = ""
        if isinstance(strategy_or_task, type) and issubclass(strategy_or_task, AbstractTask):
            task = strategy_or_task
        elif isinstance(strategy_or_task, AbstractStrategy):
            strategy_sql_clause = f"AND \"FK_strategy_id\" = {strategy_or_task.id} "
            task = strategy_or_task.task
        else:
            raise RuntimeError("First argument must be a strategy or a task")

        human_annotator_sql_clause = "IS NULL"
        if not_annot_by is not None: human_annotator_sql_clause = f"= '{not_annot_by}'"
        if newer_than is not None:
            if not isinstance(newer_than, datetime): raise RuntimeError("newer_than not a datetime")
            if task.output_type not in LAST_UPDATED_COLUMNS: raise NotImplementedError
            last_updated_col = LAST_UPDATED_COLUMNS[task.output_type]
            strategy_sql_clause += f"AND \"{last_updated_col}\" >= '{newer_than}' "

        if having_notes:
            having_notes_join_sql = f"""
                LEFT OUTER JOIN "{self.prefix}notes" AS n
                    ON e."id" = n."FK_episode_id"
            """
            having_notes_where_sql = f"""
                AND ({self.interpolate_config_vars(task.note_where_sql)})
                AND n."note_text" IS NOT NULL
            """

        if task.output_type == out_t.SECTION:
            notes_without_section_sql = text(f"""
                SELECT n."id"
                    FROM "{self.prefix}notes" AS n
                    LEFT OUTER JOIN "{self.prefix}note_sections" as ns
                        ON ns."FK_note_id" = n."id"
                            AND "section_name" = '{task.name}'
                            {strategy_sql_clause}
                    WHERE "note_text" IS NOT NULL 
                        AND ("section_name" IS NULL OR 
                            ("section_name" = '{task.name}' AND "section_value" IS NULL))
                        AND ({self.interpolate_config_vars(task.note_where_sql)})
                    ORDER BY "id"
                """)
            df = pd.read_sql_query(notes_without_section_sql, self.conn)
        elif task.output_type == out_t.FEATURE:
            notes_without_feature_sql = text(f"""
                SELECT n."id"
                    FROM "{self.prefix}notes" AS n
                    LEFT OUTER JOIN "{self.prefix}note_features" as nf
                        ON nf."FK_note_id" = n."id"
                            AND "feature_name" = '{task.name}'
                            {strategy_sql_clause}
                    {self.interpolate_config_vars(task.note_join_sql)}
                    WHERE "note_text" IS NOT NULL 
                        AND ("feature_name" IS NULL OR 
                            ("feature_name" = '{task.name}' AND "feature_value" IS NULL))
                        AND ({self.interpolate_config_vars(task.note_where_sql)})
                    ORDER BY "id"
                """)
            df = pd.read_sql_query(notes_without_feature_sql, self.conn)
        elif task.output_type == out_t.EPISODE_LABEL:
            eps_without_label_sql = text(f"""
                SELECT DISTINCT e."id"
                    FROM "{self.ep_table}" AS e
                    LEFT OUTER JOIN "{self.prefix}episode_labels" as el
                        ON el."FK_episode_id" = e."id"
                            AND "task_name" = '{task.name}'
                            {strategy_sql_clause}
                            AND "FK_human_annotator" {human_annotator_sql_clause}
                    {self.interpolate_config_vars(task.episode_join_sql)}
                    {having_notes_join_sql}
                    WHERE el."task_name" IS NULL
                        AND ({self.interpolate_config_vars(task.episode_where_sql)})
                        {having_notes_where_sql}
                    ORDER BY "id"
                """)
            df = pd.read_sql_query(eps_without_label_sql, self.conn)            
        else:
            raise NotImplementedError  # We haven't yet implemented any PATIENT_LABEL tasks
        return df['id'].to_numpy()


    def get_ids_annotated_for_task(self, annot_task, id_task=None, annot_by=None):
        """Gets the IDs for notes, episodes, or patients (corresponding to task `id_task`) for 
        which a human has already provided annotations for task `annot_task`. These can be the 
        same task.
        
        If `annot_by` is provided, only IDs for entities annotated by the given human's
        username are included."""

        if id_task is None: id_task = annot_task
        annot_by_sql_clause = "IS NOT NULL"
        if annot_by is not None: annot_by_sql_clause = f"= '{annot_by}'"

        if id_task.output_type in (out_t.SECTION, out_t.FEATURE, out_t.NOTE_LABEL):
            if annot_task.output_type == out_t.EPISODE_LABEL:
                notes_annot_for_task_sql = text(f"""
                    SELECT DISTINCT n."id"
                        FROM "{self.prefix}notes" AS n
                        LEFT JOIN "{self.ep_table}" as e
                            ON n."FK_episode_id" = e."id"
                        LEFT OUTER JOIN "{self.prefix}episode_labels" as el
                            ON el."FK_episode_id" = e."id"
                                AND "task_name" = '{annot_task.name}'
                                AND "FK_human_annotator" {annot_by_sql_clause}
                        WHERE "task_name" IS NOT NULL
                        ORDER BY "id"
                    """)
                df = pd.read_sql_query(notes_annot_for_task_sql, self.conn)
            else:
                raise NotImplementedError
        elif (id_task.output_type == out_t.EPISODE_LABEL and 
                annot_task.output_type == out_t.EPISODE_LABEL):
            eps_annot_for_task_sql = text(f"""
                SELECT DISTINCT e."id"
                    FROM "{self.ep_table}" AS e
                    LEFT OUTER JOIN "{self.prefix}episode_labels" as el
                        ON el."FK_episode_id" = e."id"
                            AND "task_name" = '{annot_task.name}'
                            AND "FK_human_annotator" {annot_by_sql_clause}
                    WHERE "task_name" IS NOT NULL
                    ORDER BY "id"
                """)
            df = pd.read_sql_query(eps_annot_for_task_sql, self.conn)
        else:
            raise NotImplementedError  # We haven't yet implemented any PATIENT_LABEL tasks
        return df['id'].to_numpy()


    #############################
    # SECTIONS OF NOTES
    #############################

    def insert_note_section(self, note_id, strategy, section_value):
        if strategy.task.output_type != out_t.SECTION:
            raise RuntimeError("Only strategies for creating sections are allowed.")

        params = {
            "FK_note_id": note_id, 
            "section_name": strategy.task.name,
            "section_value": section_value,
            "FK_strategy_id": strategy.id,
            "now": datetime.now()
        }
        insert_section_sql = text(f"""
            INSERT INTO "{self.prefix}note_sections"
                    ("FK_note_id", "section_name", "section_value", "FK_strategy_id",
                    "last_updated")
                VALUES (:FK_note_id, :section_name, :section_value, :FK_strategy_id,
                    :now)
            """)
        
        self.conn.execute(insert_section_sql, params)
        self.conn.commit()
    
    def filter_to_notes_with_section(self, note_ids, strategy):
        """Takes an iterable of note_ids and returns only the ones that have at least one
        particular section extracted for the given strategy."""
        if strategy.task.output_type != out_t.SECTION:
            raise RuntimeError("Only strategies for creating sections are allowed.")
        
        select_note_ids_having_section_sql = text(f"""
            SELECT DISTINCT "FK_note_id"
                FROM "{self.prefix}note_sections"
                WHERE "section_name" = '{strategy.task.name}'
            """)
        df = pd.read_sql_query(select_note_ids_having_section_sql, self.conn) 
        note_ids_having_section = df['FK_note_id'].to_numpy()
        return np.intersect1d(note_ids, note_ids_having_section)

    
    def get_note_section(self, note_id, strategy):
        df = self.get_note_sections([note_id], strategy)
        return create_row_like(**df.iloc[0].to_dict())


    def get_note_sections(self, note_ids, strategy):
        if strategy.task.output_type != out_t.SECTION:
            raise RuntimeError("Only strategies for creating sections are allowed.")

        params = {
            "note_ids": [int(note_id) for note_id in note_ids],
            "section_name": strategy.task.name
        }
        select_sections_sql = text(f"""
            SELECT "id", "FK_note_id", "section_name", "section_value"
                FROM "{self.prefix}note_sections"
                WHERE "FK_note_id" = ANY(:note_ids)
                    AND "section_name" = :section_name
                ORDER BY "id" ASC
            """)

        return pd.read_sql_query(select_sections_sql, self.conn, params=params)



    #############################
    # FEATURES OF NOTES
    #############################

    def upsert_note_feature(self, note_id, strategy, llm_output_raw, feature_value=None, 
            note_section_id=None, runtime=None):
        if strategy.task.output_type != out_t.FEATURE:
            raise RuntimeError("Only strategies for creating features are allowed.")
        if runtime is None:
            try: runtime = strategy.stop_timer()
            except RuntimeError: runtime = None

        params = {
            "note_id": int(note_id), 
            "feature_name": strategy.task.name,
            "note_section_id": int(note_section_id), 
            "strategy_id": int(strategy.id), 
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
        
        self.conn.execute(update_nf_sql, params)
        self.conn.commit()


    #############################
    # EPISODE LABELS
    #############################

    def get_episode_labels(self, ep_label_strategy, ep_ids=None, human_annotated=False):
        params = {
            "strategy_id": ep_label_strategy.id,
            "task_id": ep_label_strategy.task_id,
            "ep_ids": [int(ep_id) for ep_id in ep_ids] if ep_ids is not None else None
        }
        ep_ids_sql = '' if ep_ids is None else 'AND "FK_episode_id" = ANY(:ep_ids)'
        columns = ''' "id", "FK_note_feature_id", "FK_episode_id", "FK_strategy_id",
            "FK_task_id", "task_name", "label_name", "label_value", 
            "line_number", "FK_human_annotator"
            '''
        order_by = 'ORDER BY "FK_episode_id", "line_number", "label_name", "id"'

        if type(human_annotated) is str:
            params["human_annotator"] = human_annotated
            get_labels_sql = text(f"""
                SELECT {columns}
                    FROM "{self.prefix}episode_labels" 
                    WHERE "FK_task_id" = :task_id
                        AND "FK_strategy_id" IS NULL
                        AND "FK_human_annotator" = :human_annotator
                        {ep_ids_sql}
                    {order_by}
                """)        
        elif human_annotated is True:
            get_labels_sql = text(f"""
                SELECT {columns}
                    FROM "{self.prefix}episode_labels" 
                    WHERE "FK_task_id" = :task_id
                        AND "FK_strategy_id" IS NULL
                        AND "FK_human_annotator" IS NOT NULL
                        {ep_ids_sql}
                    {order_by}
                """)
        else:
            get_labels_sql = text(f"""
                SELECT {columns}
                    FROM "{self.prefix}episode_labels" 
                    WHERE "FK_task_id" = :task_id
                        AND "FK_strategy_id" = :strategy_id
                        AND "FK_human_annotator" IS NULL
                        {ep_ids_sql}
                    {order_by}
                """)

        return pd.read_sql_query(get_labels_sql, self.conn, params=params)


    def import_episode_labels(self, episode_label_task, input_xlsx, sheet_name=None, 
            human_username=None):
        if sheet_name is None: sheet_name = 0
        if human_username is None: human_username = getpass.getuser()
        self.ensure_annotator_exists(human_username)

        strats = find_strategies(output_type=out_t.EPISODE_LABEL, task=episode_label_task)
        if len(strats) == 0:
            raise RuntimeError("Could not find an episode labelling task by that name")
        strategy = strats[0](self, self.config)

        df = pd.read_excel(input_xlsx, sheet_name=sheet_name)
        long_df = df.assign(label_name=df['human_labels'].str.split(r'\s*[|]\s*'))
        long_df = long_df.explode('label_name').reset_index(drop=True)
        filtered_df = long_df.dropna(subset=['label_name'])
        if len(filtered_df) < len(long_df):
            echo_warn(f"{len(long_df) - len(filtered_df)} rows in the XLSX had zero labels")
        long_df = filtered_df[['FK_episode_id', 'label_name']].copy()

        vocab = strategy.task.vocab
        invalid_terms = [term for term in long_df.label_name if term not in vocab]
        if len(invalid_terms) > 0:
            raise RuntimeError(f"Invalid labels not in the vocab: {invalid_terms}")

        long_df['FK_task_id'] = strategy.task_id
        long_df['task_name'] = strategy.task.name
        long_df['FK_human_annotator'] = human_username
        long_df['label_value'] = 1.0

        params = {
            "ep_ids": [int(ep_id) for ep_id in long_df['FK_episode_id'].unique()],
            "task_id": strategy.task_id,
            "human_username": human_username
        }
        delete_labels_sql = text(f"""
            DELETE FROM "{self.prefix}episode_labels" 
                WHERE "FK_episode_id" = ANY(:ep_ids)
                    AND "FK_task_id" = :task_id
                    AND "FK_strategy_id" IS NULL
                    AND "FK_human_annotator" = :human_username
            """)
        self.conn.execute(delete_labels_sql, params)

        self.append_df_to_table(long_df, 'episode_labels')  # This also .commit()'s changes

        unique_eps = len(long_df['FK_episode_id'].unique())
        echo_info(f"{len(long_df)} labels imported for {unique_eps} episodes.")
    

    def replace_episode_labels(self, ep_label_strategy, row_values, labels_dict):
        if ep_label_strategy.task.output_type != out_t.EPISODE_LABEL:
            raise RuntimeError("Only strategies for creating episode labels are allowed.")
        
        params = {
            "episode_id": row_values.episode_id,
            "strategy_id": ep_label_strategy.id,
            "task_name": ep_label_strategy.task.name,
            "note_feature_id": row_values.note_feature_id,
            "task_id": ep_label_strategy.task_id
        }
        delete_labels_sql = text(f"""
            DELETE FROM "{self.prefix}episode_labels" 
                WHERE "FK_episode_id" = :episode_id
                    AND "FK_strategy_id" = :strategy_id
                    AND "task_name" = :task_name
                    AND "FK_human_annotator" IS NULL
            """)
        self.conn.execute(delete_labels_sql, params)

        for label_name, line_no in labels_dict.items():
            params.update({
                "label_name": label_name,
                "line_no": line_no
            })
            insert_label_sql = text(f"""
                INSERT INTO "{self.prefix}episode_labels" ("FK_note_feature_id", 
                    "FK_episode_id", "FK_strategy_id", "FK_task_id", "task_name", 
                    "label_name", "label_value", "line_number", "FK_human_annotator")
                VALUES (:note_feature_id, :episode_id, :strategy_id, :task_id, :task_name,
                    :label_name, 1.0, :line_no, NULL)
                """)
            self.conn.execute(insert_label_sql, params)
        
        self.conn.commit()


    #############################
    # HUMAN ANNOTATORS
    #############################
    def ensure_annotator_exists(self, human_username, warn=True):
        tbl = self.table("annotators")
        query = tbl.select().where(tbl.c.username == human_username)
        user = self.conn.execute(query).fetchone()
        if user is None:
            if warn: echo_warn(f"Creating entry for human annotator '{human_username}'")
            insert_user_sql = text(f"""
                INSERT INTO "{self.prefix}annotators" ("username", "admin")
                VALUES (:username, FALSE)
                """)
            self.conn.execute(insert_user_sql, {"username": human_username})