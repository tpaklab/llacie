import os
import pandas as pd

from click import echo
from sqlalchemy import create_engine, text
from tqdm import tqdm

from .config import ConfigError
from .utils import chunker

class EDWDatabase(object):
    ADMIT_INPATIENT_NOTE_TYPE_CDS = (4, 6, 8, 19, 26, 1000010)
    # H&P, ED Notes, ED Triage Notes, ED Provider Notes, H&P (View-Only), 
    # ED Progress/Update Note
    CONSULT_INPATIENT_NOTE_TYPE_CDS = (2,)
    # Consult Notes
    PROGRESS_INPATIENT_NOTE_TYPE_CDS = (1,)
    # Progress Notes
    DISCHARGE_INPATIENT_NOTE_TYPE_CDS = (5,)
    # Discharge Summary

    def __init__(self, config):
        if config['EDW_URI'] is None:
            raise ConfigError("Could not build EDW connection string. Please"
                " check that EDW_HOST, EDW_USER, and EDW_PASS are configured"
                " in .env, .env.example, and/or ~/.edw_env.")
        
        override_odbcsysini = config['EDW_OVERRIDE_ODBCSYSINI']
        if override_odbcsysini != '' and override_odbcsysini is not None:
            if override_odbcsysini[0] == '/':
                os.environ['ODBCSYSINI'] = override_odbcsysini
            else:
                os.environ['ODBCSYSINI'] = os.path.abspath(os.path.join(__file__,
                    "..", override_odbcsysini))

        self.conn = create_engine(config['EDW_URI']).connect()
        self.config = config


    def test_query(self):
        test = self.conn.execute(
            text("SELECT TOP 3 * FROM Epic.Reference.Department AS dep"))
        echo(test.fetchall())
    

    def gen_note_metadata(self, all_ec_ids, inpt_note_type_cds, chunk_size=1000):
        """Generator for note metadata given an iterable of `PatientEncounterID`s
        and an iterable of `InpatientNoteTypeCD`s.
        
        Yields pandas `DataFrame`s with up to `chunk_size` rows each."""

        desc = f"In batches of {chunk_size} rows"
        for ec_ids in tqdm(chunker(all_ec_ids, chunk_size), desc=desc):
            select_notes_sql = f"""
                SELECT ecn.NoteID,
                        ecn.PatientEncounterID,
                        ecn.InpatientNoteTypeDSC,
                        ecn.CurrentAuthorID,
                        ecnei.AuthorServiceDSC,
                        ecnei.AuthorProviderTypeDSC,
                        ecn.CreateDTS,
                        ecn.DateOfServiceDTS,
                        ecn.LastFiledDTS,
                        ecn.EDWLastModifiedDTS,
                        ecnei.NoteFiledDTS
                    FROM Epic.Clinical.Note_PHS AS ecn
                    LEFT JOIN Epic.Clinical.NoteEncounterInformation_PHS AS ecnei
                        ON ecn.NoteID = ecnei.NoteID
                    WHERE ecn.PatientEncounterID IN (
                            {','.join([str(ec_id) for ec_id in ec_ids])})
                        AND ecn.InpatientNoteTypeCD IN (
                            {','.join([str(code) for code in inpt_note_type_cds])})
                        AND ecn.AmbulatoryNoteFLG = 'N'
                        AND (ecn.DeletedStatusCategoryCD IS NULL 
                            OR ecn.DeletedStatusCategoryCD != 2)
                        AND (ecn.UnsignedFLG IS NULL OR ecn.UnsignedFLG != 'Y')
                        AND ecnei.MostRecentContactFLG = 'Y'
                    ORDER BY ecn.CreateDTS
                """
            df = pd.read_sql_query(text(select_notes_sql), self.conn)
            df_sorted = df.sort_values(by=['NoteID', 'NoteFiledDTS'])
            df_dedup = df_sorted.drop_duplicates(subset='NoteID', keep='last').copy()
            yield df_dedup

    def gen_admission_enc_note_metadata(self, adm_ec_ids, *args, **kwargs):
        note_type_codes = (self.ADMIT_INPATIENT_NOTE_TYPE_CDS +
            self.CONSULT_INPATIENT_NOTE_TYPE_CDS + self.PROGRESS_INPATIENT_NOTE_TYPE_CDS)
        return self.gen_note_metadata(adm_ec_ids, note_type_codes, *args, **kwargs)

    def gen_discharge_enc_note_metadata(self, dc_ec_ids, *args, **kwargs):
        return self.gen_note_metadata(dc_ec_ids,
            self.DISCHARGE_INPATIENT_NOTE_TYPE_CDS, *args, **kwargs)


    def gen_note_text(self, all_note_ids, chunk_size=1000):
        """Generator for note full text given an iterable of `NoteID`s.
        
        Yields pandas `DataFrame`s with up to `chunk_size` rows each."""

        desc = f"In batches of {chunk_size} rows"
        for note_ids in tqdm(chunker(all_note_ids, chunk_size), desc=desc):
            # Get full note texts for the latest versions of each note
            note_id_sql = "','".join([str(note_id) for note_id in note_ids])
            select_note_text_sql = f"""
                SELECT NoteCSNID, ContactDateRealNBR, NoteID, LineNBR, NoteTXT
                    FROM (
                        SELECT NoteCSNID, ContactDateRealNBR, NoteID, LineNBR, NoteTXT,
                                MAX(ContactDateRealNBR) OVER (PARTITION BY NoteID)
                                    AS max_contact_date_for_note_id
                            FROM Epic.Clinical.NoteText_PHS
                            WHERE NoteID IN ('{note_id_sql}')
                        ) AS ecnt
                    WHERE max_contact_date_for_note_id = ContactDateRealNBR
                    ORDER BY NoteID ASC, LineNBR ASC
                """
            df = pd.read_sql_query(text(select_note_text_sql), self.conn)

            # Sometimes there are zero rows, for a chunk of NoteIDs with no NoteTXT
            if len(df) == 0: continue
            # Join the text from multiple lines/rows into a single field
            def join_func(pieces):
                return "\n".join([(p if p is not None else "") for p in pieces])
            df['note_text'] = df.groupby(['NoteID'])['NoteTXT'].transform(join_func)
            # Yield a selection of columns, after deduplicating
            df = df[['NoteCSNID', 'ContactDateRealNBR', 'NoteID', 'note_text']]
            yield df.drop_duplicates()
    
    
    def close(self):
        self.conn.close()