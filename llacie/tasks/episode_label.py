from .abstract import AbstractEpisodeLabelTask

from ..vocab import Vocab

class PresentingSymptomsEpisodeLabelV1Task(AbstractEpisodeLabelTask):
    """\
    Presenting symptoms for the given episode, limiting to patients with suspected infection.
    
    This implementation uses the original vocab generated for the K08 submission. This vocab
    is expected to be updated periodically, which will bump the number at the end of this class 
    name and the `name` attribute below.

    Typically, this task is implemented by examining features generated for the H&P notes
    (e.g., `PresentingSymptomsFeatureTask`).
    """
    name = "pres_sx_eplab1"
    max_human_labels = 10
    episode_join_sql = """
        LEFT JOIN "{{cohort_table}}" AS esc ON
            e."id" = esc."FK_episode_id"
    """
    episode_where_sql = """
        "infectionCriteria" IS TRUE
    """
    note_where_sql = """
        "InpatientNoteTypeDSC" IN ('H&P')
        AND EXTRACT(EPOCH FROM ("DateOfServiceDTS" - "startDTS")) < 24 * 60 * 60
    """

    vocab = Vocab("dt.pres_sx.ngrams.v1.xlsx", sheet_name="symptom_ngrams_top75ile")


class PresentingSymptomsEpisodeLabelV2Task(PresentingSymptomsEpisodeLabelV1Task):
    """\
    Presenting symptoms for the given episode, limiting to patients with suspected infection.
    
    This implementation uses an **updated vocab**, trying to simplify some of the ambiguities in
    the earlier vocab via additional curation (mostly merging and trimming terms).
    """
    name = "pres_sx_eplab2"

    vocab = Vocab("dt.pres_sx.ngrams.v2.xlsx", sheet_name="symptom_ngrams_top75ile.edited")


class PresentingSymptomsEpisodeLabelV2Top30Task(PresentingSymptomsEpisodeLabelV2Task):
    """\
    Presenting symptoms for the given episode, limiting to patients with suspected infection, and
    using the V2 updated vocab but also restricting to the top 30 most common terms, which we 
    pulled out for further analysis/illustration in the JAMA NO manuscript. These terms are 
    explicitly whitelisted in the attribute below.
    """
    name = "pres_sx_eplab2_top30"

    term_whitelist = {
        "fever", "dyspnea", "cough", "abdominal pain", "pain", "nausea", "chills", "fatigue",
        "vomiting", "altered mental status", "diarrhea", "chest pain", "weakness", "swelling",
        "sputum changes", "redness", "headache", "poor appetite", "malaise", "back pain",
        "hypoxemia", "hypotension", "dizziness", "dysuria", "rhinorrhea", "lightheadedness",
        "fall", "urinary frequency", "diaphoresis", "leg swelling"        
    }