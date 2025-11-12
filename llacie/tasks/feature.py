from .abstract import AbstractFeatureTask

class PresentingSymptomsFeatureTask(AbstractFeatureTask):
    """\
    Presenting symptoms found in the `hpi_short` section of a note.
    
    Limited to H&P notes only, for patients with suspected infection,
    and applying the exclusions listed in Pak et al. CID 2023:
    https://pubmed.ncbi.nlm.nih.gov/37531612/
    """
    name = "presenting_sx"
    note_join_sql = """
        LEFT JOIN "{{cohort_table}}" AS esc ON
            n."FK_episode_id" = esc."FK_episode_id"
    """
    note_where_sql = """
        "InpatientNoteTypeDSC" IN ('H&P')
        AND "infectionCriteria" IS TRUE
        AND "excl_ST0_combined" IS FALSE
    """