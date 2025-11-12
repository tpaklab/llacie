from .abstract import AbstractSectionTask

class ShortHPISectionTask(AbstractSectionTask):
    """\
    Simplest version of the HPI extractable from an H&P or ED note,
    starting with the summary statement and **not** including the ED course,
    A&P, any SmartText'ed ED data or findings, medications, A&P, ROS, 
    PMH, Impression, or other similar sections of the note.

    This may include the Onc history or descriptions of recent hospitalizations
    if they are included before the ED course or other sections as listed above.

    For now, we focus on H&Ps rather than the ED notes. We also don't consider
    Consult notes, although they'd probably work OK as well.
    """
    name = "hpi_short"
    note_where_sql = """
        "InpatientNoteTypeDSC" IN ('H&P')
    """