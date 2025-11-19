from abc import ABC, abstractmethod
from enum import Enum
from textwrap import dedent

class TaskOutputType(Enum):
    # Constants for the type of task
    # Section tasks divide a note into sections like HPI, Exam, A&P etc.
    SECTION = "section"
    # Feature tasks extract certain pieces of unstructured data from a note, like symptoms
    FEATURE = "feature"
    # Label tasks attempt to condense multiple features per note, patient, or episode
    #   into an overall categorization for the same item
    NOTE_LABEL = "note_label"
    EPISODE_LABEL = "episode_label"
    PATIENT_LABEL = "patient_label"


class AbstractTask(ABC):
    def __init__(self):
        raise RuntimeError('Task classes are not meant to be instantiated')

    @classmethod
    @property
    @abstractmethod
    def output_type(cls): raise NotImplementedError

    @classmethod
    def desc(cls): return dedent(cls.__doc__)

    @classmethod
    def as_row(cls):
        return {
            "output_type": cls.output_type.value,
            "name": cls.name,
            "desc": cls.desc()
        }


class AbstractSectionTask(AbstractTask):
    output_type = TaskOutputType.SECTION

    @classmethod
    @property
    @abstractmethod
    def name(cls): raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def note_where_sql(cls): raise NotImplementedError


class AbstractFeatureTask(AbstractTask):
    output_type = TaskOutputType.FEATURE

    @classmethod
    @property
    @abstractmethod
    def name(cls): raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def note_join_sql(cls): raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def note_where_sql(cls): raise NotImplementedError


class AbstractNoteLabelTask(AbstractTask):
    output_type = TaskOutputType.NOTE_LABEL
    vocab = None
    max_human_labels = 1

    @classmethod
    @property
    @abstractmethod
    def name(cls): raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def note_join_sql(cls): raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def note_where_sql(cls): raise NotImplementedError


class AbstractEpisodeLabelTask(AbstractTask):
    output_type = TaskOutputType.EPISODE_LABEL
    vocab = None
    max_human_labels = 1

    @classmethod
    @property
    @abstractmethod
    def name(cls): raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def episode_join_sql(cls): raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def episode_where_sql(cls): raise NotImplementedError

    # Used to find the relevant notes for an episode when creating human annotations
    @classmethod
    @property
    @abstractmethod
    def note_where_sql(cls): raise NotImplementedError


class AbstractPatientLabelTask(AbstractTask):
    output_type = TaskOutputType.PATIENT_LABEL
    vocab = None
    max_human_labels = 1

    @classmethod
    @property
    @abstractmethod
    def name(cls): raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def patient_join_sql(cls): raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def patient_where_sql(cls): raise NotImplementedError

    # Used to find the relevant notes for a patient when creating human annotations
    @classmethod
    @property
    @abstractmethod
    def note_where_sql(cls): raise NotImplementedError
