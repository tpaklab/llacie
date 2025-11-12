from abc import ABC, abstractmethod
from copy import copy

class AbstractWorkerCache(ABC):
    """
    The WorkerCache classes implement intermediate data structures used to store inputs and outputs 
    for worker processes during long running processes.
    
    These were invented primarily for the AbstractVllmStrategy pathway of creating features
    using `vllm` on GPU nodes in ERISXdl. These nodes don't have any networking capabilities when
    running, except for the ability to read and write from a shared filesystem. Therefore,
    we can't talk directly to the LlacieDatabase from those nodes, and need an intermediary.
    """
    def __init__(self, upstream_db, config, **options):
        self.upstream_db = upstream_db
        # Caches create a **copy** of the config passed into them to prevent
        # any modifications from propagating upwards (e.g., into the App)
        self.config = copy(config)

    @abstractmethod
    def rename(self): raise NotImplementedError

    @abstractmethod
    def finalize(self): raise NotImplementedError

    @abstractmethod
    def sweep_into_upstream_db(self): raise NotImplementedError


class AbstractFeatureWorkerCache(AbstractWorkerCache):
    """
    These are the methods expected for a WorkerCache that would support generating a feature from
    note sections.
    """

    ### These methods are used by the worker process

    @abstractmethod
    def get_note_section(self): raise NotImplementedError

    @abstractmethod
    def upsert_note_feature(self): raise NotImplementedError

    ### These methods are used by the parent process, which has access to `self.upstream_db`

    @abstractmethod
    def cache_note_sections(self): raise NotImplementedError