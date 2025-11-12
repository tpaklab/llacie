import time
from abc import ABC, abstractmethod
from textwrap import dedent
from copy import copy

class AbstractStrategy(ABC):
    def __init__(self, db, config):
        # Strategies are persisted lazily into the database
        # See `get_or_register_task()` and `get_or_register_strategy()` in `LlacieDatabase`
        self.db = db
        # Strategies create a **copy** of the config passed into them to prevent
        # any modifications from propagating upwards (e.g., into the App)
        self.config = copy(config)
        self._start_time = None
        self._within_worker = config['LLACIE_WORKER']

    @property
    @abstractmethod
    def task(self): raise NotImplementedError

    @property
    @abstractmethod
    def name(self): raise NotImplementedError

    @property
    @abstractmethod
    def version(self): raise NotImplementedError

    @property
    @abstractmethod
    def prereq_tasks(self): raise NotImplementedError

    @abstractmethod
    def run(self, ids): raise NotImplementedError

    @property
    def desc(self): return dedent(self.__doc__)

    @property
    def db_row(self): return self.db.get_or_register_strategy(self)

    @property
    def id(self): return self.db_row.id

    @property
    def task_id(self): return self.db.get_or_register_task(self.task).id

    @property
    def within_worker(self): return self._within_worker

    def as_row(self):
        """Fields that get saved to the database"""
        task_row = self.db.get_or_register_task(self.task)
        return {
            "FK_task_id": task_row.id,
            "name": self.name,
            "version": self.version,
            "desc": self.desc
        }

    # TODO: method for checking if the Strategy's prereqs are all complete

    def get_unfinished_ids(self, newer_than=None):
        return self.db.get_unfinished_ids_for_strategy_or_task(self, newer_than=newer_than)
    
    # Methods for timing tasks within runs of a strategy
    # Returns an interval, in seconds, as a float value
    def start_timer(self):
        self._start_time = time.perf_counter()

    def stop_timer(self):
        if self._start_time is None: raise RuntimeError("Strategy run timer wasn't started!")
        runtime = time.perf_counter() - self._start_time
        self._start_time = None
        return runtime