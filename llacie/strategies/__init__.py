from os import path
from glob import glob
from fnmatch import fnmatch
from importlib import import_module
from inspect import getmembers

from .abstract import AbstractStrategy
from ..tasks.abstract import TaskOutputType as out_t, AbstractTask

class StrategyFinder:
    _index = None
    GLOB = '*/*/*.py'
    BASE_PATH = path.dirname(__file__)

    def _create_index(self):
        """Creates an index of the modules/classes within this package
        for use by the `find_strategies` method below."""
        self._index = []
        already_imported = set()

        for py_file in glob(path.join(self.BASE_PATH, self.GLOB)):
            if path.basename(py_file).startswith('__'): continue

            py_file = py_file[len(self.BASE_PATH):]
            module_name = path.splitext(py_file)[0].replace(path.sep, '.')
            mod = import_module(module_name, package=__name__)

            for name, obj in getmembers(mod):
                if name.startswith('_'): continue
                if obj is AbstractStrategy or name.startswith('Abstract'): continue
                if obj in already_imported: continue
                if not isinstance(obj, type) or not issubclass(obj, AbstractStrategy):
                    continue

                already_imported.add(obj)
                self._index.append({
                    "output_type": obj.task.output_type,
                    "task": obj.task, 
                    "task_name": obj.task.name,
                    "strategy_name": obj.name,
                    "class_name": name,
                    "strategy": obj
                })

    @property
    def index(self):
        """Lazily create the indexes, once, upon first access"""
        if self._index is not None: return self._index
        self._create_index()
        return self._index
    
    def name_glob_match(self, full_name, name_glob):
        if name_glob.count(".") < 2: name_glob = f"*.{name_glob}"
        return fnmatch(full_name, name_glob)
    
    def find_strategies(self, output_type=None, task=None, name_glob=None):
        strategies = []

        for row in self.index:
            if name_glob is not None: 
                if not self.name_glob_match(row["strategy_name"], name_glob): continue

            if output_type is not None:
                if isinstance(output_type, str):
                    if row["output_type"].value != output_type: continue
                else:
                    if not isinstance(output_type, out_t):
                        raise RuntimeError("output_type must be `str` or `TaskOutputType`")
                    if output_type is not row["output_type"]: continue

            if task is not None:
                if isinstance(task, str):
                    if row["task_name"] != task: continue
                else:
                    if not issubclass(task, AbstractTask):
                        raise RuntimeError("task must be `str` or subclass of `AbstractTask`")
                    if row["task"] is not task: continue

            strategies.append(
                (row["output_type"], row["task_name"], row["strategy_name"], row["strategy"]))

        strategies = [row[3] for row in sorted(strategies, key = lambda row: row[0:3])]
        return strategies


# We create one global instance of the above class
_finder = StrategyFinder()

# Export this prebound method
find_strategies = _finder.find_strategies