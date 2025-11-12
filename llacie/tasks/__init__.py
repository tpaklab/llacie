from os import path
from glob import glob
from importlib import import_module
from inspect import getmembers

from .abstract import TaskOutputType, AbstractTask

class TaskFinder:
    _index = None
    GLOB = '*.py'
    BASE_PATH = path.dirname(__file__)

    def _create_index(self):
        """Creates an index of the modules/classes within this package
        for use by the `find_tasks` method below."""
        self._index = []
        already_imported = set()

        for py_file in glob(path.join(self.BASE_PATH, self.GLOB)):
            if path.basename(py_file).startswith('__'): continue
            if path.basename(py_file).startswith('abstract'): continue

            py_file = py_file[len(self.BASE_PATH):]
            module_name = path.splitext(py_file)[0].replace(path.sep, '.')
            mod = import_module(module_name, package=__name__)

            for name, obj in getmembers(mod):
                if name.startswith('_') or name.startswith('Abstract'): continue
                if obj in already_imported: continue
                if not isinstance(obj, type) or not issubclass(obj, AbstractTask):
                    continue

                already_imported.add(obj)
                self._index.append({
                    "output_type": obj.output_type,
                    "task": obj, 
                    "task_name": obj.name,
                    "class_name": name,
                })

    # Lazily create the indexes, once, upon first access
    @property
    def index(self):
        if self._index is not None: return self._index
        self._create_index()
        return self._index
    
    def find_tasks(self, name=None, output_type=None):
        tasks = []

        for row in self.index:
            if name is not None and row["task_name"] != name: continue

            if output_type is not None:
                if isinstance(output_type, str):
                    if row["output_type"].value != output_type: continue
                else:
                    if not isinstance(output_type, TaskOutputType):
                        raise RuntimeError("output_type must be `str` or `TaskOutputType`")
                    if output_type is not row["output_type"]: continue

            tasks.append((row["task_name"], row["class_name"], row["task"]))

        tasks = [row[2] for row in sorted(tasks, key = lambda row: row[0:1])]
        return tasks


# We create one global instance of the above class
_finder = TaskFinder()

# Export this prebound method
find_tasks = _finder.find_tasks