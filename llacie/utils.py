import random
from os import path
from click import echo, secho, get_current_context
from collections import namedtuple

PACKAGE_DIR = path.dirname(__file__)

def chunker(seq, size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def _force_color_in_workers():
    try:
        from llacie import app
        ctx = get_current_context()
        current_app = ctx.find_object(app.App)
        if current_app is None: return None
        return current_app.is_worker or None
    except RuntimeError:
        pass
    return None

def echo_err(message):
    secho(message, fg='red', err=True, color=_force_color_in_workers())

def echo_warn(message):
    secho(message, fg='yellow', err=True, color=_force_color_in_workers())

def echo_info(message):
    secho(message, fg='green', err=True, color=_force_color_in_workers())

def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    if isinstance(val, bool): return val
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))

def create_row_like(**kwargs):
    """Create a row-like object with positional and attribute access"""
    RowLike = namedtuple('RowLike', kwargs.keys())
    return RowLike(**kwargs)

def generate_unique_id(existing_ids, digits=10):
    """Generate a large random integer not in `existing_ids`. Each integer is `digits` long.
    Returns the new ID and also adds it to `existing_ids`."""
    while True:
        new_id = random.randint(10 ** digits, 10 ** (digits + 1) - 1)
        if new_id not in existing_ids:
            existing_ids.add(new_id)
            return new_id

def echo_strategies(strategies, title="Sections"):
    last_task_name = None
    echo(f"\n{(title + ':').ljust(24)} Available strategies:")
    echo("════════════════════════ ════════════════════════════════════════")
    for i, strat in enumerate(strategies):
        next_task_name = strategies[i + 1].task.name if i + 1 < len(strategies) else None
        if strat.task.name == last_task_name:
            sep = "  └─" if next_task_name != strat.task.name else "  ├─"
            task_formatted = (" " * 20)
        else: 
            sep = "◁───" if next_task_name != strat.task.name else "◁─┬─"
            task_formatted = strat.task.name.ljust(20)
        echo(f"    {task_formatted} {sep} {strat.name}")
        last_task_name = strat.task.name
    echo("")

def only_one_strategy(strategies, title="Sections"):
    if len(strategies) > 1:
        echo("Error: Your parameters matched several strategies. Here's what matched:")
        echo_strategies(strategies, title="Sections")
        echo("Use the --strategy-name option to select a single strategy.")
        return False
    elif len(strategies) == 0:
        echo("Error: No strategies matching those parameters were found.")
        return False
    return True
    