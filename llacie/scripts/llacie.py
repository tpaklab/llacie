import os
import click
import collections
from ..app import App
from ..utils import echo_err

class OrderedGroup(click.Group):
    def __init__(self, name=None, commands=None, **attrs):
        super(OrderedGroup, self).__init__(name, commands, **attrs)
        self.commands = commands or collections.OrderedDict()

    def list_commands(self, ctx):
        return self.commands

@click.group(cls=OrderedGroup)
@click.pass_context
def main(ctx):
    """LLaCIE: Large Language model Clinical Information Extraction
    
    This is the main menu. Please use one of the subcommands listed below."""
    ctx.obj = ctx.with_resource(App())

@main.command()
@click.option('--overwrite/--no-overwrite', default=False,
    help="Forces the dropping and re-creation of tables if they already exist.")
@click.option('--missing-tables-only', default=False, is_flag=True,
    help="Create only missing tables, if possible; do NOT drop any existing tables.")
@click.pass_obj
def init_db(app, overwrite, missing_tables_only):
    """(0) Creates the database structure used by LLaCIE."""
    app.db.create_tables(overwrite, missing_tables_only)

@main.group(cls=OrderedGroup)
@click.pass_obj
def import_notes(app):
    """(1) Import notes from various sources."""
    pass

@import_notes.command(name="text")
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--overwrite/--no-overwrite', default=False,
    help="Permits the `notes` table to be truncated if it already contains rows.")
@click.pass_obj
def import_notes_text(app, input_path, overwrite):
    """Import notes from plain text files."""
    app.import_notes_text(input_path, overwrite)

@import_notes.command(name="edw")
@click.option('--overwrite/--no-overwrite', default=False,
    help="Permits the `notes` table to be truncated if it already contains rows.")
@click.option('--resume-note-text', default=False, is_flag=True,
    help="Resumes downloading of note text only, without rebuilding the table.")
@click.pass_obj
def import_notes_edw(app, overwrite, resume_note_text):
    """Import notes from the MGB EDW (Enterprise Data Warehouse)."""
    app.import_notes_edw(overwrite, resume_note_text)

@main.group(cls=OrderedGroup)
@click.pass_obj
def sections(app):
    """(2) Extract sections from each note."""
    pass

@sections.command(name="list")
@click.option('-n', '--section-name', default=None, type=str, 
    help="Filter to only strategies for one section name.")
@click.option('-s', '--strategy-name', default=None, type=str, 
    help="Filter to strategies matching this name. Partial matches using shell-style globs are allowed.")
@click.pass_obj
def list_sections(app, section_name=None, strategy_name=None):
    """List available sections, along with strategies for each."""
    app.list_sections_with_strategies(section_name, strategy_name)

@sections.command(name="extract")
@click.option('-n', '--section-name', default=None, type=str, 
    help="Specifies which section to extract.")
@click.option('-s', '--strategy-name', default=None, type=str, 
    help="Specifies which strategy to run. Partial matches using shell-style globs are allowed.")
@click.option('-d', '--dry-run', default=False, is_flag=True, 
    help="Prints how many notes would be processed and exits; the strategy is not run.")
@click.pass_obj
def extract_sections(app, section_name, strategy_name, dry_run):
    """Extract a section using a specified strategy."""
    app.run_section_strategy(section_name, strategy_name, dry_run)


@main.group(cls=OrderedGroup)
@click.pass_obj
def features(app):
    """(3) Extract features from each note."""
    pass

@features.command(name="list")
@click.option('-n', '--feature-name', default=None, type=str, 
    help="Filter to only strategies for one feature name.")
@click.option('-s', '--strategy-name', default=None, type=str, 
    help="Filter to strategies matching this name. Partial matches using shell-style globs are allowed.")
@click.pass_obj
def list_features(app, feature_name=None, strategy_name=None):
    """List available features, along with strategies for each."""
    app.list_features_with_strategies(feature_name, strategy_name)

@features.command(name="extract")
@click.option('-n', '--feature-name', default=None, type=str, 
    help="Filter to only strategies for one feature name.")
@click.option('-s', '--strategy-name', default=None, type=str, 
    help="Filter to strategies matching this name. Partial matches using shell-style globs are allowed.")
@click.option('-d', '--dry-run', default=False, is_flag=True, 
    help="Prints how many notes would be processed and exits; the strategy is not run.")
@click.option('-A', '--annotated-for-task', default=None, type=str, 
    help="Run only for notes that have human annotations for the given task name.")
@click.option('-M', '--max-note-ids', default=None, type=int, 
    help="Maximum number of notes to process in this invocation.")
@click.option('-R', '--redo-older-than', default=None, type=click.DateTime(), 
    help=("Any features extracted before this time will be considered stale "
        "and re-extracted."))
@click.option('--worker', is_flag=True, hidden=True, 
    help="For internal use only, used to run this subcommand within a bsub job.")
@click.argument('note_ids', type=int, nargs=-1)
@click.pass_obj
def extract_features(app, feature_name=None, strategy_name=None, dry_run=False, 
        annotated_for_task=None, max_note_ids=None, worker=False, redo_older_than=None, 
        note_ids=tuple()):
    """Extract features from notes using a specified strategy.
    
    The NOTE_IDS argument is for internal use only, when this is called as a worker 
    process, to run the strategy on specific rows in the database."""
    if worker:
        app.is_worker = True
        annotated_for_task = None
    elif len(note_ids) > 0:
        return echo_err("Error: Not allowed to specify NOTE_IDS when invoking this manually.")
    app.run_feature_strategy(
        feature_name, 
        strategy_name, 
        dry_run,
        annotated_for_task, 
        max_note_ids,
        redo_older_than,
        note_ids
    )


@main.group(cls=OrderedGroup)
@click.pass_obj
def episode_labels(app):
    """(4) Apply labels to episodes of care using the features."""
    pass

@episode_labels.command(name="list")
@click.option('-n', '--episode-label-task', default=None, type=str, 
    help="Filter to only strategies for one episode labelling task.")
@click.option('-s', '--strategy-name', default=None, type=str, 
    help="Filter to strategies matching this name. Partial matches using shell-style globs are allowed.")
@click.pass_obj
def list_episode_labels(app, episode_label_task=None, strategy_name=None):
    """List available episode labels, along with strategies for each."""
    app.list_episode_labels_with_strategies(episode_label_task, strategy_name)

@episode_labels.command(name="extract")
@click.option('-n', '--episode-label-task', default=None, type=str, 
    help="Filter to only strategies for episode labelling task name.")
@click.option('-s', '--strategy-name', default=None, type=str, 
    help="Filter to strategies matching this name. Partial matches using shell-style globs are allowed.")
@click.option('-d', '--dry-run', default=False, is_flag=True, 
    help="Prints how many episodes would be processed and exits; the strategy is not run.")
@click.option('-A', '--annotated-for-task', default=None, type=str, 
    help="Run only for episodes that have human annotations for the given task name.")
@click.pass_obj
def extract_episode_labels(app, episode_label_task=None, strategy_name=None, dry_run=False,
        annotated_for_task=None):
    """Create episode labels using a specified strategy."""
    app.run_episode_label_strategy(episode_label_task, strategy_name, dry_run, annotated_for_task)

@episode_labels.command(name="import")
@click.argument('episode_label_task', type=str)
@click.argument('input_xlsx', type=click.File('rb'))
@click.option('-w', '--sheet-name', default=None, type=str, 
    help="Specify the name of the worksheet in the XLSX that will be imported. "
    "Defaults to the first sheet.")
@click.option('-u', '--human-username', default=os.getenv('USER'), type=str, 
    help="Specify the username to whom imported annotations will be attributed. "
    "Defaults to the current user.")
@click.pass_obj
def import_episode_labels(app, episode_label_task, input_xlsx, sheet_name=None, 
        human_username=None):
    """Import human-annotated episode labels from an XLSX file.
    
    Expects an XLSX worksheet with a header row and two specific columns:

    - `FK_episode_id` - must correspond to a valid `id` in the episodes table

    - `human_labels` - the human-created label names, delimited by | (pipes)"""
    app.db.import_episode_labels(episode_label_task, input_xlsx, sheet_name, human_username)

@episode_labels.command(name="evaluate")
@click.option('-n', '--episode-label-task', default=None, type=str, 
    help="Filter to only strategies for this episode labelling task name.")
@click.option('-s', '--strategy-name', default=None, type=str, 
    help="Filter to strategies matching this name. Partial matches using shell-style globs are allowed.")
@click.option('-u', '--human-username', default=None, type=str, 
    help="Specify the username for whom all strategies' annotations will be compared. "
    "Defaults to the first available human annotation.")
@click.option('-o', '--other-human', default=None, type=str, 
    help="Specify another human's username, whose labels will be treated an alternate strategy, "
    "in effect evaluating inter-labeler reliability.")
@click.option('-l', '--also-labeled-by', default=None, type=str, 
    help="Specify another human's username, who must have labeled each of the episodes included "
    "for the evaluation. Can be different from -u.")
@click.option('-B', '--bootstrap-samples', default=1000, type=int, 
    help="For bootstrapped confidence intervals, specify the # of resamples. Default is 1000.")
@click.pass_obj
def evaluate_episode_labels(app, episode_label_task=None, strategy_name=None,
        human_username=None, other_human=None, also_labeled_by=None, bootstrap_samples=None):
    """Evaluate episode labelling strategies against human annotations."""
    app.evaluate_episode_label_strategies(episode_label_task, strategy_name, human_username,
        other_human, also_labeled_by, bootstrap_samples)