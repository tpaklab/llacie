import json

from textwrap import dedent

from .abstract import AbstractStrategy
from ..inference.uci_health_server import UCI_HEALTH
from ..utils import echo_warn, echo_info


class AbstractUciHealthStrategy(AbstractStrategy):
    """
    Abstract class for running inference against the UCI Health Server (OpenAI-compatible API)
    with a prespecified prompt template, across all of the specified notes, substituting one
    section for each note into the prompt template to extract a feature.

    Unlike the vllm/llama-cpp-python backends, this uses a remote HTTP API, so no GPU, Slurm,
    or local model paths are needed.
    """
    SECTION_STRATEGY_CLASS = None

    NICE_MODEL_NAME = "GPT OSS 120B"

    LLM_SYSTEM_PROMPT = dedent("""\
        You are a clinical researcher that reads medical charts and answers questions about them.
        Use only the information in the text provided to answer the question.
        If a patient denies something, do not include it in your answer.
        After you provide an answer, you immediately stop talking.""")
    LLM_USER_PROMPT = dedent("""\
        Read the following patient history and list the patient's presenting symptoms.
        Give your answer as a JSON array containing up to ten strings.
        Each string contains between one and three words.

        {input}""")

    SAMPLING_PARAMS = {
        "temperature": 0.5,
        "reasoning_effort": "high",
        "seed": 0
    }


    def __init__(self, db, config, **options):
        super().__init__(db, config)

        if not isinstance(self.SECTION_STRATEGY_CLASS, type):
            raise NotImplementedError("You must specify a SECTION_STRATEGY_CLASS")

        self.sec_strat = self.SECTION_STRATEGY_CLASS(db, config, **options)
        self.sec_task_name = self.sec_strat.task.name


    def create_engine(self):
        llm = UCI_HEALTH()
        llm.load_config()
        llm.SYSTEM_PROMPT = self.LLM_SYSTEM_PROMPT
        return llm


    def _run_llm(self, note_ids):
        llm = self.create_engine()

        echo_info(f"Running {len(note_ids)} notes thru the {self.NICE_MODEL_NAME} model "
            f"using the uci_health_server backend")

        for note_id in note_ids:
            row = self.db.get_note_section(note_id, self.sec_strat)
            if row is None:
                raise RuntimeError(f"No {self.sec_task_name} section found for note id {note_id}")

            user_message = self.LLM_USER_PROMPT.format(input=row.section_value)

            self.start_timer()
            response, _ = llm.create_query(user_message, **self.SAMPLING_PARAMS)
            runtime = self.stop_timer()

            try:
                output = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                echo_warn(f"Could not parse JSON response for note id {note_id}")
                output = None

            if output is not None and isinstance(output, list):
                feature = "\n".join(output) if output and isinstance(output[0], str) else json.dumps(output)
                output_raw = json.dumps(output)
                self.db.upsert_note_feature(note_id, self, output_raw, feature, row.id, runtime)
            else:
                echo_warn(f"No valid LLM output for note id {note_id}")


    def run(self, all_note_ids, max_note_ids=None, batch_size=None, dry_run=False):
        note_ids = self.db.filter_to_notes_with_section(all_note_ids, self.sec_strat)
        if (num_filtered := len(all_note_ids) - len(note_ids)) > 0:
            echo_warn(f"Skipping {num_filtered} notes without an {self.sec_task_name} section")
        if max_note_ids is not None and len(note_ids) > max_note_ids:
            echo_warn(f"Limiting to the first {max_note_ids} notes")
            note_ids = note_ids[:max_note_ids]
        if dry_run:
            return echo_info("Info: Dry run, exiting without running any LLMs")

        self._run_llm(note_ids)
