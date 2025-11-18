import sys
import json
import re
from os import path
from textwrap import dedent
from socket import gethostname

from .abstract import AbstractStrategy
from ..batch.slurm import SlurmJobManager
from ..inference.llama_cpp import LlamaCpp
from ..utils import PACKAGE_DIR, echo_warn, echo_info, strtobool

class AbstractLlamaCppStrategy(AbstractStrategy):
    """
    Abstract class for running llama.cpp with constant command-line parameters and with a 
    prespecified prompt template, across all of the specified notes, substituting one section
    for each note into the prompt template.

    The output is currently constrained to be a JSON array of strings.
    
    Supports the use of Slurm to schedule jobs across a cluster when SLURM_ENABLED = True.

    This class is intended to be subclassed and configured primarily by overriding the class 
    constants, and beyond that, methods could be overridden as well.
    """
    # can be, e.g., ShortHPISectionRegexStrategy from .section.hpi_short.regex
    SECTION_STRATEGY_CLASS = None

    SLURM_ENABLED = True

    # These are example default options based on the Llama 3 8B strategy for presenting symptoms
    LLAMA_CPP_BIN = 'main-e849648'            # which llama.cpp `main` executable to use
    LLAMA_CPP_THREADS = 16
    SLURM_THREADS_PER_JOB = 16
    LLAMA_MODEL_PATH = 'models/llama3/Meta-Llama-3-8B-Instruct-bf16.gguf'
    GRAMMAR_FILE = path.join(PACKAGE_DIR, 'grammars/json_arr_of_strings.gbnf')
    LLM_MAX_TOKENS = 2044
    LLAMA_CPP_OPTS = (f"-m {LLAMA_MODEL_PATH} -n 256 -c {LLM_MAX_TOKENS + 4} "
        f"--grammar-file {GRAMMAR_FILE} -r '<|eot_id|>'")
    TOKENIZER_MODEL_PATH = 'models/llama3'    # For tokenized input length estimation only.
    TOKENIZER_HF_MODEL_ID = None              # Either set this or TOKENIZER_MODEL_PATH.
    LLAMA_VOCAB_PATH = None                   # Should be None for everything except Llama 1
    NICE_MODEL_NAME = "Llama3 8B Instruct (bf16)"
    ALT_BOS_TOKEN = '<|begin_of_text|>'
    TRIM_EOT_TOKEN_REGEX = r'<[|]eot_id[|]>$' # If not None, used to trim end of LLM output

    LLM_PROMPT = dedent("""\
        <|start_header_id|>system<|end_header_id|>
        
        You are a helpful assistant that reads medical charts and answers questions about them. 
        Use only the information in the text provided to answer the question.
        If a patient denies something, do not include it in your answer.
        After you provide an answer, you immediately stop talking.
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        
        Read the following patient history and list the patient's presenting symptoms.
        Include only symptoms present now or within the past few days.
        Ignore any symptoms from past medical history or prior healthcare encounters.
        Give your answer as a JSON array containing up to ten strings.
        Each string contains between one and three words.

        {input}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
        """)


    def __init__(self, db, config, **options):
        super().__init__(db, config, **options)
        # Ensure this config option exists because LlamaCpp() needs it
        self.config['LLAMA_CPP_THREADS'] = config.get("LLAMA_CPP_THREADS", self.LLAMA_CPP_THREADS)
        # Ensure this config option exists because SlurmJobManager() uses it
        self.config['SLURM_THREADS_PER_JOB'] = config.get("SLURM_THREADS_PER_JOB",
            self.SLURM_THREADS_PER_JOB)
        
        if not isinstance(self.SECTION_STRATEGY_CLASS, type):
            raise NotImplementedError("You must specify a SECTION_STRATEGY_CLASS")

        self.sec_strat = self.SECTION_STRATEGY_CLASS(db, config, **options)
        self.sec_task_name = self.sec_strat.task.name
        self.slurm_enabled = strtobool(config.get("SLURM_ENABLED", self.SLURM_ENABLED))


    def _run_llm(self, note_ids):
        llm = LlamaCpp(
            self.config,
            self.LLAMA_CPP_OPTS,
            self.LLAMA_CPP_BIN, 
            vocab_path = self.LLAMA_VOCAB_PATH,
            hf_model_id = self.TOKENIZER_HF_MODEL_ID,
            tokenizer_model_path = self.TOKENIZER_MODEL_PATH,
            alt_bos_token = self.ALT_BOS_TOKEN
        )

        for note_id in note_ids:
            # Get the note section to work on
            row = self.db.get_note_section(note_id, self.sec_strat)
            if row is None: 
                raise RuntimeError(f"No {self.sec_task_name} section found for note id {note_id}")
            section_value = row.section_value
            self.start_timer()

            # Ensure the prompt will fit in the context window
            full_prompt = llm.autotrim_prompt(section_value, self.LLM_PROMPT, self.LLM_MAX_TOKENS)
            if full_prompt is None:
                echo_warn(f"Autotrimming failed for note id {note_id}")
                # Couldn't autotrim; likely a malformed section, but try using the whole thing
                full_prompt = self.LLM_PROMPT.format(input = section_value)
            
            # Run the prompt through llama.cpp
            echo_info(f"Running note id {note_id} thru the {self.NICE_MODEL_NAME} model")
            llm_output_raw = llm(full_prompt)
            if self.TRIM_EOT_TOKEN_REGEX is not None:
                llm_output_raw = re.sub(self.TRIM_EOT_TOKEN_REGEX, "", llm_output_raw)

            if llm_output_raw is not None and len(llm_output_raw) > 0: 
                # Cleanup and parse the output, and save everything back to postgres.
                # Since we constrained output to be a JSON array of strings, parsing is simple
                try:
                    parsed = json.loads(llm_output_raw)
                    feature = "\n".join(parsed)
                except (json.decoder.JSONDecodeError, TypeError):
                    echo_warn(f"Could not decode LLM output as JSON for note id {note_id}:")
                    echo_warn(llm_output_raw)
                else:
                    self.db.upsert_note_feature(note_id, self, llm_output_raw, feature, row.id)
            else:
                echo_warn(f"No LLM output generated for note id {note_id}")


    def run(self, all_note_ids, max_note_ids=None, batch_size=None, dry_run=False):
        if self.within_worker:
            if dry_run: return echo_info("Info: Dry run, exiting without running any LLMs")
            note_ids = all_note_ids
            echo_info(f"Worker started on {gethostname()}")
        else:
            # There is no point running this on notes without the required section
            note_ids = self.db.filter_to_notes_with_section(all_note_ids, self.sec_strat)
            if (num_filtered := len(all_note_ids) - len(note_ids)) > 0:
                echo_warn(f"Skipping {num_filtered} notes without an {self.sec_task_name} section")
            if max_note_ids is not None and (len(note_ids) > max_note_ids):
                echo_warn(f"Limiting to the first {max_note_ids} notes")
                note_ids = note_ids[:max_note_ids]
            if dry_run: return echo_info("Info: Dry run, exiting without running any LLMs")
            
            if self.slurm_enabled:
                # Create a SlurmJobManager and submit jobs that run worker processes
                # To start a worker, you invoke the same command with `--worker` appended
                worker_cmd = sys.argv
                worker_cmd.append("--worker")
                job_manager = SlurmJobManager(worker_cmd, self.config)
                return job_manager.run(note_ids)

        # Either slurm is disabled or this is one of the worker processes.
        # Proceed to running llama.cpp
        self._run_llm(note_ids)