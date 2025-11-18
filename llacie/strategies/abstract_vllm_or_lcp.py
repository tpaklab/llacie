import sys
import json

from os import path
from textwrap import dedent
from socket import gethostname
from glob import glob

from .abstract import AbstractStrategy
from ..batch.apptainer_slurm import ApptainerSlurmJobManager
from ..cache.sqlite import SqliteFeatureWorkerCache
from ..inference.vllm import Vllm
from ..inference.llama_cpp_python import LlamaCppPython
from ..utils import echo_warn, echo_info, strtobool

class AbstractVllmOrLcpStrategy(AbstractStrategy):
    """
    Abstract class for running vllm or llama-cpp-python with constant parameters and a prespecified 
    prompt template, across all of the specified notes, substituting one section for each note into
    the prompt template to extract a feature. The backend (vllm vs llama-cpp-python) is configurable
    and all options are translated to match either backend wherever possible.

    The output is currently constrained to be a JSON array of strings. This could be overridden
    by inheriting classes, by redefining `self.JSON_SCHEMA`.
    
    Slurm may be used to schedule the jobs. On ERISXdl, we are required to use both slurm and 
    Apptainer containers to access the GPUs on the GPU nodes.

    This class is intended to be subclassed and configured *primarily* by overriding the class 
    constants, and beyond that, its methods might be overridden as well.
    """
    # can be, e.g., ShortHPISectionRegexStrategy from .section.hpi_short.regex
    SECTION_STRATEGY_CLASS = None             

    # These are example default options based on the Llama 3 8B strategy for presenting symptoms
    # This is a HuggingFace respository ID for the full PyTorch/Transformers model, which vllm uses
    MODEL_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'
    # As a fallback, you can also provide a GGUF version, which is what llama-cpp-python uses
    GGUF_MODEL_ID = 'QuantFactory/Meta-Llama-3-8B-Instruct-GGUF'
    GGUF_MODEL_FILENAME = '*.Q5_K_S.gguf'

    SLURM_ENABLED = True
    SLURM_THREADS_PER_JOB = 10
    ARRAY_OF_STRINGS_JSON_SCHEMA = """
    {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "array",
        "items": {
            "type": "string"
        }
    }
    """
    JSON_SCHEMA = ARRAY_OF_STRINGS_JSON_SCHEMA

    # This flag and config var allows this class to toggle between vllm and llama-cpp-python backends
    VLLM_ENABLED = False
    # Need to set vllm's LLM(..., dtype='half') because the Nvidia V100s in the ERISXdl cluster do 
    # not natively support bfloat16. 
    # This is expected to trigger a warning: "Casting torch.bfloat16 to torch.float16"
    VLLM_DTYPE = 'half'

    # The maximum number of output tokens you'd ever expect to generate per prompt
    LLM_TOKENS_TO_GENERATE = 256
    # Arguments to SamplingParams in vllm
    # See: https://docs.vllm.ai/en/latest/api/vllm/#vllm.SamplingParams
    # These are translated into equivalent arguments for llama-cpp-python's create_chat_completion
    # https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion
    SAMPLING_PARAMS = {
        "temperature": 0.1,
        "top_p": 0.8,
        "repetition_penalty": 1.05
    }

    NICE_MODEL_NAME = "Llama3 8B Instruct"

    # Model context length. Note vllm cannot generate past the end of the context window.
    LLM_MAX_MODEL_LENGTH = 2048  

    # Prompts
    LLM_SYSTEM_PROMPT = dedent("""\
        You are a clinical researcher that reads medical charts and answers questions about them. 
        Use only the information in the text provided to answer the question.
        If a patient denies something, do not include it in your answer.
        After you provide an answer, you immediately stop talking.""")
    LLM_USER_PROMPT = dedent("""\
        Read the following patient history and list the patient's presenting symptoms.
        Include only symptoms present now or reported for the days to weeks leading up to admission.
        Ignore any symptoms from past medical history or prior hospital admissions.
        Give your answer as a JSON array containing up to ten strings.
        Each string contains between one and three words.
        
        {input}""")


    def __init__(self, db, config, **options):
        super().__init__(db, config, **options)
        # Ensure this config option exists because ApptainerSlurmJobManager's parent class needs it
        self.config['SLURM_THREADS_PER_JOB'] = config.get("SLURM_THREADS_PER_JOB",
            self.SLURM_THREADS_PER_JOB)
        vllm_enabled = strtobool(config.get("VLLM_ENABLED", self.VLLM_ENABLED))
        self._dtype = config.get("VLLM_DTYPE", self.VLLM_DTYPE)
        self.backend = "vllm" if vllm_enabled else "llama-cpp-python"
        
        if not isinstance(self.SECTION_STRATEGY_CLASS, type):
            raise NotImplementedError("You must specify a SECTION_STRATEGY_CLASS")

        self.sec_strat = self.SECTION_STRATEGY_CLASS(db, config, **options)
        self.sec_task_name = self.sec_strat.task.name

        self._set_model_paths(config)

        self._sampling_params = self._translate_sampling_params({
            **self.SAMPLING_PARAMS, 
            "max_tokens": self.LLM_TOKENS_TO_GENERATE
        })
        self._engine_args = self._create_engine_args()

        self.slurm_enabled = strtobool(config.get("SLURM_ENABLED", self.SLURM_ENABLED))
        self.workers_have_networking = config.get("APPTAINER_WORKERS_HAVE_NETWORKING", 
            ApptainerSlurmJobManager.APPTAINER_WORKERS_HAVE_NETWORKING)

        if self.slurm_enabled and not self.workers_have_networking:
            self.worker_cache = SqliteFeatureWorkerCache(db, config, self)


    def _set_model_paths(self, config):
        self.local_models_path = config.get("LOCAL_MODELS_PATH", None)
        self.model_path = self.MODEL_ID if self.backend == "vllm" else self.GGUF_MODEL_ID
        self.tokenizer_model_path = self.MODEL_ID
        if self.local_models_path: 
            self.model_path = f"{self.local_models_path}/{self.model_path}"
            self.tokenizer_model_path = f"{self.local_models_path}/{self.tokenizer_model_path}"
            if self.backend == "llama-cpp-python":
                self.model_path = f"{self.model_path}/{self.GGUF_MODEL_FILENAME}"
                try: self.model_path = glob(self.model_path)[0]
                except IndexError: raise RuntimeError(f"Could not find GGUF: {self.model_path}")


    def _translate_sampling_params(self, params):
        """Converts sampling params from vllm's SamplingParams keys to the arguments used
        by create_chat_completion in llama-cpp-python. Most are exactly the same, but this fixes
        the few key exceptions. See:
        https://docs.vllm.ai/en/latest/api/vllm/#vllm.SamplingParams
        https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion"""
        if self.backend == "vllm": return params
        if "repetition_penalty" in params: 
            params["repeat_penalty"] = params.pop("repetition_penalty")
        return params


    def _create_engine_args(self):
        args = {}
        if self.backend == "vllm":
            args["dtype"] = self._dtype
            args["max_model_len"] = self.LLM_MAX_MODEL_LENGTH
        else: # self.backend == "llama-cpp-python"
            args["tokenizer_model_path"] = self.tokenizer_model_path
            args["verbose"] = False
            args["n_ctx"] = self.LLM_MAX_MODEL_LENGTH
            if not self.local_models_path: 
                args["filename"] = self.GGUF_MODEL_FILENAME
        return args


    def create_engine(self):
        engine = Vllm if self.backend == "vllm" else LlamaCppPython
        return engine(self.config, self.model_path, self.LLM_SYSTEM_PROMPT, self.LLM_USER_PROMPT,
            self.JSON_SCHEMA, **self._engine_args)


    def _run_llm(self, note_ids):
        if self.within_worker and not self.workers_have_networking: db = self.worker_cache
        else: db = self.db

        llm = self.create_engine()
        max_prompt_length = self.LLM_MAX_MODEL_LENGTH - self.LLM_TOKENS_TO_GENERATE - 4

        self.start_timer()

        prompts = []
        section_ids = []
        
        for note_id in note_ids:
            # Get the note section to work on
            row = db.get_note_section(note_id, self.sec_strat)
            if row is None: 
                raise RuntimeError(f"No {self.sec_task_name} section found for note id {note_id}")
            section_value = row.section_value

            # Ensure the prompt will fit in the context window, with 4 tokens of wiggle room
            full_prompt = llm.autotrim_prompt(max_prompt_length, input=section_value)
            if full_prompt is None:
                echo_warn(f"Autotrimming failed for note id {note_id}")
                # Couldn't autotrim; likely a malformed section, but try using the whole thing
                full_prompt = llm.create_prompt(input=section_value)
            
            section_ids.append(row.id)
            prompts.append(full_prompt)
            
        # Run all the prompts through the LLM backend
        # Outputs are already deserialized from JSON into Python lists or dicts
        echo_info(f"Running {len(note_ids)} notes thru the {self.NICE_MODEL_NAME} model "
            f"using the {self.backend} backend")
        outputs = llm(prompts, **self._sampling_params)
        runtime = self.stop_timer() / len(outputs)

        for i, output in enumerate(outputs):
            note_id = note_ids[i]
            section_id = section_ids[i]
            if output is not None and isinstance(output, list): 
                feature = "\n".join(output)
                output_raw = json.dumps(output)
                db.upsert_note_feature(note_id, self, output_raw, feature, section_id, runtime)
            else:
                echo_warn(f"No LLM output generated for note id {note_id}")


    def run(self, all_note_ids, max_note_ids=None, batch_size=None, dry_run=False):
        if self.within_worker:
            # We are within a worker process; run the LLM on the given chunk of note IDs
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
                worker_cmd = sys.argv.copy()
                # We want a relative path to the `llacie` executable, because the worker processes
                #   may be running in a different virtualenv than this one
                worker_cmd[0] = path.basename(worker_cmd[0])
                # To start a worker, you invoke the same command with `--worker` appended
                worker_cmd.append("--worker")
                job_manager = ApptainerSlurmJobManager(worker_cmd, self.config, self.worker_cache)
                return job_manager.run(note_ids)
        
        # Either slurm is disabled or we are inside one the worker processes
        # In this case, we should start running vLLM on the note IDs
        self._run_llm(note_ids)

        # For worker processes without networking, we need to mark the results on disk as complete
        if self.within_worker and not self.workers_have_networking: self.worker_cache.finalize()