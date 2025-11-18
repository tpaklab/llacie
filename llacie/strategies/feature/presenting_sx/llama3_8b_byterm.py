import json
from textwrap import dedent
from collections import defaultdict

from ...abstract_vllm_or_lcp import AbstractVllmOrLcpStrategy
from ...section.hpi_short.regex import ShortHPISectionRegexStrategy
from ....tasks.feature import PresentingSymptomsFeatureTask
from ....vocab import Vocab
from ....utils import chunker, echo_info, echo_warn


class PresentingSxFeatureLlama3Instruct8BByTermStrategy(AbstractVllmOrLcpStrategy):
    """\
    Attempts to extract the `presenting_sx` feature using Llama 3 Instruct 8B model,
    an 8B parameter model in the Llama family, and with the prompt including terms in the vocab.

    This is modified from the `feature.presenting_sx.llama3_8b` strategy by using multiple 
    prompts to ask about every symptom in a vocab, ~5 terms at a time. The downside is that many 
    more prompts (and tokens) are required per note, but there is a potential upside in accuracy
    since the task given to the LLM is more constrained and .
    """
    task = PresentingSymptomsFeatureTask
    name = "feature.presenting_sx.llama3_8b_byterm"
    version = "0.0.1"
    prereq_tasks = []

    SECTION_STRATEGY_CLASS = ShortHPISectionRegexStrategy

    # HuggingFace respository ID for the full PyTorch/Transformers model, which vllm uses
    MODEL_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'
    # As a fallback, you can also provide a GGUF version, which is what llama-cpp-python uses
    GGUF_MODEL_ID = 'QuantFactory/Meta-Llama-3-8B-Instruct-GGUF'
    GGUF_MODEL_FILENAME = '*.Q5_K_S.gguf'
    
    SLURM_IDS_PER_JOB = 50
    SLURM_THREADS_PER_JOB = 16
    OBJECT_WITH_FIVE_PROPERTIES_JSON_SCHEMA = """
    {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": {
            "type": "boolean"
        },
        "minProperties": 1,
        "maxProperties": 5
    }
    """
    JSON_SCHEMA = OBJECT_WITH_FIVE_PROPERTIES_JSON_SCHEMA

    VLLM_ENABLED = False
    VLLM_DTYPE = 'half'
    LLM_TOKENS_TO_GENERATE = 1024
    # Arguments to SamplingParams in vllm
    # See: https://docs.vllm.ai/en/latest/api/vllm/#vllm.SamplingParams
    # These are translated into equivalent arguments for llama-cpp-python's create_chat_completion
    SAMPLING_PARAMS = {
        "temperature": 0.1,
        "top_p": 0.8,
        "repetition_penalty": 1.05
    }

    NICE_MODEL_NAME = "Llama3 8B Instruct **ByTerm**"

    # Model context length. Note that Vllm will not generate past the end of the context window.
    LLM_MAX_MODEL_LENGTH = 3044  
    LLM_SYSTEM_PROMPT = dedent("""\
        You are a clinical researcher that reads medical charts and answers questions about them. 
        Use only the information in the text provided to answer the question.
        After you provide an answer, you immediately stop talking.""")
    LLM_USER_PROMPT = dedent("""\
        Read the following patient history and figure out which of the following presenting symptoms
        are reported for this patient: {symptoms}.
        Include only symptoms present now or reported for the days to weeks leading up to admission.
        Ignore any symptoms from past medical history or prior hospital admissions.
        Give your answer as a JSON object with the symptoms as the keys and true/false as the 
        values. For example, if all symptoms are absent, the answer would be: {symptoms_absent_json}
        
        Now, here is the patient history:
        
        {input}""")
    
    # Specify a vocab to draw terms from, and how many terms to use at a time in the prompt
    VOCAB_XLSX = "dt.pres_sx.ngrams.v2.xlsx"
    VOCAB_SHEET_NAME = "symptom_ngrams_top75ile.edited"
    TERMS_PER_PROMPT = 5


    def __init__(self, db, config, **options):
        super().__init__(db, config, **options)
        # We decrease the default number of note IDs per Slurm job, since each note now generates
        # hundreds of prompts to the LLM (rather than one each)
        self.config['SLURM_IDS_PER_JOB'] = config.get("SLURM_IDS_PER_JOB",
            self.SLURM_IDS_PER_JOB)


    def _run_llm(self, note_ids):
        if self.within_worker and not self.workers_have_networking: db = self.worker_cache
        else: db = self.db

        llm = self.create_engine()
        max_prompt_length = self.LLM_MAX_MODEL_LENGTH - self.LLM_TOKENS_TO_GENERATE - 4
        vocab = Vocab(self.VOCAB_XLSX, sheet_name=self.VOCAB_SHEET_NAME)

        self.start_timer()

        prompts = []
        section_ids = {}
        out_note_ids = []

        for note_id in note_ids:
            # Get the note section to work on
            row = db.get_note_section(note_id, self.sec_strat)
            if row is None: 
                raise RuntimeError(f"No {self.sec_task_name} section found for note id {note_id}")
            section_value = row.section_value
            section_ids[note_id] = row.id

            for terms in chunker(vocab.terms, self.TERMS_PER_PROMPT):
                prompt_args = {
                    "input": section_value,
                    "symptoms": ", ".join(terms),
                    "symptoms_absent_json": json.dumps({term: False for term in terms})
                }

                # Ensure the prompt will fit in the context window, with 4 tokens of wiggle room
                full_prompt = llm.autotrim_prompt(max_prompt_length, **prompt_args)
                if full_prompt is None:
                    echo_warn(f"Autotrimming failed for note id {note_id}")
                    # Couldn't autotrim; likely a malformed section, but try using the whole thing
                    full_prompt = llm.create_prompt(**prompt_args)
                
                out_note_ids.append(note_id)
                prompts.append(full_prompt)
            
        # Run all the prompts through the LLM backend
        # Outputs are already deserialized from JSON into Python lists or dicts
        echo_info(f"Running {len(note_ids)} notes thru the {self.NICE_MODEL_NAME} model "
            f"using the {self.backend} backend")
        outputs = llm(prompts, **self._sampling_params)
        runtime = self.stop_timer() / len(note_ids)

        outputs_by_note_id = defaultdict(dict)
        for i, output in enumerate(outputs):
            note_id = out_note_ids[i]
            if output is not None and isinstance(output, dict): 
                outputs_by_note_id[note_id].update(output)
            else:
                echo_warn(f"Invalid LLM output generated for note id {note_id}")
        
        for note_id, output in outputs_by_note_id.items():
            present_terms = [term for term, presence in output.items() if presence is True]
            feature = "\n".join(present_terms)
            output_raw = json.dumps(output)
            section_id = section_ids[note_id]
            db.upsert_note_feature(note_id, self, output_raw, feature, section_id, runtime)

