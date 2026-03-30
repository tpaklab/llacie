from textwrap import dedent
from ...abstract_vllm_or_lcp import AbstractVllmOrLcpStrategy
from ...section.hpi_short.regex_temp import ShortHPISectionRegexStrategy
from ....tasks.feature import PresentingSymptomsFeatureTask


class PresentingSXMistralStrategy(AbstractVllmOrLcpStrategy):
    """\
    Attempts to extract the `presenting_sx` feature using Qwen3 0.6B Embedding model, 
    """


    task=PresentingSymptomsFeatureTask
    name="feature.presenting_sx.mistral"
    version = "0.0.1"
    prereq_tasks = []

    SECTION_STRATEGY_CLASS = ShortHPISectionRegexStrategy


    MODEL_ID = 'mistralai/Mistral-7B-Instruct-v0.1'
    GGUF_MODEL_ID = 'TheBloke/Mistral-7B-Instruct-v0.1-GGUF'
    GGUF_MODEL_FILENAME = 'mistral-7b-instruct-v0.1.Q5_K_S.gguf'


    SLURM_THREADS_PER_JOB = 16
    ARRAY_OF_TEN_SHORT_STRINGS_JSON_SCHEMA = """
    {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "array",
        "minItems": 1,
        "maxItems": 10,
        "items": {
            "type": "string",
            "minLength": 2,
            "maxLength": 100
        }
    }
    """
    JSON_SCHEMA = ARRAY_OF_TEN_SHORT_STRINGS_JSON_SCHEMA

    VLLM_ENABLED = False
    VLLM_DTYPE = 'half'
    # Arguments to SamplingParams in vllm
    # See: https://docs.vllm.ai/en/latest/api/vllm/#vllm.SamplingParams
    # These are translated into equivalent arguments for llama-cpp-python's create_chat_completion
    SAMPLING_PARAMS = {
    "temperature":0.1,         
    "top_p":0.8,                 
    "top_k":20,                 
    "repetition_penalty":1.05,   
    "min_p":0.01
    # "chat_template_kwargs": {"enable_thinking": False}
         }
    NICE_MODEL_NAME = "Mistral 7B"

   # LLM_MAX_MODEL_LENGTH = 3044  

    LLM_SYSTEM_PROMPT = dedent("""\
        You are a clinical researcher that reads medical charts and answers questions about them. 
        Use only the information in the text provided to answer the question.
        If a patient denies something, do not include it in your answer.
        After you provide an answer, you immediately stop talking. /no_think""")
    LLM_USER_PROMPT = dedent("""\
        Read the following patient history and list the patient's presenting symptoms.
        Include only symptoms present now or reported for the days to weeks leading up to admission.
        Ignore any symptoms from past medical history or prior hospital admissions.
        Give your answer as a JSON array containing up to ten strings.
        Each string contains between one and three words. /no_think
        
        {input}""")
