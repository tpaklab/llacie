from textwrap import dedent

from ...abstract_vllm_or_lcp import AbstractVllmOrLcpStrategy
from ...section.hpi_short.regex import ShortHPISectionRegexStrategy
from ....tasks.feature import PresentingSymptomsFeatureTask

class PresentingSxFeatureLlama3Instruct8BStrategy(AbstractVllmOrLcpStrategy):
    """\
    Attempts to extract the `presenting_sx` feature using Llama 3 Instruct 8B model,
    an 8B parameter model in the Llama family released by Meta in April 2024. 

    This uses the original weights casted to torch.float16, because the Tesla V100s that we can 
    run this on in ERISXdl do not support bfloat16. 
    https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

    Also uses the `hpi_short` sections created by the section.hpi_short.regex strategy.
    """
    task = PresentingSymptomsFeatureTask
    name = "feature.presenting_sx.llama3_8b"
    version = "0.0.1"
    prereq_tasks = []

    SECTION_STRATEGY_CLASS = ShortHPISectionRegexStrategy

    # HuggingFace respository ID for the full PyTorch/Transformers model, which vllm uses
    MODEL_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'
    # As a fallback, you can also provide a GGUF version, which is what llama-cpp-python uses
    GGUF_MODEL_ID = 'QuantFactory/Meta-Llama-3-8B-Instruct-GGUF'
    GGUF_MODEL_FILENAME = '*.Q5_K_S.gguf'
    
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
    LLM_TOKENS_TO_GENERATE = 1024
    # Arguments to SamplingParams in vllm
    # See: https://docs.vllm.ai/en/latest/api/vllm/#vllm.SamplingParams
    # These are translated into equivalent arguments for llama-cpp-python's create_chat_completion
    SAMPLING_PARAMS = {
        "temperature": 0.1,
        "top_p": 0.8,
        "repetition_penalty": 1.05
    }

    NICE_MODEL_NAME = "Llama3 8B Instruct"

    # Model context length. Note that Vllm will not generate past the end of the context window.
    LLM_MAX_MODEL_LENGTH = 3044  
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
