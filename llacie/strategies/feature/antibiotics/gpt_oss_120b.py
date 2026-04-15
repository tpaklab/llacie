from textwrap import dedent

from ...custom_health_server import AbstractUciHealthStrategy
from ...section.medicines.spacy import AntibioticsSpacyStrategy
from ....tasks.feature import AntibioticsFeatureTask

class AntibioticsFeatureLlama3Instruct8BStrategy(AbstractUciHealthStrategy):
    """\
    Attempts to extract the `antibiotics` feature using Llama 3 Instruct 8B model,
    an 8B parameter model in the Llama family released by Meta in April 2024. 

    This uses the original weights casted to torch.float16, because the Tesla V100s that we can 
    run this on in ERISXdl do not support bfloat16. 
    https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

    """
    task = AntibioticsFeatureTask
    name = "feature.antibiotics.gpt_oss_120B"
    version = "0.0.1"
    prereq_tasks = []

    SECTION_STRATEGY_CLASS = AntibioticsSpacyStrategy

    
    NICE_MODEL_NAME = "GPT OSS 120B"

    # Model context length. Note that Vllm will not generate past the end of the context window.
    LLM_MAX_MODEL_LENGTH = 120000
    LLM_SYSTEM_PROMPT = dedent("""\
        You are a clinical researcher that reads medical charts and answers questions about them. 
        Use only the information in the text provided to answer the question.
        If a patient denies something, do not include it in your answer.
        After you provide an answer, you immediately stop talking.""")
    LLM_USER_PROMPT = dedent("""\
        Read the following patient history and list the antibiotics the patient was taking before coming to the hospital.
        Include only antibiotics prescribed or taken prior to this hospital admission, such as those started by a primary care provider
        ,urgent care, or during a recent prior hospitalization.
        Do not include antibiotics started for the first time during this admission.
        Return a json list consisting of dicts. Each dict under the array should follow the below format.
        Each dict should follow the below format:
        (Drug Name: str: The name of the drug, 1 to 2 words long
        Route: str | The route with which the medication is given
        Dose: str | Dose given in mg or g
        Frequency: str | Daily/weekly etc.
        Duration: int | Duration in days
        Start date: str | MM/DD/YYYY format
        End date: str | MM/DD/YYYY format)
        If you ever do not know the values under any key, fill it with NULL . Do not guess if you do not know. 
        {input}""")
