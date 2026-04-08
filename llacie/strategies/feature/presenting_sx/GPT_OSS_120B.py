from textwrap import dedent

from ...custom_health_server import AbstractUciHealthStrategy
from ...section.hpi_short.regex_temp import ShortHPISectionRegexStrategy
from ....tasks.feature import PresentingSymptomsFeatureTask

class PresentingSxFeatureGPTOSS120BStrategy(AbstractUciHealthStrategy):
    """\
    Attempts to extract the `presenting_sx` feature using GPT OSS 120B model,

    Also uses the `hpi_short` sections created by the section.hpi_short.regex strategy.
    """
    task = PresentingSymptomsFeatureTask
    name = "feature.presenting_sx.GPT_OSS_120B"
    version = "0.0.1"
    prereq_tasks = []

    SECTION_STRATEGY_CLASS = ShortHPISectionRegexStrategy

    # SAMPLING_PARAMS = {
    #     "temperature": 0.1,
    #     "top_p": 0.8,
    #     "repetition_penalty": 1.05
    # }

    NICE_MODEL_NAME = "GPT-OSS 120B"

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
