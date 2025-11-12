from .llama3_8b import PresentingSxFeatureLlama3Instruct8BStrategy
from ...section.hpi_short.regex import ShortHPISectionRegexStrategy
from ....tasks.feature import PresentingSymptomsFeatureTask

class PresentingSxFeatureLlama31Instruct8BStrategy(PresentingSxFeatureLlama3Instruct8BStrategy):
    """\
    Attempts to extract the `presenting_sx` feature using Llama 3.1 Instruct 8B model,
    an 8B parameter model in the Llama family released by Meta in July 2024. 

    This uses the original weights casted to torch.float16, because the Tesla V100s that we can 
    run this on in ERISXdl do not support bfloat16. 
    https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

    Also uses the `hpi_short` sections created by the section.hpi_short.regex strategy.
    """
    task = PresentingSymptomsFeatureTask
    name = "feature.presenting_sx.llama31_8b"
    version = "0.0.1"
    prereq_tasks = []

    SECTION_STRATEGY_CLASS = ShortHPISectionRegexStrategy

    MODEL_ID = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

    # Model context length. Note that Vllm will not generate past the end of the context window.
    # Llama 3.1 supports up to 131,072 (128k) but this is also constrained by GPU memory
    LLM_MAX_MODEL_LENGTH = 4096 

    NICE_MODEL_NAME = "Llama3.1 8B Instruct"