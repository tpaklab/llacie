from textwrap import dedent

from ...abstract_llama_cpp import AbstractLlamaCppStrategy
from ...section.hpi_short.regex import ShortHPISectionRegexStrategy
from ....tasks.feature import PresentingSymptomsFeatureTask

class PresentingSxFeatureLLama1Strategy(AbstractLlamaCppStrategy):
    """\
    Attempts to extract the `presenting_sx` feature using LLaMA 1, 13B model,
    with no fine-tuning.

    Uses the `hpi_short` sections created by the section.hpi_short.regex strategy.
    """
    task = PresentingSymptomsFeatureTask
    name = "feature.presenting_sx.llama1"
    version = "0.0.1"
    prereq_tasks = []

    SECTION_STRATEGY_CLASS = ShortHPISectionRegexStrategy
    SLURM_ENABLED = True

    LLAMA_CPP_BIN = 'main-41c6741'
    LLAMA_CPP_THREADS = 16
    SLURM_THREADS_PER_JOB = 16
    LLAMA_MODEL_PATH = 'models/llama/13B/ggml-model-q4_0.bin'
    # Note: this strategy predates the use of grammars to constrain output!
    LLM_MAX_TOKENS = 1020
    LLAMA_CPP_OPTS = f"-m {LLAMA_MODEL_PATH} -n 128 --temp 0.1 --top_p 1.0 -c {LLM_MAX_TOKENS + 4}"
    # The Llama 1 and 2 models specifically use LlamaTokenizer in the transformers module
    LLAMA_VOCAB_PATH = 'models/llama/tokenizer.model'

    LLM_PROMPT = dedent("""\
        {input}

        This patient had the following presenting symptoms: """)
