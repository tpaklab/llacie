import os
import re

from transformers import AutoTokenizer
from json.decoder import JSONDecodeError

##### Note that ordinarily we should place all imports at the top, per PEP8
##### But in this unusual case, we avoid importing the `vllm` and `outlines` packages up here
##### because they ONLY exist inside the containerized environment used by worker processes

from ..utils import echo_err

class Vllm(object):
    """
    Runs LLMs on the ERISXdl GPU cluster using the `vllm` package.
    This package is one of the fastest LLM inference packages, with loads of optimizations
        for high throughput and running slimmed-down CUDA kernels on NVIDIA GPUs
    Structured outputs are also supported using the `outlines` package and JSON schemas.
    See: https://docs.vllm.ai/en/latest/index.html
    """
    
    HOME_DIR = os.path.expanduser('~')
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


    def __init__(self, config, model_path, system_prompt, user_prompt, json_schema=None, 
            **engine_args):
        """
        Instantiates an instance of the engine. For a comprehensive list of `engine_args` see:
        https://docs.vllm.ai/en/stable/models/engine_args.html
        """
        self.config = config
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.json_schema = json_schema if json_schema is not None else self.JSON_SCHEMA

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            local_files_only = model_path.count('/') > 1
        )

        from vllm import LLM
        self._model = LLM(model=model_path, **engine_args)

    
    def create_prompt(self, **kwargs):
        messages = [
            {"role": "system", "content": self.system_prompt.format(**kwargs)},
            {"role": "user", "content": self.user_prompt.format(**kwargs)}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize = False, 
            add_generation_prompt = True
        )


    def autotrim_prompt(self, prompt_max_tokens, trim_key="input", **kwargs):
        """Creates a prompt and ensures it is within the token limit.

        `kwargs` are substituted into the chat template created by `system_prompt` and
        by `user_prompt`, using Python's native str.format().

        `prompt_max_tokens` is the token limit, usually $context_window - 4.

        If the generated prompt is longer than `prompt_max_tokens` after tokenization, the item 
        in the dict() `kwargs` at key `trim_key` is trimmed by about one sentence at a time until 
        the threshold is reached. By default, `kwargs["input"]` is trimmed.
        
        Returns the full prompt with the trimmed inputs inserted, or None if the process fails."""
        trimmed_text = kwargs.get(trim_key, "")
        if self.tokenizer is None:
            echo_err("A tokenizer for this model could not be loaded.")
            return None
        while True:
            full_prompt = self.create_prompt(**kwargs)
            if len(self.tokenizer.encode(full_prompt)) <= prompt_max_tokens: break
            # If it's too long, cleave off the last sentence-like piece, and try again
            sentences_and_periods = re.split(r'([.]\s+|[.]$)', trimmed_text)
            if len(sentences_and_periods) < 4:
                return None
            else: 
                trimmed_text = ''.join(sentences_and_periods[:-4]) + '.'
                kwargs[trim_key] = trimmed_text
        return full_prompt
    

    def __call__(self, prompts, **sampling_params):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        struct_out_params = StructuredOutputsParams(json=self.json_schema)
        sampling_params = SamplingParams(**sampling_params, structured_outputs=struct_out_params)

        try:
            outputs = self._model.generate(prompts, sampling_params=sampling_params)
        except JSONDecodeError as e:
            echo_err(f"Invalid JSON. '{e.msg}' at char {e.pos}:\n{e.doc}")
            return [None] * len(prompts)

        return outputs