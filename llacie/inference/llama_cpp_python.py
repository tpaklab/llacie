import os
import re
import json

from llama_cpp import Llama
from transformers import AutoTokenizer
from json.decoder import JSONDecodeError
from tqdm import tqdm

from ..utils import echo_err

class LlamaCppPython(object):
    """
    Runs LLMs using the `llama-cpp-python` package, which creates Python bindings for llama.cpp.
    Structured outputs are supported using constrained decoding and JSON schemas.
    See: https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#json-schema-mode
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
            tokenizer_model_path=None, **engine_args):
        """
        Instantiates an instance of the engine. For a comprehensive list of `engine_args` see
        the arguments for the Llama() constructor:
        https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama
        """
        self.config = config
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.json_schema = json_schema if json_schema is not None else self.JSON_SCHEMA

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_path, 
            local_files_only = tokenizer_model_path.count('/') > 1
        )

        # If provided with the "filename" engine argument, the model_path is a HuggingFace repo ID
        if engine_args.get("filename"):
            # In this case, the model is loaded from HuggingFace
            self._model = Llama.from_pretrained(model_path, **engine_args)
        else:
            self._model = Llama(model_path, **engine_args)

    
    def create_prompt(self, **kwargs):
        messages = [
            {"role": "system", "content": self.system_prompt.format(**kwargs)},
            {"role": "user", "content": self.user_prompt.format(**kwargs)}
        ]
        return messages


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
            tokenized_prompt = self.tokenizer.apply_chat_template(
                full_prompt,
                tokenize = True, 
                add_generation_prompt = True
            )
            if len(tokenized_prompt) <= prompt_max_tokens: break
            # If it's too long, cleave off the last sentence-like piece, and try again
            sentences_and_periods = re.split(r'([.]\s+|[.]$)', trimmed_text)
            if len(sentences_and_periods) < 4:
                return None
            else: 
                trimmed_text = ''.join(sentences_and_periods[:-4]) + '.'
                kwargs[trim_key] = trimmed_text
        return full_prompt
    

    def __call__(self, prompts, **sampling_params):
        response_format = {
            "type": "json_object",
            "schema": json.loads(self.json_schema),
        }

        outputs = []
        for prompt in tqdm(prompts):
            try:
                full_output = self._model.create_chat_completion(
                    prompt,
                    response_format=response_format,
                    **sampling_params
                )
                output = json.loads(full_output["choices"][0]["message"]["content"])
                outputs.append(output)
            except (IndexError, KeyError):
                echo_err(f"Invalid response from create_chat_completion: {full_output}")
                outputs.append(None)
            except JSONDecodeError as e:
                echo_err(f"Invalid JSON. '{e.msg}' at char {e.pos}:\n{e.doc}")
                outputs.append(None)

        return outputs