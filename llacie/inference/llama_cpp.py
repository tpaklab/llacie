import os
import re
import tempfile
import subprocess as sub

# Squelch warnings that the transformers package gives about not having PyTorch etc. installed
# We are importing it only for its tokenizers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import LlamaTokenizer, AutoTokenizer

from ..utils import echo_err

# Note: This eventually could be replaced with Python bindings for llama.cpp:
# https://github.com/abetlen/llama-cpp-python

class LlamaCpp(object):
    HOME_DIR = os.path.expanduser('~')
    LLAMA_CPP_DIR = f"{HOME_DIR}/src/llama.cpp"
    LLAMA_CPP_THREADS = 16
    LLAMA_CPP_TMPDIR = f"{HOME_DIR}/tmp/llama_input"


    def __init__(self, config, opts, llama_cpp_bin='main', vocab_path=None, hf_model_id=None,
            tokenizer_model_path=None, alt_bos_token=None):
        self.llama_cpp_bin = llama_cpp_bin
        self.opts = opts
        self.llama_cpp_dir = config.get("LLAMA_CPP_DIR", self.LLAMA_CPP_DIR)
        self.cwd = self.llama_cpp_dir
        self.threads = config.get("LLAMA_CPP_THREADS", self.LLAMA_CPP_THREADS)
        self.tmp_dir = config.get("LLAMA_CPP_TMPDIR", self.LLAMA_CPP_TMPDIR)
        self.alt_bos_token = alt_bos_token

        self.tokenizer = None
        if vocab_path is not None:
            vocab_path = os.path.join(self.llama_cpp_dir, vocab_path)
            self.tokenizer = LlamaTokenizer(vocab_path, legacy = False)
        elif tokenizer_model_path is not None:
            tokenizer_model_path = os.path.join(self.llama_cpp_dir, tokenizer_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
        elif hf_model_id is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)


    def autotrim_prompt(self, input_text, prompt, prompt_max_tokens):
        """Creates a prompt and ensures it's within the token limit.

        `input_text` is substituted into `prompt` using str.format, at the
        {input} marker. 

        `prompt_max_tokens` is the token limit, usually $context_window - 4.
        
        Returns the full prompt with trimmed `input_text` already inserted, 
        or None if the process fails."""
        trimmed_text = input_text
        if self.tokenizer is None:
            echo_err("Neither `vocab_path` nor `hf_model_id` was given; therefore a tokenizer "
                "could not be loaded for this model.")
            return None
        while True:
            full_prompt = prompt.format(input = trimmed_text)
            if len(self.tokenizer.encode(full_prompt)) <= prompt_max_tokens: break
            # If it's too long, cleave off the last sentence-like piece, and try again
            sentences_and_periods = re.split(r'([.]\s+|[.]$)', trimmed_text)
            if len(sentences_and_periods) < 4:
                return None
            else: trimmed_text = ''.join(sentences_and_periods[:-4]) + '.'
        return full_prompt


    def _llama_error(self, err_msg):
        echo_err("\n".join(filter(lambda line: "error:" in line, err_msg.split("\n"))))
        return None


    def __call__(self, full_prompt):
        llama_bin = os.path.join(self.llama_cpp_dir, self.llama_cpp_bin)
        raw_llm_output = None
        os.makedirs(self.tmp_dir, exist_ok=True)

        # <s> is a special beginning-of-string (BOS) token. We don't allow it anywhere
        # except the beginning of the prompt.
        full_prompt = re.sub(r'(?<!^)<s>', '', full_prompt)
        # We want to return only the LLM-created output (and drop the prompt which is also 
        # echoed to STDOUT). If the prompt starts with <s>, this is treated as a BOS token and
        # not echoed. Otherwise, llama.cpp echoes one space " ", for unclear reasons.
        prompt_echo_length = len(full_prompt) + 1
        if full_prompt.startswith('<s>'): prompt_echo_length -= 4
        # Some models have a special BOS token (like Llama 3) that llama.cpp adds automatically
        # to the prompt. This is also echoed to STDOUT and must be trimmed.
        if self.alt_bos_token is not None: prompt_echo_length += len(self.alt_bos_token) - 2

        with tempfile.NamedTemporaryFile(dir=self.tmp_dir) as tmp_input:
            tmp_input.write(full_prompt.encode())
            tmp_input.flush()
            required_opts = f"-t {str(self.threads)}"
            cmd = f"{llama_bin} {required_opts} {self.opts} -f {tmp_input.name}"
            res = sub.run(cmd, stdout=sub.PIPE, stderr=sub.PIPE, shell=True, cwd=self.cwd)
            if res.returncode != 0: return self._llama_error(res.stderr.decode('utf-8'))
            raw_llm_output = res.stdout.decode('utf-8')[prompt_echo_length: ]

        return raw_llm_output.strip(" .\n")