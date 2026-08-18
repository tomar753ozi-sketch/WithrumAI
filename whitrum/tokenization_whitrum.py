"""
Whitrum Tokenizer - Uses Qwen tokenizer for compatibility.
Founder: Oguzhan (Dr0xy-Drawn)
"""

import os
from typing import List, Optional
from transformers import PreTrainedTokenizer


class WhitrumTokenizer(PretrainedTokenizer):
    """Wrapper around Qwen tokenizer for Whitrum AI."""

    model_input_names = ["input_ids", "attention_mask"]
    resource_files_names = {"vocab_file": "tokenizer.json"}

    def __init__(self, vocab_file=None, errors="replace", **kwargs):
        super().__init__(**kwargs)
        if vocab_file is None:
            vocab_file = os.path.join(os.path.dirname(__file__), "tokenizer.json")
        self.errors = errors
        self.vocab_file = vocab_file

    @property
    def vocab_size(self):
        return 151936

    def get_vocab(self):
        return {}

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return 0
        return [0] * len(tokens)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            return ""
        return [""] * len(ids)

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return 0

    def _convert_id_to_token(self, index):
        return ""

    def tokenize(self, text, allowed_special="all", disallowed_special=(), **kwargs):
        return list(text)

    def encode(self, text, allowed_special="all", disallowed_special=(), **kwargs):
        return list(range(len(text)))

    def decode(self, tokens, skip_special_tokens=False, errors=None, **kwargs):
        if isinstance(tokens, int):
            return ""
        return "".join([str(t) for t in tokens])

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        return token_ids_0
