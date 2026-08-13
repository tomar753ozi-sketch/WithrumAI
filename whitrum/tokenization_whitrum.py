import base64, os, json
from typing import List, Optional
from transformers import PretrainedTokenizer


class WhitrumTokenizer(PretrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]
    resource_files_names = {"vocab_file": "vocab.bpe"}

    def __init__(self, vocab_file=None, errors="replace", padding_side="left", **kwargs):
        super().__init__(**kwargs)
        if vocab_file is None:
            vocab_file = os.path.join(os.path.dirname(__file__), "vocab.bpe")
        self.errors = errors
        self.mergeable_ranks = {}
        if os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        token = base64.b64decode(parts[0])
                        rank = int(parts[1])
                        self.mergeable_ranks[token] = rank
        self.special_tokens = {
            "<|im_start|>": 60000,
            "<|im_end|>": 60001,
            "<|pad|>": 60002,
            "<|unk|>": 60003,
            "<|thought|>": 60004,
            "<|/thought|>": 60005,
        }
        self.tokenizer = self._build_tiktoken()
        self.decoder = {v: k for k, v in self.mergeable_ranks.items()}
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

    def _build_tiktoken(self):
        import tiktoken
        enc = tiktoken.Encoding("whitrum")
        enc._mergeable_ranks = self.mergeable_ranks
        enc._special_tokens = self.special_tokens
        return enc

    @property
    def vocab_size(self):
        return 60006

    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        return {self.decoder[i]: i for i in range(len(self.decoder))}

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.mergeable_ranks.get(tokens.encode("utf-8"), self.special_tokens.get(tokens, self.unk_token_id))
        return [self.mergeable_ranks.get(t.encode("utf-8"), self.special_tokens.get(t, self.unk_token_id)) for t in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            return self.decoder.get(ids, "".encode("utf-8"))
        tokens = []
        for i in ids:
            if i in self.special_tokens and skip_special_tokens:
                continue
            tokens.append(self.decoder.get(i, "".encode("utf-8")))
        return tokens

    def _tokenize(self, text):
        raise NotImplementedError

    def _convert_token_to_id(self, token):
        return self.mergeable_ranks.get(token.encode("utf-8"), self.unk_token_id)

    def _convert_id_to_token(self, index):
        return self.decoder.get(index, "".encode("utf-8"))

    def tokenize(self, text, allowed_special="all", disallowed_special=(), **kwargs):
        tokens = []
        text = unicodedata.normalize("NFC", text)
        for t in self.tokenizer.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special):
            tokens.append(self.decoder.get(t, ""))
        return tokens

    def encode(self, text, allowed_special="all", disallowed_special=(), **kwargs):
        return self.tokenizer.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)

    def decode(self, tokens, skip_special_tokens=False, errors=None, **kwargs):
        if isinstance(tokens, int):
            tokens = [tokens]
        if skip_special_tokens:
            tokens = [i for i in tokens if i < 60000]
        return self.tokenizer.decode(tokens, errors=errors or self.errors)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        filename = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.bpe")
        with open(filename, "w", encoding="utf-8") as f:
            for token, rank in sorted(self.mergeable_ranks.items(), key=lambda x: x[1]):
                f.write(base64.b64encode(token).decode("utf-8") + " " + str(rank) + "\n")
        return (filename,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos = [self.special_tokens["<|im_start|>"]]
        eos = [self.special_tokens["<|im_end|>"]]
        output = bos + token_ids_0 + eos
        if token_ids_1 is not None:
            output = output + token_ids_1 + eos
        return output


import unicodedata
