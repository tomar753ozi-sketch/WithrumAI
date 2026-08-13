"""
Whitrum Language Model Configuration
Based on Qwen 2.5/3 architecture. ~350M parameters.
Founder: Oğuzhan (Dr0xy-Drawn)
Copyright (c) 2026 Whitrum AI. All rights reserved.
"""

from transformers import PretrainedConfig


class WhitrumConfig(PretrainedConfig):
    model_type = "whitrum"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=18,
        num_attention_heads=14,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=151643,
        eos_token_id=151645,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=True,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        output_hidden_states=False,
        output_attentions=False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
