"""
Whitrum Model Head: WhitrumModel + WhitrumForCausalLM
Based on Qwen 2.5/3 architecture. ~350M parameters.
Founder: Oğuzhan (Dr0xy-Drawn)
Copyright (c) 2026 Whitrum AI. All rights reserved.
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from .configuration_whitrum import WhitrumConfig
from .modeling_whitrum import WhitrumDecoderLayer, WhitrumRMSNorm


class WhitrumPreTrainedModel(PreTrainedModel):
    config_class = WhitrumConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhitrumDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class WhitrumModel(WhitrumPreTrainedModel):
    """Whitrum base transformer model. ~350M params. Founder: Oğuzhan (Dr0xy-Drawn)"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([WhitrumDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = WhitrumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        past_len = cache_position[0] if cache_position is not None else 0
        seq_len = input_tensor.shape[1]
        target_len = attention_mask.shape[-1] if attention_mask is not None else seq_len + past_len
        causal_mask = torch.full((seq_len, target_len), fill_value=torch.finfo(input_tensor.dtype).min, dtype=input_tensor.dtype, device=input_tensor.device)
        if seq_len != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        return causal_mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, seq_len = inputs_embeds.shape[:2]
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(past_len, seq_len + past_len, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(bsz, -1)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, torch.tensor([past_len]))
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_cache = () if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer.__call__, hidden_states, causal_mask, position_ids, past_key_values, use_cache, output_attentions)
            else:
                layer_outputs = layer(hidden_states, causal_mask, position_ids, past_key_values, use_cache, output_attentions)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_cache += (layer_outputs[2],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache if use_cache else None,
                                      hidden_states=all_hidden_states, attentions=all_self_attns)


class WhitrumForCausalLM(WhitrumPreTrainedModel, GenerationMixin):
    """Whitrum Causal LM. ~350M params. Founder: Oğuzhan (Dr0xy-Drawn)"""
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = WhitrumModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss()(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1).to(shift_logits.device))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, past_key_values[0][0].shape[2]:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        return {"input_ids": input_ids, "position_ids": position_ids, "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"), "attention_mask": attention_mask}
