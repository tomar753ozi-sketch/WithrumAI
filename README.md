# Whitrum AI

**Founder: Oğuzhan (Dr0xy-Drawn)**

A ~350M parameter causal language model based on Qwen 2.5/3 architecture.

## Architecture

| Param | Value |
|-------|-------|
| Parameters | ~350M |
| Layers | 18 |
| Hidden Size | 896 |
| Attention Heads | 14 |
| KV Heads (GQA) | 2 |
| Intermediate Size | 4864 |
| Vocab Size | 151,936 |
| Max Seq Len | 8,192 |
| Context | 32K tokens |

## Features

- RoPE (Rotary Position Embedding)
- SwiGLU activation
- RMSNorm
- Grouped Query Attention (GQA)
- QKV bias
- Tie word embeddings

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Whitrum-AI/Whitrum-350M")
tokenizer = AutoTokenizer.from_pretrained("Whitrum-AI/Whitrum-350M")
```

## License

Apache 2.0

## Credits

- Architecture: Based on Qwen 2.5/3 by Alibaba Cloud
- Adapted by: **Oğuzhan (Dr0xy-Drawn)**
