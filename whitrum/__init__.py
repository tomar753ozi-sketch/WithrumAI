"""
Whitrum AI - Language Model Package
Based on Qwen 2.5/3 architecture. ~350M parameters.
Founder: Oguzhan (Dr0xy-Drawn)
"""

from .configuration_whitrum import WhitrumConfig
from .modeling_whitrum_head import WhitrumModel, WhitrumForCausalLM
from .tokenization_whitrum import WhitrumTokenizer

__version__ = "0.1.0"
__author__ = "Oguzhan (Dr0xy-Drawn)"
