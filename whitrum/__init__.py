"""
Whitrum AI - Language Model Package
Based on Qwen 2.5/3 architecture. ~350M parameters.
Founder: Oğuzhan (Dr0xy-Drawn)
Copyright (c) 2026 Whitrum AI. All rights reserved.
"""

from .configuration_whitrum import WhitrumConfig
from .modeling_whitrum_head import WhitrumModel, WhitrumForCausalLM

__version__ = "0.1.0"
__author__ = "Oğuzhan (Dr0xy-Drawn)"
