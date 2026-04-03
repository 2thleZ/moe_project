from dataclasses import dataclass
import torch

@dataclass
class MoEConfig:
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_experts: int = 8
    top_k: int = 2
    activation: str = "gelu" # "gelu" or "silu"
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    routing_mode: str = "natural" # natural, force_uniform, force_skewed
