import torch
import torch.nn as nn
import torch.nn.functional as F
from .configs import MoEConfig

class ExpertMLP(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)
        self.act = F.gelu if config.activation == "gelu" else F.silu

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class ExpertLayer(nn.Module):
    """Container for multiple experts running on a single GPU."""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.experts = nn.ModuleList([ExpertMLP(config) for _ in range(config.num_experts)])
        
    def forward(self, x, expert_idx):
        """
        x: [tokens_for_expert, hidden_dim]
        expert_idx: int
        """
        return self.experts[expert_idx](x)
