import torch
import torch.nn as nn
import torch.nn.functional as F
from .configs import MoEConfig

class TopKRouter(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: [num_tokens, hidden_dim]
        Returns:
            routing_weights: [num_tokens, top_k]
            selected_experts: [num_tokens, top_k] int
        """
        logits = self.gate(x) # [num_tokens, num_experts]
        
        # apply synthetic routing constraints for benchmarking
        if self.config.routing_mode == "force_uniform":
            # distribute tokens uniformly across all experts
            num_tokens = x.size(0)
            experts = torch.arange(self.config.num_experts, device=x.device)
            selected_experts = experts.repeat(num_tokens // self.config.num_experts + 1)[:num_tokens].unsqueeze(1)
            if self.config.top_k > 1:
                # pad remaining top-k slots with zero index
                selected_experts = torch.cat([selected_experts, torch.zeros(num_tokens, self.config.top_k - 1, device=x.device, dtype=torch.long)], dim=1)
            routing_weights = torch.ones_like(selected_experts, dtype=x.dtype, device=x.device) / self.config.top_k
            return routing_weights, selected_experts
            
        elif self.config.routing_mode == "force_skewed":
            # collapse all token assignments to zero index
            selected_experts = torch.zeros(x.size(0), self.config.top_k, dtype=torch.long, device=x.device)
            routing_weights = torch.ones_like(selected_experts, dtype=x.dtype, device=x.device) / self.config.top_k
            return routing_weights, selected_experts
            
        # determine natural top-k routing indices
        routing_weights, selected_experts = torch.topk(logits, self.config.top_k, dim=-1)
        
        # normalize routing weights via softmax
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(x.dtype)
        
        return routing_weights, selected_experts
