import torch
import torch.nn as nn
from .configs import MoEConfig
from .router import TopKRouter
from .experts import ExpertLayer
from .dispatch import pt_dispatch, pt_combine

class MoELayer(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.router = TopKRouter(config)
        self.expert_layer = ExpertLayer(config)
        self.to(device=config.device, dtype=config.dtype)

    def forward(self, x: torch.Tensor):
        """
        x: [num_tokens, hidden_dim]
        """
        # 1. Routing
        routing_weights, selected_experts = self.router(x)
        
        # 2. Dispatch
        # dispatched_x: [num_tokens * top_k, hidden_dim]
        dispatched_x, expert_counts, sort_indices = pt_dispatch(
            x, selected_experts, self.config.num_experts
        )
        
        # 3. Expert Compute
        expert_outputs = torch.empty_like(dispatched_x)
        offset = 0
        for i in range(self.config.num_experts):
            count = expert_counts[i].item()
            if count > 0:
                expert_in = dispatched_x[offset:offset+count]
                expert_out = self.expert_layer(expert_in, i)
                expert_outputs[offset:offset+count] = expert_out
                offset += count
                
        # 4. Combine
        combined_x = pt_combine(expert_outputs, sort_indices, routing_weights)
        
        return combined_x
