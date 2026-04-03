import torch
import torch.nn as nn
import torch.distributed as dist
from .configs import MoEConfig
from .router import TopKRouter
from .experts import ExpertLayer
from .dispatch import pt_dispatch, pt_combine
from .distributed import all_to_all_forward, all_to_all_backward

class DistributedMoELayer(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        if config.num_experts % self.world_size != 0:
            raise ValueError(f"Total Experts ({config.num_experts}) must be divisible by world_size ({self.world_size})")
            
        self.experts_per_rank = config.num_experts // self.world_size
        
        # Router is identically replicated across all GPUs (DP-style)
        self.router = TopKRouter(config)
        
        # Local experts only (EP-style)
        # Re-use config but override the expert count for strict VRAM containment
        local_config = MoEConfig(
            hidden_dim=config.hidden_dim,
            ffn_dim=config.ffn_dim,
            num_experts=self.experts_per_rank,
            top_k=config.top_k,
            activation=config.activation,
            dtype=config.dtype,
            device=config.device,
            routing_mode=config.routing_mode
        )
        self.local_experts = ExpertLayer(local_config)
        
    def forward(self, x: torch.Tensor):
        num_tokens, hidden_dim = x.shape
        
        # 1. Local Routing (All GPUs execute simultaneously on their chunk of batch size)
        routing_weights, selected_experts = self.router(x)
        
        # 2. Determine target ranks for network addressing
        # e.g., if expert 5 is assigned, and we have 4 experts per rank, it goes to rank 1.
        target_ranks = selected_experts // self.experts_per_rank
        
        # 3. Sort/Group tokens by Target Rank to package them for networking
        dispatched_x, rank_counts, network_sort_indices = pt_dispatch(
            x, target_ranks, self.world_size
        )
        
        # Flatten expert associations to pair them up with grouped tokens
        flat_experts = selected_experts.view(-1)
        dispatched_experts = flat_experts[network_sort_indices].unsqueeze(1)
        
        # 4. NETWORK EXCHANGE (NVLink/NCCL Cross-GPU Send) --> See distributed.py
        recv_tokens, recv_expert_ids, recv_counts, send_sqls, recv_sqls = all_to_all_forward(
            dispatched_x, dispatched_experts, rank_counts, self.world_size
        )
        
        # 5. Local Compute (Process whatever we just received over the network)
        # Shift global expert IDs down to their relative 0-indexed local index
        local_expert_ids = recv_expert_ids - (self.rank * self.experts_per_rank)
        
        # Perform memory sort for the local compute
        local_dispatched_x, local_expert_counts, local_sort_indices = pt_dispatch(
            recv_tokens, local_expert_ids, self.experts_per_rank
        )
        
        expert_outputs = torch.empty_like(local_dispatched_x)
        offset = 0
        for i in range(self.experts_per_rank):
            count = local_expert_counts[i].item()
            if count > 0:
                expert_in = local_dispatched_x[offset:offset+count]
                expert_out = self.local_experts(expert_in, i)
                expert_outputs[offset:offset+count] = expert_out
                offset += count
                
        # Un-permute local memory dispatch buffer
        dummy_weights = torch.ones(recv_tokens.shape[0], 1, dtype=recv_tokens.dtype, device=recv_tokens.device)
        processed_recv_tokens = pt_combine(expert_outputs, local_sort_indices, dummy_weights)
        
        # 6. NETWORK EXCHANGE (Reverse NVLink/NCCL Cross-GPU Send)
        returned_tokens = all_to_all_backward(processed_recv_tokens, send_sqls, recv_sqls)
        
        # 7. Final Output Combine (Un-permute network sort locally and apply original weights)
        combined_x = pt_combine(returned_tokens, network_sort_indices, routing_weights)
        
        return combined_x
