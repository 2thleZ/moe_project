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
        
        # identically replicate router across all GPUs (DP-style)
        self.router = TopKRouter(config)
        
        # load local experts only (EP-style)
        # override the expert count for strict VRAM containment
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
        self.to(dtype=config.dtype)
        
    def forward(self, x: torch.Tensor, return_timings: bool = False):
        num_tokens, hidden_dim = x.shape
        timings = {}
        
        def get_time(start, end):
            torch.cuda.synchronize()
            return start.elapsed_time(end)

        if return_timings:
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_routing_end = torch.cuda.Event(enable_timing=True)
            ev_dispatch_end = torch.cuda.Event(enable_timing=True)
            ev_nccl_fw_end = torch.cuda.Event(enable_timing=True)
            ev_expert_end = torch.cuda.Event(enable_timing=True)
            ev_nccl_bw_end = torch.cuda.Event(enable_timing=True)
            ev_combine_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()

        # route locally (GPUs execute simultaneously on their chunk of batch size)
        routing_weights, selected_experts = self.router(x)
        
        if return_timings: ev_routing_end.record()
        
        # compute target ranks for network addressing
        # example: if expert 5 is assigned to a layer with 4 experts per rank, it resolves to rank 1.
        target_ranks = selected_experts // self.experts_per_rank
        
        # sort and group tokens by target rank to prepare for networking
        dispatched_x, rank_counts, network_sort_indices = pt_dispatch(
            x, target_ranks, self.world_size
        )
        
        # flatten expert associations to pair with grouped tokens
        flat_experts = selected_experts.view(-1)
        dispatched_experts = flat_experts[network_sort_indices].unsqueeze(1)
        
        if return_timings: ev_dispatch_end.record()
        
        # execute forward network exchange via NCCL
        recv_tokens, recv_expert_ids, recv_counts, send_sqls, recv_sqls = all_to_all_forward(
            dispatched_x, dispatched_experts, rank_counts, self.world_size
        )
        
        if return_timings: ev_nccl_fw_end.record()
        
        # process tokens received over the network
        # remap global expert IDs to their relative local 0-index
        local_expert_ids = recv_expert_ids - (self.rank * self.experts_per_rank)
        
        # sort memory for the local computation block
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
                
        # un-permute local memory dispatch buffer
        dummy_weights = torch.ones(recv_tokens.shape[0], 1, dtype=recv_tokens.dtype, device=recv_tokens.device)
        processed_recv_tokens = pt_combine(expert_outputs, local_sort_indices, dummy_weights)
        
        if return_timings: ev_expert_end.record()
        
        # execute reverse network exchange via NCCL
        returned_tokens = all_to_all_backward(processed_recv_tokens, send_sqls, recv_sqls)
        
        if return_timings: ev_nccl_bw_end.record()
        
        # un-permute network sort locally and apply original weights
        combined_x = pt_combine(returned_tokens, network_sort_indices, routing_weights)
        
        if return_timings:
            ev_combine_end.record()
            torch.cuda.synchronize()
            timings = {
                "routing": ev_start.elapsed_time(ev_routing_end),
                "dispatch": ev_routing_end.elapsed_time(ev_dispatch_end),
                "nccl_fw": ev_dispatch_end.elapsed_time(ev_nccl_fw_end),
                "expert_compute": ev_nccl_fw_end.elapsed_time(ev_expert_end),
                "nccl_bw": ev_expert_end.elapsed_time(ev_nccl_bw_end),
                "combine": ev_nccl_bw_end.elapsed_time(ev_combine_end)
            }
            return combined_x, timings
            
        return combined_x
