import torch
from typing import Tuple

def pt_dispatch(x: torch.Tensor, selected_experts: torch.Tensor, num_experts: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch memory permutation for MoE dispatch.
    Args:
        x: [num_tokens, hidden_dim]
        selected_experts: [num_tokens, top_k] 
        num_experts: int
    Returns:
        dispatched_x: [num_tokens * top_k, hidden_dim] sorted by expert
        expert_token_counts: [num_experts] how many tokens each expert gets
        sort_indices: [num_tokens * top_k] original indices for un-permuting
    """
    num_tokens, top_k = selected_experts.shape
    hidden_dim = x.shape[1]
    
    # flatten selected experts
    flat_experts = selected_experts.view(-1)
    
    # expert-wise sorting of tokens (tokens for expert0, tokens for expert1, ..., tokens for expertN)
    sort_indices = torch.argsort(flat_experts)
    
    # permute token inputs
    sorted_token_indices = sort_indices // top_k
    dispatched_x = x[sorted_token_indices] # primary performance penalty - Triton kernel addresses this
    
    # count tokens per expert
    expert_token_counts = torch.bincount(flat_experts, minlength=num_experts)
    
    return dispatched_x, expert_token_counts, sort_indices

def pt_combine(expert_outputs: torch.Tensor, sort_indices: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch inverse memory permutation.
    Args:
        expert_outputs: [num_tokens * top_k, hidden_dim] (still sorted by expert)
        sort_indices: [num_tokens * top_k] from pt_dispatch
        routing_weights: [num_tokens, top_k]
    Returns:
        combined_x: [num_tokens, hidden_dim] back to original order
    """
    num_tokens, top_k = routing_weights.shape
    hidden_dim = expert_outputs.shape[-1] 
    
    # invert permutation to restore token indices
    # applying argsort the second time for inverse permutation
    inverse_sort_indices = torch.argsort(sort_indices)
    
    # restore original flattened token order
    restored_x = expert_outputs[inverse_sort_indices]
    
    # reshape to original dimensions
    restored_x = restored_x.view(num_tokens, top_k, hidden_dim)
    
    # scale by routing weights and reduce
    # reshape weights for broadcasting
    combined_x = (restored_x * routing_weights.unsqueeze(-1)).sum(dim=1)
    
    return combined_x
