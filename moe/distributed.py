import torch
import torch.distributed as dist

def all_to_all_forward(send_tokens: torch.Tensor, send_expert_ids: torch.Tensor, send_counts: torch.Tensor, world_size: int):
    """
    exchanges tokens and routing assignments across multiple GPUs using NCCL.
    
    Args:
        send_tokens: [num_dispatched, hidden_dim] locally sorted by destination rank
        send_expert_ids: [num_dispatched, 1] locally sorted
        send_counts: [world_size] representing how many tokens this GPU wants to send to each rank
    """
    # exchange counts to pre-allocate receive buffers
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    
    send_splits = send_counts.tolist()
    recv_splits = recv_counts.tolist()
    total_recv = sum(recv_splits)
    
    # exchange top-k tokens
    recv_tokens = torch.empty((total_recv, send_tokens.size(1)), dtype=send_tokens.dtype, device=send_tokens.device)
    dist.all_to_all_single(
        recv_tokens,
        send_tokens,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits
    )
    
    # exchange expert assignments
    recv_expert_ids = torch.empty((total_recv, send_expert_ids.size(1)), dtype=send_expert_ids.dtype, device=send_expert_ids.device)
    dist.all_to_all_single(
        recv_expert_ids,
        send_expert_ids,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits
    )
    
    return recv_tokens, recv_expert_ids, recv_counts, send_splits, recv_splits

def all_to_all_backward(recv_expert_outputs: torch.Tensor, send_splits: list[int], recv_splits: list[int]):
    """
    returns processed expert representations to original routing GPUs.
    routing splits are inverted for the backward pass.
    """
    total_send_back = sum(send_splits)
    send_back_tokens = torch.empty(
        (total_send_back, recv_expert_outputs.size(1)),
        dtype=recv_expert_outputs.dtype,
        device=recv_expert_outputs.device
    )
    
    dist.all_to_all_single(
        send_back_tokens,
        recv_expert_outputs,
        output_split_sizes=send_splits,   # memory layout of original tokens
        input_split_sizes=recv_splits     # memory layout of locally computed tokens
    )
    return send_back_tokens
