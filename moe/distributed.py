import torch
import torch.distributed as dist

def all_to_all_forward(send_tokens: torch.Tensor, send_expert_ids: torch.Tensor, send_counts: torch.Tensor, world_size: int):
    """
    Exchanges tokens and their routing assignments across multiple GPUs using NCCL.
    
    Args:
        send_tokens: [num_dispatched, hidden_dim] locally sorted by destination rank
        send_expert_ids: [num_dispatched, 1] locally sorted
        send_counts: [world_size] representing how many tokens this GPU wants to send to each rank
    """
    # 1. Exchange counts so we know how much buffer memory to allocate to receive
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    
    send_splits = send_counts.tolist()
    recv_splits = recv_counts.tolist()
    total_recv = sum(recv_splits)
    
    # 2. Exchange Top-K Tokens
    recv_tokens = torch.empty((total_recv, send_tokens.size(1)), dtype=send_tokens.dtype, device=send_tokens.device)
    dist.all_to_all_single(
        recv_tokens,
        send_tokens,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits
    )
    
    # 3. Exchange Expert Assignments
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
    Sends the processed locally-computed expert representations back to the original routing GPUs.
    Notice the splits are intentionally reversed.
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
        output_split_sizes=send_splits,   # Memory space originally used relative to our rank's token origin
        input_split_sizes=recv_splits     # Memory we just computed on this rank
    )
    return send_back_tokens
