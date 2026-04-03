import torch
import triton
import triton.language as tl

@triton.jit
def gather_kernel(
    x_ptr,               # *Pointer to input [num_tokens, hidden_dim]
    out_ptr,             # *Pointer to output [num_tokens * top_k, hidden_dim]
    sort_indices_ptr,    # *Pointer to sorted index array [num_tokens * top_k]
    hidden_dim: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    A custom Triton kernel to efficiently permute hidden states of tokens.
    PyTorch's advanced indexing `x[indices]` can be highly unoptimized.
    This kernel reads from `x` according to `sort_indices` and writes sequentially to `out`.
    """
    # Program ID represents the output token index
    pid = tl.program_id(axis=0) 
    
    # Load the original token index from the sorted indices array
    # If top_k > 1, sort_indices ranges from 0 to (num_tokens * top_k - 1)
    # The actual token in x is original_index // top_k
    flat_idx = tl.load(sort_indices_ptr + pid)
    token_idx = flat_idx // top_k
    
    # Offsets for memory block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    
    # Read the row from input
    x_ptrs = x_ptr + (token_idx * hidden_dim) + offsets
    row = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Write the row sequentially to output buffer
    out_ptrs = out_ptr + (pid * hidden_dim) + offsets
    tl.store(out_ptrs, row, mask=mask)


def triton_dispatch(x: torch.Tensor, selected_experts: torch.Tensor, num_experts: int):
    """
    Triton-accelerated MoE dispatch memory permutation.
    """
    num_tokens, top_k = selected_experts.shape
    hidden_dim = x.shape[1]
    
    # 1. Compute offsets in PyTorch using fast argsort
    flat_experts = selected_experts.view(-1)
    sort_indices = torch.argsort(flat_experts).to(torch.int32)
    expert_counts = torch.bincount(flat_experts, minlength=num_experts)
    
    # 2. Allocate output memory
    dispatched_x = torch.empty((num_tokens * top_k, hidden_dim), device=x.device, dtype=x.dtype)
    
    # 3. Launch Triton Kernel
    # Grid size = total output rows
    grid = (num_tokens * top_k,)
    
    # Adjust BLOCK_SIZE to nearest power of 2 >= hidden_dim
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    
    gather_kernel[grid](
        x,
        dispatched_x,
        sort_indices,
        hidden_dim,
        top_k,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return dispatched_x, expert_counts, sort_indices

