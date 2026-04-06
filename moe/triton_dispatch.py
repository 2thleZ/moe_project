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
    # program ID identifies the output token index
    pid = tl.program_id(axis=0) 
    
    # load original token index from sorted indices array
    # offset ranges account for top_k duplication
    # derive actual token input index
    flat_idx = tl.load(sort_indices_ptr + pid)
    token_idx = flat_idx // top_k
    
    # compute offsets for memory block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    
    # fetch row from input tensor
    x_ptrs = x_ptr + (token_idx * hidden_dim) + offsets
    row = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # store row sequentially to output buffer
    out_ptrs = out_ptr + (pid * hidden_dim) + offsets
    tl.store(out_ptrs, row, mask=mask)


def triton_dispatch(x: torch.Tensor, selected_experts: torch.Tensor, num_experts: int):
    """
    Triton-accelerated MoE dispatch memory permutation.
    """
    num_tokens, top_k = selected_experts.shape
    hidden_dim = x.shape[1]
    
    # compute permutation indices
    flat_experts = selected_experts.view(-1)
    sort_indices = torch.argsort(flat_experts).to(torch.int32)
    expert_counts = torch.bincount(flat_experts, minlength=num_experts)
    
    # allocate output buffer
    dispatched_x = torch.empty((num_tokens * top_k, hidden_dim), device=x.device, dtype=x.dtype)
    
    # dispatch triton kernel
    # define grid bounds
    grid = (num_tokens * top_k,)
    
    # align block size to nearest power of 2
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

