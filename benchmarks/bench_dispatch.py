import torch
import time
from moe.configs import MoEConfig
from moe.dispatch import pt_dispatch
try:
    from moe.triton_dispatch import triton_dispatch
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def benchmark_pt_dispatch():
    """Benchmark raw PyTorch MoE memory dispatch bandwidth."""
    # Settings sweeps
    seq_lens = [512, 1024, 2048, 4096, 8192]
    hidden_dim = 1024
    num_experts = 8
    top_k = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Benchmarking PyTorch Native MoE Dispatch vs Triton on {device.upper()}")
    print("-" * 65)
    print(f"{'Seq Len':<10} | {'Hidden':<8} | {'Experts':<8} | {'TopK':<5} | {'PT (ms)':<10} | {'Triton (ms)':<10}")
    print("-" * 65)
    
    for seq_len in seq_lens:
        x = torch.randn(seq_len, hidden_dim, device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        # Random routing assignments
        experts = torch.randint(0, num_experts, (seq_len, top_k), device=device)
        
        # Warmup
        for _ in range(5):
            _ = pt_dispatch(x, experts, num_experts)
            
        if device == "cuda":
            torch.cuda.synchronize()
            
        start = time.time()
        iters = 100
        for _ in range(iters):
            _ = pt_dispatch(x, experts, num_experts)
            
        if device == "cuda":
            torch.cuda.synchronize()
            
        end = time.time()
        avg_ms_pt = ((end - start) / iters) * 1000
        
        # Benchmark Triton
        avg_ms_triton = "N/A"
        if HAS_TRITON and device == "cuda":
            # Warmup
            for _ in range(5):
                _ = triton_dispatch(x, experts, num_experts)
            torch.cuda.synchronize()
            
            start_triton = time.time()
            for _ in range(iters):
                _ = triton_dispatch(x, experts, num_experts)
            torch.cuda.synchronize()
            end_triton = time.time()
            
            avg_ms_triton = f"{((end_triton - start_triton) / iters) * 1000:.3f}"
            
        print(f"{seq_len:<10} | {hidden_dim:<8} | {num_experts:<8} | {top_k:<5} | {avg_ms_pt:.3f} ms | {avg_ms_triton} ms")


if __name__ == "__main__":
    benchmark_pt_dispatch()
