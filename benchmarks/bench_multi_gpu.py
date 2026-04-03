import torch
import torch.distributed as dist
import os
import time
from moe.configs import MoEConfig
from moe.multi_gpu_layer import DistributedMoELayer

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def benchmark_multi_gpu():
    local_rank = setup()
    world_size = dist.get_world_size()
    
    seq_lens = [512, 1024, 2048, 4096, 8192]
    hidden_dim = 1024
    num_experts = 8 # Total across all GPUs
    top_k = 2
    
    config = MoEConfig(
        hidden_dim=hidden_dim, 
        num_experts=num_experts, 
        top_k=top_k, 
        device="cuda"
    )
    layer = DistributedMoELayer(config).to(local_rank)
    
    if local_rank == 0:
        print(f"Benchmarking Multi-GPU EP MoE via Distributed NCCL on {world_size} GPUs")
        print("-" * 75)
        print(f"{'Seq Len':<10} | {'Hidden':<8} | {'Total Exp':<10} | {'TopK':<5} | {'Latency (ms)':<15}")
        print("-" * 75)
        
    for seq_len in seq_lens:
        x = torch.randn(seq_len, hidden_dim, device=local_rank, dtype=torch.bfloat16)
        
        # Warmup
        for _ in range(3):
            _ = layer(x)
            
        dist.barrier()
        torch.cuda.synchronize()
        start = time.time()
        
        iters = 50
        for _ in range(iters):
            _ = layer(x)
            
        dist.barrier()
        torch.cuda.synchronize()
        end = time.time()
        
        if local_rank == 0:
            avg_ms = ((end - start) / iters) * 1000
            print(f"{seq_len:<10} | {hidden_dim:<8} | {num_experts:<10} | {top_k:<5} | {avg_ms:.3f} ms")
            
    cleanup()

if __name__ == "__main__":
    benchmark_multi_gpu()
