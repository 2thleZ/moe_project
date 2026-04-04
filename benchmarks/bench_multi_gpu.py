import torch
import torch.distributed as dist
import os
import time
import csv
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
        device="cuda",
        dtype=torch.bfloat16
    )
    layer = DistributedMoELayer(config).to(device=local_rank, dtype=config.dtype)
    
    results = []
    
    if local_rank == 0:
        print(f"Benchmarking Multi-GPU EP MoE (Expert Parallelism) on {world_size} GPUs")
        print("-" * 110)
        header = f"{'Seq Len':<10} | {'Total(ms)':<10} | {'Router':<10} | {'Dispatch':<10} | {'NCCL FW':<10} | {'Compute':<10} | {'NCCL BW':<10} | {'Combine':<10}"
        print(header)
        print("-" * 110)
        
    for seq_len in seq_lens:
        x = torch.randn(seq_len, hidden_dim, device=local_rank, dtype=config.dtype)
        
        # Warmup
        for _ in range(5):
            _ = layer(x)
            
        dist.barrier()
        torch.cuda.synchronize()
        
        # Timing accumulation
        accum_timings = {
            "routing": 0.0, "dispatch": 0.0, "nccl_fw": 0.0,
            "expert_compute": 0.0, "nccl_bw": 0.0, "combine": 0.0
        }
        total_start = time.time()
        
        iters = 50
        for _ in range(iters):
            _, iter_timings = layer(x, return_timings=True)
            for k, v in iter_timings.items():
                accum_timings[k] += v
            
        dist.barrier()
        torch.cuda.synchronize()
        total_end = time.time()
        
        if local_rank == 0:
            avg_total = ((total_end - total_start) / iters) * 1000
            avg_breakdown = {k: v / iters for k, v in accum_timings.items()}
            
            row = f"{seq_len:<10} | {avg_total:<10.3f} | {avg_breakdown['routing']:<10.3f} | {avg_breakdown['dispatch']:<10.3f} | {avg_breakdown['nccl_fw']:<10.3f} | {avg_breakdown['expert_compute']:<10.3f} | {avg_breakdown['nccl_bw']:<10.3f} | {avg_breakdown['combine']:<10.3f}"
            print(row)
            
            results.append({
                "seq_len": seq_len,
                "total_ms": avg_total,
                **avg_breakdown
            })
            
    if local_rank == 0:
        # Save to CSV for plotting
        with open("benchmarks/results_breakdown.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seq_len", "total_ms", "routing", "dispatch", "nccl_fw", "expert_compute", "nccl_bw", "combine"])
            writer.writeheader()
            writer.writerows(results)
        print("\nResults saved to benchmarks/results_breakdown.csv")
            
    cleanup()

if __name__ == "__main__":
    benchmark_multi_gpu()
