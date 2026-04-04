# Research-scale Mixture-of-Experts (MoE) & Benchmark Harness

This repository contains a high-performance, configurable, single-node Mixture-of-Experts (MoE) implementation in PyTorch, featuring custom Triton-optimized memory dispatch and multi-GPU Expert Parallelism (EP).

## 🚀 Overview

The project provides a modular framework for studying the performance characteristics of MoE layers. It is designed for systems research, focusing on routing behavior, memory permutation overheads, and multi-GPU communication scaling.

### Key Features:
- **Configurable MoE Layer:** Supports arbitrary expert counts, FFN dimensions, top-K routing (k=1, 2), and multiple activation functions (GELU, SiLU).
- **Triton Dispatch:** Custom GPU kernel in OpenAI Triton to accelerate token permutation, bypassing unoptimized PyTorch generic indexing.
- **Expert Parallelism (EP):** Distributed MoE support using NCCL `all_to_all_single` collective operations across multiple GPUs.
- **Instrumented Benchmarking:** Granular, CUDA-event-synchronized timing of every stage (Routing, Dispatch, NCCL, Expert Compute).
- **Synthetic Workloads:** Built-in support for multiple routing distributions (Natural, Uniform, Skewed/Worst-case) to stress-test load balancing.

## 🏗️ Architecture

```text
    [ Input Tokens ] -> [ Router ] -> [ Target Rank Sorting ]
                                            |
                                  +---------+---------+ (Network - NCCL)
                                  |                   |
                           [ GPU 0 (EP) ]      [ GPU 1 (EP) ]
                           [ Expert 0,1 ]      [ Expert 2,3 ]
                                  |                   |
                                  +---------+---------+ (Network - NCCL)
                                            |
                                   [ Result Combine ] -> [ Output Tokens ]
```

## 🛠️ Usage

### Quickstart (Agate Cluster)
To run the full suite (Correctness Tests -> Benchmarking -> Plotting) on 2 A100 GPUs:
```bash
sbatch scripts/run_multi_gpu.sh
```

### Manual Execution:
**Correctness Tests:**
```bash
python3 -m pytest tests/test_correctness.py
```
**Distributed Benchmark (2 GPUs):**
```bash
python3 -m torch.distributed.run --nproc_per_node=2 benchmarks/bench_multi_gpu.py
```

## 📊 Key Findings

Following our benchmarking pass on A100/L40S hardware:
1. **Network Dominance:** At moderate sequence lengths, NCCL `all_to_all` communication accounts for nearly 40-50% of the total forward pass latency.
2. **Triton Speedup:** Our custom Triton dispatch kernel provides an **~18% reduction** in memory permutation time for large sequence lengths (8192+) compared to the native PyTorch gather baseline.
3. **Dispatch Overhead:** For smaller sequence lengths, the overhead of launching kernels and calculating routing offsets is significant, showing the importance of fused gating operations.
4. **Scale Sensitivity:** Expert parallelism scales the available memory for weights, but sensitive interconnects (NVLink) are mandatory to offset the increased data movement.

## 📁 Repository Structure
- `moe/`: Core library (Router, Experts, Dispatch, Distributed).
- `benchmarks/`: Performance measurement scripts and CSV results.
- `tests/`: Extensive correctness suite (Identity tests, Top-K and Type checks).
- `scripts/`: Batch submission scripts and plotting utilities.

## 📜 Requirements
- Python 3.10+
- PyTorch 2.11.0+
- Triton 3.6.0+
- Matplotlib & Pandas (for plotting)
