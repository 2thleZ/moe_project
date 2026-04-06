#!/bin/bash
#SBATCH --job-name=moe_nsys
#SBATCH --output=moe_nsys_%j.log
#SBATCH --partition=a100-4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -e

# environment setup
echo "--- Initializing Environment ---"
module load python3
module load cuda
source activate moe_project

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "--- Running Nsight Systems Profiler ---"
# We profile the single-GPU dispatch script to cleanly see the Triton vs PyTorch kernel differences
# -t cuda,nvtx: Traces CUDA API, GPU workloads, and PyTorch NVTX markers
# -s none: Disable CPU sampling for a cleaner GPU-focused trace
# --force-overwrite: Overwrite old profile if it exists
nsys profile \
    -t cuda,nvtx \
    -s none \
    -o benchmarks/dispatch_profile \
    --force-overwrite true \
    python3 benchmarks/bench_dispatch.py

echo "--- Profiling Complete ---"
echo "Download benchmarks/dispatch_profile.nsys-rep to view in Nsight Systems UI natively on your machine."
