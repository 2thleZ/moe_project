#!/bin/bash
#SBATCH --job-name=moe_ep
#SBATCH --output=moe_ep_%j.log
#SBATCH --partition=a100-4
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -e

echo "--- Initializing Environment ---"
module load python3
source activate moe_project

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "--- Running Distributed Benchmarks (2 GPUs) ---"
python3 -m torch.distributed.run --nproc_per_node=2 benchmarks/bench_multi_gpu.py

echo "--- Job Complete ---"
