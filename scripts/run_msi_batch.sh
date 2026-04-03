#!/bin/bash
#SBATCH --job-name=moe_bench
#SBATCH --output=moe_results_%j.log
#SBATCH --partition=a100-4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -e

# 1. Environment Setup
echo "--- Initializing Environment ---"
module load python3
source activate moe_project

# Ensure pytest is installed
if ! python3 -c "import pytest" &>/dev/null; then
    echo "Installing missing dependency: pytest..."
    pip install pytest
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)

# 2. Correctness Phase (Fail job early if math is broken)
echo "--- Running Correctness Tests ---"
python3 -m pytest tests/test_correctness.py -v

# 3. Performance Phase (Only runs if pass tests)
echo "--- Running Performance Benchmarks ---"
python3 benchmarks/bench_dispatch.py

echo "--- Job Complete ---"
