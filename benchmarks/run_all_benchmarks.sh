#!/usr/bin/env bash
set -e

# ==============================================================================
# AxiomLM Master Reproducibility Benchmark Script
# Runs 1:1 parity evaluations across AxiomLM, Apple MLX, and llama.cpp (llama-bench).
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "========================================================================"
echo "AxiomLM Benchmark Suite: Starting 1-Click Reproducibility Runner"
echo "Target: Apple Silicon M3 Pro GPU (Metal 4)"
echo "========================================================================"

# Activate virtual environment if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the master python reproduction runner
python3 benchmarks/reproduce_all.py "$@"

echo "========================================================================"
echo "Benchmark suite finished. See benchmarks/results/latest_benchmark_run.json"
echo "========================================================================"
