#!/bin/bash
# Run ablation experiments with multiple seeds to reduce randomness

set -e

cd /Users/guanghan.ning/Documents/Obsidian\ Vault/applications/intology/Onsite_Task/OpenEvolve

SEEDS=(42 123 456 789 1024)
NUM_RUNS=${#SEEDS[@]}

echo "========================================"
echo "Circle Packing Ablation Experiments"
echo "Running $NUM_RUNS experiments per config"
echo "========================================"

# Baseline experiments
echo ""
echo "=========================================="
echo "Running BASELINE experiments"
echo "=========================================="

for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    run_num=$((i + 1))
    output_dir="examples/circle_packing/output_baseline_run${run_num}"

    echo ""
    echo "--- Baseline Run $run_num/$NUM_RUNS (seed=$seed) ---"

    python3 openevolve-run.py \
        examples/circle_packing/initial_program.py \
        examples/circle_packing/evaluator.py \
        --config examples/circle_packing/config_baseline.yaml \
        --output "$output_dir" \
        --seed "$seed"

    echo "Baseline Run $run_num completed. Output: $output_dir"
done

# E-PUCT experiments
echo ""
echo "=========================================="
echo "Running E-PUCT experiments"
echo "=========================================="

for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    run_num=$((i + 1))
    output_dir="examples/circle_packing/output_epuct_run${run_num}"

    echo ""
    echo "--- E-PUCT Run $run_num/$NUM_RUNS (seed=$seed) ---"

    python3 openevolve-run.py \
        examples/circle_packing/initial_program.py \
        examples/circle_packing/evaluator.py \
        --config examples/circle_packing/config_epuct.yaml \
        --output "$output_dir" \
        --seed "$seed"

    echo "E-PUCT Run $run_num completed. Output: $output_dir"
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "To analyze results, run:"
echo "  python examples/circle_packing/analyze_results.py"
