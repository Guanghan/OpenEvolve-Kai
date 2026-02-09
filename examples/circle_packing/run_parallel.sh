#!/bin/bash
# Run remaining experiments in parallel (2-3 at a time to avoid rate limits)

cd "/Users/guanghan.ning/Documents/Obsidian Vault/applications/intology/Onsite_Task/OpenEvolve"

# Function to run experiment in background
run_exp() {
    config=$1
    output=$2
    seed=$3
    name=$4

    echo "[$(date +%H:%M:%S)] Starting: $name (seed=$seed)"
    python3 openevolve-run.py \
        examples/circle_packing/initial_program.py \
        examples/circle_packing/evaluator.py \
        --config "examples/circle_packing/$config" \
        --output "examples/circle_packing/$output" \
        --seed "$seed" \
        > "examples/circle_packing/${output}.log" 2>&1
    echo "[$(date +%H:%M:%S)] Completed: $name"
}

echo "========================================"
echo "Running remaining 9 experiments"
echo "Parallel: 3 at a time"
echo "========================================"

# Batch 1: Baseline runs 2-4 (3 parallel)
echo ""
echo "=== Batch 1: Baseline runs 2-4 ==="
run_exp config_baseline.yaml output_baseline_run2 123 "Baseline-2" &
run_exp config_baseline.yaml output_baseline_run3 456 "Baseline-3" &
run_exp config_baseline.yaml output_baseline_run4 789 "Baseline-4" &
wait
echo "Batch 1 completed!"

# Batch 2: Baseline run 5 + E-PUCT runs 2-3 (3 parallel)
echo ""
echo "=== Batch 2: Baseline-5 + E-PUCT 2-3 ==="
run_exp config_baseline.yaml output_baseline_run5 1024 "Baseline-5" &
run_exp config_epuct.yaml output_epuct_run2 123 "E-PUCT-2" &
run_exp config_epuct.yaml output_epuct_run3 456 "E-PUCT-3" &
wait
echo "Batch 2 completed!"

# Batch 3: E-PUCT runs 4-5 (2 parallel)
echo ""
echo "=== Batch 3: E-PUCT 4-5 ==="
run_exp config_epuct.yaml output_epuct_run4 789 "E-PUCT-4" &
run_exp config_epuct.yaml output_epuct_run5 1024 "E-PUCT-5" &
wait
echo "Batch 3 completed!"

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo ""
echo "Check logs in: examples/circle_packing/output_*.log"
echo "Analyze results: python examples/circle_packing/analyze_results.py"
echo "Plot results: python examples/circle_packing/plot_results.py"
