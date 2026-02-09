#!/bin/bash
# Run ALL remaining 9 experiments in parallel

cd "/Users/guanghan.ning/Documents/Obsidian Vault/applications/intology/Onsite_Task/OpenEvolve"

# Activate virtual environment
source .venv/bin/activate

# Set API key (inherit from parent shell or set here)
export OPENAI_API_KEY="${OPENAI_API_KEY:-your-api-key-here}"

echo "========================================"
echo "Running ALL 9 experiments in parallel"
echo "Python: $(which python)"
echo "========================================"
echo ""

# Baseline runs 2-5
echo "[$(date +%H:%M:%S)] Starting Baseline runs 2-5..."
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config_baseline.yaml --output examples/circle_packing/output_baseline_run2 --seed 123 > examples/circle_packing/output_baseline_run2.log 2>&1 &
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config_baseline.yaml --output examples/circle_packing/output_baseline_run3 --seed 456 > examples/circle_packing/output_baseline_run3.log 2>&1 &
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config_baseline.yaml --output examples/circle_packing/output_baseline_run4 --seed 789 > examples/circle_packing/output_baseline_run4.log 2>&1 &
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config_baseline.yaml --output examples/circle_packing/output_baseline_run5 --seed 1024 > examples/circle_packing/output_baseline_run5.log 2>&1 &

# E-PUCT runs 2-5
echo "[$(date +%H:%M:%S)] Starting E-PUCT runs 2-5..."
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config_epuct.yaml --output examples/circle_packing/output_epuct_run2 --seed 123 > examples/circle_packing/output_epuct_run2.log 2>&1 &
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config_epuct.yaml --output examples/circle_packing/output_epuct_run3 --seed 456 > examples/circle_packing/output_epuct_run3.log 2>&1 &
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config_epuct.yaml --output examples/circle_packing/output_epuct_run4 --seed 789 > examples/circle_packing/output_epuct_run4.log 2>&1 &
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config_epuct.yaml --output examples/circle_packing/output_epuct_run5 --seed 1024 > examples/circle_packing/output_epuct_run5.log 2>&1 &

echo ""
echo "All 9 experiments launched!"
echo ""
echo "Monitor progress:"
echo "  while true; do clear; tail -1 examples/circle_packing/output_*.log 2>/dev/null; sleep 5; done"
echo ""

# Wait for all background jobs
wait

echo ""
echo "========================================"
echo "[$(date +%H:%M:%S)] All experiments completed!"
echo "========================================"
echo ""
echo "Analyze: python examples/circle_packing/analyze_results.py"
echo "Plot:    python examples/circle_packing/plot_results.py"
