#!/bin/bash
# Run ALL 5 Lineage experiments in parallel

cd "/Users/guanghan.ning/Documents/Obsidian Vault/applications/intology/Onsite_Task/OpenEvolve"

# Activate virtual environment
source .venv/bin/activate

# Set API key (inherit from parent shell or set here)
export OPENAI_API_KEY="${OPENAI_API_KEY:-your-api-key-here}"

echo "========================================"
echo "Running 5 Lineage experiments in parallel"
echo "Python: $(which python)"
echo "========================================"
echo ""

# Lineage runs 1-5
echo "[$(date +%H:%M:%S)] Starting Lineage runs 1-5..."
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config_lineage.yaml --output examples/circle_packing/output_lineage_run1 --seed 42 > examples/circle_packing/output_lineage_run1.log 2>&1 &
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config_lineage.yaml --output examples/circle_packing/output_lineage_run2 --seed 123 > examples/circle_packing/output_lineage_run2.log 2>&1 &
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config_lineage.yaml --output examples/circle_packing/output_lineage_run3 --seed 456 > examples/circle_packing/output_lineage_run3.log 2>&1 &
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config_lineage.yaml --output examples/circle_packing/output_lineage_run4 --seed 789 > examples/circle_packing/output_lineage_run4.log 2>&1 &
python openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py --config examples/circle_packing/config_lineage.yaml --output examples/circle_packing/output_lineage_run5 --seed 1024 > examples/circle_packing/output_lineage_run5.log 2>&1 &

echo ""
echo "All 5 Lineage experiments launched!"
echo ""
echo "Monitor progress:"
echo "  while true; do clear; tail -1 examples/circle_packing/output_lineage_*.log 2>/dev/null; sleep 5; done"
echo ""

# Wait for all background jobs
wait

echo ""
echo "========================================"
echo "[$(date +%H:%M:%S)] All Lineage experiments completed!"
echo "========================================"
