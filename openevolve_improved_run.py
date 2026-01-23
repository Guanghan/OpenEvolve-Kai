#!/usr/bin/env python3
"""
OpenEvolve with Improvements Runner (Deep Integration)

This script runs OpenEvolve with the improvement modules enabled:
- BidirectionalReflectionMemory
- EPUCTSelector
- LineageTracker

Usage:
    python openevolve_improved_run.py initial_program.py evaluator.py --config config.yaml --output output_dir

The improvements will be loaded from the 'improvements' section of the config file.

Deep Integration:
- ImprovementsManager is initialized in OpenEvolve.__init__
- ProcessParallelController calls on_iteration_end() for each result
- Prompt sampler is patched to include reflections
- All stats are collected through the result processing loop
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from openevolve.config import load_config
from openevolve.controller import OpenEvolve

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OpenEvolve with improvement modules (deep integration)"
    )
    parser.add_argument(
        "initial_program",
        type=str,
        help="Path to the initial program file",
    )
    parser.add_argument(
        "evaluator",
        type=str,
        help="Path to the evaluator file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations (overrides config)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def load_improvements_config(config_path: Optional[str]) -> Optional[dict]:
    """Load improvements configuration from YAML file as raw dict"""
    if not config_path or not os.path.exists(config_path):
        logger.info("No improvements config found, using defaults")
        return None

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if "improvements" not in config_dict:
        logger.info("No 'improvements' section in config, improvements disabled")
        return None

    improvements_dict = config_dict["improvements"]

    if not improvements_dict.get("enabled", True):
        logger.info("Improvements disabled in config")
        return None

    return improvements_dict


async def main_async():
    args = parse_args()

    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load configuration
    config = load_config(args.config)

    # Override iterations if specified
    if args.iterations:
        config.max_iterations = args.iterations

    # Load improvements configuration as raw dict
    improvements_config = load_improvements_config(args.config)

    # Determine output directory
    output_dir = args.output or os.path.join(
        os.path.dirname(args.initial_program), "openevolve_improved_output"
    )

    # Log improvements status
    if improvements_config:
        enabled_modules = []
        if improvements_config.get("reflection_memory", {}).get("enabled", True):
            enabled_modules.append("ReflectionMemory")
        if improvements_config.get("epuct_selector", {}).get("enabled", True):
            enabled_modules.append("E-PUCT")
        if improvements_config.get("lineage_tracker", {}).get("enabled", True):
            enabled_modules.append("LineageTracker")

        print("\n" + "=" * 60)
        print("OPENEVOLVE WITH IMPROVEMENTS (DEEP INTEGRATION)")
        print("=" * 60)
        print(f"Enabled modules: {', '.join(enabled_modules)}")
        print(f"Output directory: {output_dir}")
        print("=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("OPENEVOLVE (BASELINE - NO IMPROVEMENTS)")
        print("=" * 60 + "\n")

    # Create OpenEvolve with improvements config
    # The improvements_config is passed as a raw dict to OpenEvolve.__init__
    # which will initialize the ImprovementsManager internally
    evolve = OpenEvolve(
        initial_program_path=args.initial_program,
        evaluation_file=args.evaluator,
        config=config,
        output_dir=output_dir,
        improvements_config=improvements_config,  # NEW: Pass to controller
    )

    # Run evolution
    best_program = await evolve.run(
        iterations=args.iterations,
        checkpoint_path=args.checkpoint,
    )

    # Print final result
    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)

    if best_program:
        print(f"\nBest program found: {best_program.id}")
        print(f"Metrics: {best_program.metrics}")
        print(f"Iteration found: {best_program.iteration_found}")
        print(f"\nBest program saved to: {os.path.join(output_dir, 'best')}")
    else:
        print("\nNo valid programs found during evolution")

    print("=" * 60 + "\n")

    return best_program


def main():
    """Synchronous entry point"""
    return asyncio.run(main_async())


if __name__ == "__main__":
    main()
