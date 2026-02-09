#!/usr/bin/env python3
"""
Analyze results from multiple experiment runs and compute statistics.
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

def load_best_program_info(output_dir: str) -> Dict:
    """Load best program info from a run's output directory."""
    # Try checkpoint directories first (most recent checkpoint has final result)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint_*")),
            key=lambda x: int(x.split("_")[-1])
        )
        if checkpoints:
            info_path = os.path.join(checkpoints[-1], "best_program_info.json")
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    return json.load(f)

    # Fallback to root output directory
    info_path = os.path.join(output_dir, "best_program_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            return json.load(f)

    return None


def load_token_stats(output_dir: str) -> Dict:
    """Load token statistics from a run's output directory."""
    token_path = os.path.join(output_dir, "token_stats.json")
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            return json.load(f)

    # Try checkpoint directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint_*")),
            key=lambda x: int(x.split("_")[-1])
        )
        if checkpoints:
            token_path = os.path.join(checkpoints[-1], "token_stats.json")
            if os.path.exists(token_path):
                with open(token_path, "r") as f:
                    return json.load(f)

    return None


def analyze_config(config_name: str, output_pattern: str, base_dir: str) -> Dict:
    """Analyze all runs for a given configuration."""
    results = {
        "config": config_name,
        "runs": [],
        "best_scores": [],
        "total_tokens": [],
    }

    # Find all matching output directories
    pattern = os.path.join(base_dir, output_pattern)
    output_dirs = sorted(glob.glob(pattern))

    for output_dir in output_dirs:
        run_name = os.path.basename(output_dir)

        # Load best program info
        best_info = load_best_program_info(output_dir)
        if best_info:
            metrics = best_info.get("metrics", {})
            best_score = metrics.get("combined_score", 0)
            results["best_scores"].append(best_score)

            run_data = {
                "run": run_name,
                "best_score": best_score,
                "iteration": best_info.get("current_iteration", 0),
            }

            # Load token stats
            token_stats = load_token_stats(output_dir)
            if token_stats:
                summary = token_stats.get("summary", {})
                total_tokens = summary.get("cumulative_total_tokens", 0)
                results["total_tokens"].append(total_tokens)
                run_data["total_tokens"] = total_tokens

            results["runs"].append(run_data)

    # Compute statistics
    if results["best_scores"]:
        results["stats"] = {
            "mean_score": statistics.mean(results["best_scores"]),
            "std_score": statistics.stdev(results["best_scores"]) if len(results["best_scores"]) > 1 else 0,
            "min_score": min(results["best_scores"]),
            "max_score": max(results["best_scores"]),
            "n_runs": len(results["best_scores"]),
        }

        if results["total_tokens"]:
            results["stats"]["mean_tokens"] = statistics.mean(results["total_tokens"])
            results["stats"]["std_tokens"] = statistics.stdev(results["total_tokens"]) if len(results["total_tokens"]) > 1 else 0

    return results


def print_results(results: Dict):
    """Print results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Configuration: {results['config']}")
    print(f"{'='*60}")

    if not results["runs"]:
        print("  No completed runs found.")
        return

    print(f"\nIndividual Runs:")
    print(f"  {'Run':<20} {'Best Score':<15} {'Tokens':<15}")
    print(f"  {'-'*50}")
    for run in results["runs"]:
        tokens_str = f"{run.get('total_tokens', 'N/A'):,}" if run.get('total_tokens') else "N/A"
        print(f"  {run['run']:<20} {run['best_score']:<15.4f} {tokens_str:<15}")

    if "stats" in results:
        stats = results["stats"]
        print(f"\nStatistics (n={stats['n_runs']}):")
        print(f"  Best Score:  {stats['mean_score']:.4f} +/- {stats['std_score']:.4f}")
        print(f"  Score Range: [{stats['min_score']:.4f}, {stats['max_score']:.4f}]")
        if "mean_tokens" in stats:
            print(f"  Total Tokens: {stats['mean_tokens']:,.0f} +/- {stats['std_tokens']:,.0f}")


def main():
    # Base directory for circle packing experiments
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "="*60)
    print("Circle Packing Experiment Analysis")
    print("="*60)

    # Analyze baseline runs
    baseline_results = analyze_config(
        "Baseline (No Improvements)",
        "output_baseline_run*",
        base_dir
    )
    print_results(baseline_results)

    # Analyze E-PUCT runs
    epuct_results = analyze_config(
        "E-PUCT Selector",
        "output_epuct_run*",
        base_dir
    )
    print_results(epuct_results)

    # Comparison
    if baseline_results.get("stats") and epuct_results.get("stats"):
        print("\n" + "="*60)
        print("Comparison Summary")
        print("="*60)

        baseline_mean = baseline_results["stats"]["mean_score"]
        epuct_mean = epuct_results["stats"]["mean_score"]

        improvement = ((epuct_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0

        print(f"\n  {'Config':<25} {'Mean Score':<15} {'Std Dev':<15}")
        print(f"  {'-'*55}")
        print(f"  {'Baseline':<25} {baseline_mean:<15.4f} {baseline_results['stats']['std_score']:<15.4f}")
        print(f"  {'E-PUCT':<25} {epuct_mean:<15.4f} {epuct_results['stats']['std_score']:<15.4f}")
        print(f"\n  Improvement: {improvement:+.2f}%")

        # Token efficiency comparison
        if "mean_tokens" in baseline_results["stats"] and "mean_tokens" in epuct_results["stats"]:
            baseline_tokens = baseline_results["stats"]["mean_tokens"]
            epuct_tokens = epuct_results["stats"]["mean_tokens"]

            print(f"\n  Token Usage:")
            print(f"  {'Baseline':<25} {baseline_tokens:>15,.0f}")
            print(f"  {'E-PUCT':<25} {epuct_tokens:>15,.0f}")

    # Save results to JSON
    output_path = os.path.join(base_dir, "experiment_summary.json")
    with open(output_path, "w") as f:
        json.dump({
            "baseline": baseline_results,
            "epuct": epuct_results,
        }, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
