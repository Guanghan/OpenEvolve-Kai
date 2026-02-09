#!/usr/bin/env python3
"""
Visualize results from multiple experiment runs with mean and std bands.
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib")
    exit(1)


def load_token_stats(output_dir: str) -> Optional[Dict]:
    """Load token statistics from a run's output directory."""
    # Try root directory first
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


def load_all_runs(config_name: str, output_pattern: str, base_dir: str) -> Dict:
    """Load data from all runs for a configuration."""
    pattern = os.path.join(base_dir, output_pattern)
    output_dirs = sorted(glob.glob(pattern))

    all_iterations = []
    all_scores = []
    all_tokens = []

    for output_dir in output_dirs:
        token_stats = load_token_stats(output_dir)
        if token_stats and "iterations" in token_stats:
            iterations = token_stats["iterations"]

            iters = [it["iteration"] for it in iterations]
            scores = [it["best_score"] for it in iterations]
            tokens = [it["cumulative_total_tokens"] for it in iterations]

            all_iterations.append(iters)
            all_scores.append(scores)
            all_tokens.append(tokens)

    return {
        "config": config_name,
        "n_runs": len(all_iterations),
        "iterations": all_iterations,
        "scores": all_scores,
        "tokens": all_tokens,
    }


def align_and_aggregate(data: Dict) -> Dict:
    """
    Align runs to common iteration points and compute mean/std.
    Handles runs with different lengths by using the minimum length.
    """
    if data["n_runs"] == 0:
        return None

    # Find common iteration range (use minimum length across all runs)
    min_len = min(len(iters) for iters in data["iterations"])

    # Truncate all runs to minimum length
    iterations = data["iterations"][0][:min_len]

    # Stack scores and tokens for aggregation
    scores_matrix = np.array([scores[:min_len] for scores in data["scores"]])
    tokens_matrix = np.array([tokens[:min_len] for tokens in data["tokens"]])

    return {
        "config": data["config"],
        "n_runs": data["n_runs"],
        "iterations": iterations,
        "scores_mean": np.mean(scores_matrix, axis=0),
        "scores_std": np.std(scores_matrix, axis=0),
        "scores_min": np.min(scores_matrix, axis=0),
        "scores_max": np.max(scores_matrix, axis=0),
        "tokens_mean": np.mean(tokens_matrix, axis=0),
        "tokens_std": np.std(tokens_matrix, axis=0),
    }


def plot_comparison(baseline_data: Dict, epuct_data: Dict, output_path: str, lineage_data: Dict = None):
    """
    Create comparison plots with error bands.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "baseline": "#2196F3",  # Blue
        "epuct": "#4CAF50",     # Green
        "lineage": "#FF9800",   # Orange
    }

    # Plot 1: Performance vs Iteration
    ax1 = axes[0]

    if baseline_data:
        iters = baseline_data["iterations"]
        mean = baseline_data["scores_mean"]
        std = baseline_data["scores_std"]

        ax1.plot(iters, mean, color=colors["baseline"], linewidth=2, label=f'Baseline (n={baseline_data["n_runs"]})')
        ax1.fill_between(iters, mean - std, mean + std, color=colors["baseline"], alpha=0.2)

    if epuct_data:
        iters = epuct_data["iterations"]
        mean = epuct_data["scores_mean"]
        std = epuct_data["scores_std"]

        ax1.plot(iters, mean, color=colors["epuct"], linewidth=2, label=f'E-PUCT (n={epuct_data["n_runs"]})')
        ax1.fill_between(iters, mean - std, mean + std, color=colors["epuct"], alpha=0.2)

    if lineage_data:
        iters = lineage_data["iterations"]
        mean = lineage_data["scores_mean"]
        std = lineage_data["scores_std"]

        ax1.plot(iters, mean, color=colors["lineage"], linewidth=2, label=f'Lineage (n={lineage_data["n_runs"]})')
        ax1.fill_between(iters, mean - std, mean + std, color=colors["lineage"], alpha=0.2)

    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Best Score (combined_score)", fontsize=12)
    ax1.set_title("Performance vs Iteration", fontsize=14)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Performance vs Token Usage
    ax2 = axes[1]

    if baseline_data:
        tokens = baseline_data["tokens_mean"]
        mean = baseline_data["scores_mean"]
        std = baseline_data["scores_std"]

        ax2.plot(tokens, mean, color=colors["baseline"], linewidth=2, label=f'Baseline (n={baseline_data["n_runs"]})')
        ax2.fill_between(tokens, mean - std, mean + std, color=colors["baseline"], alpha=0.2)

    if epuct_data:
        tokens = epuct_data["tokens_mean"]
        mean = epuct_data["scores_mean"]
        std = epuct_data["scores_std"]

        ax2.plot(tokens, mean, color=colors["epuct"], linewidth=2, label=f'E-PUCT (n={epuct_data["n_runs"]})')
        ax2.fill_between(tokens, mean - std, mean + std, color=colors["epuct"], alpha=0.2)

    if lineage_data:
        tokens = lineage_data["tokens_mean"]
        mean = lineage_data["scores_mean"]
        std = lineage_data["scores_std"]

        ax2.plot(tokens, mean, color=colors["lineage"], linewidth=2, label=f'Lineage (n={lineage_data["n_runs"]})')
        ax2.fill_between(tokens, mean - std, mean + std, color=colors["lineage"], alpha=0.2)

    ax2.set_xlabel("Cumulative Tokens", fontsize=12)
    ax2.set_ylabel("Best Score (combined_score)", fontsize=12)
    ax2.set_title("Performance vs Token Usage (Resource Efficiency)", fontsize=14)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    # Format x-axis with K suffix for tokens
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot to: {output_path}")


def plot_detailed(data: Dict, config_name: str, output_path: str):
    """
    Create detailed 4-panel plot for a single configuration showing all runs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    color = "#2196F3" if "baseline" in config_name.lower() else "#4CAF50"

    # Panel 1: All runs - Score vs Iteration
    ax1 = axes[0, 0]
    for i, (iters, scores) in enumerate(zip(data["iterations"], data["scores"])):
        ax1.plot(iters, scores, alpha=0.5, linewidth=1, label=f'Run {i+1}')

    # Plot mean with thick line
    agg = align_and_aggregate(data)
    if agg:
        ax1.plot(agg["iterations"], agg["scores_mean"], color=color, linewidth=3, label='Mean')

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best Score")
    ax1.set_title(f"{config_name}: All Runs")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Mean with std band
    ax2 = axes[0, 1]
    if agg:
        ax2.plot(agg["iterations"], agg["scores_mean"], color=color, linewidth=2)
        ax2.fill_between(
            agg["iterations"],
            agg["scores_mean"] - agg["scores_std"],
            agg["scores_mean"] + agg["scores_std"],
            color=color, alpha=0.3, label='±1 std'
        )
        ax2.fill_between(
            agg["iterations"],
            agg["scores_min"],
            agg["scores_max"],
            color=color, alpha=0.1, label='min-max'
        )

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best Score")
    ax2.set_title(f"{config_name}: Mean ± Std (n={data['n_runs']})")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Token usage per iteration
    ax3 = axes[1, 0]
    for i, (iters, tokens) in enumerate(zip(data["iterations"], data["tokens"])):
        # Calculate per-iteration tokens
        per_iter = [tokens[0]] + [tokens[j] - tokens[j-1] for j in range(1, len(tokens))]
        ax3.plot(iters, per_iter, alpha=0.5, linewidth=1)

    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Tokens per Iteration")
    ax3.set_title("Token Usage per Iteration")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Score vs Cumulative Tokens
    ax4 = axes[1, 1]
    for i, (tokens, scores) in enumerate(zip(data["tokens"], data["scores"])):
        ax4.plot(tokens, scores, alpha=0.5, linewidth=1)

    if agg:
        ax4.plot(agg["tokens_mean"], agg["scores_mean"], color=color, linewidth=3, label='Mean')

    ax4.set_xlabel("Cumulative Tokens")
    ax4.set_ylabel("Best Score")
    ax4.set_title("Score vs Token Usage")
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    ax4.legend(loc="lower right")
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"{config_name} - Detailed Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved detailed plot to: {output_path}")


def plot_final_scores_boxplot(baseline_data: Dict, epuct_data: Dict, output_path: str, lineage_data: Dict = None):
    """
    Create boxplot comparing final scores across runs.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get final scores from each run
    baseline_final = [scores[-1] for scores in baseline_data["scores"]] if baseline_data["n_runs"] > 0 else []
    epuct_final = [scores[-1] for scores in epuct_data["scores"]] if epuct_data["n_runs"] > 0 else []
    lineage_final = [scores[-1] for scores in lineage_data["scores"]] if lineage_data and lineage_data["n_runs"] > 0 else []

    data_to_plot = []
    labels = []
    colors = []

    if baseline_final:
        data_to_plot.append(baseline_final)
        labels.append(f'Baseline\n(n={len(baseline_final)})')
        colors.append("#2196F3")

    if epuct_final:
        data_to_plot.append(epuct_final)
        labels.append(f'E-PUCT\n(n={len(epuct_final)})')
        colors.append("#4CAF50")

    if lineage_final:
        data_to_plot.append(lineage_final)
        labels.append(f'Lineage\n(n={len(lineage_final)})')
        colors.append("#FF9800")

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Add individual points
        for i, (data, color) in enumerate(zip(data_to_plot, colors)):
            x = np.random.normal(i + 1, 0.04, size=len(data))
            ax.scatter(x, data, color=color, alpha=0.8, s=50, zorder=3)

    ax.set_ylabel("Final Best Score", fontsize=12)
    ax.set_title("Final Score Distribution Across Runs", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean values as text
    for i, data in enumerate(data_to_plot):
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.text(i + 1, ax.get_ylim()[1], f'μ={mean_val:.4f}\nσ={std_val:.4f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved boxplot to: {output_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "="*60)
    print("Loading experiment data...")
    print("="*60)

    # Load all runs
    baseline_raw = load_all_runs("Baseline", "output_baseline_run*", base_dir)
    epuct_raw = load_all_runs("E-PUCT", "output_epuct_run*", base_dir)
    lineage_raw = load_all_runs("Lineage", "output_lineage_run*", base_dir)

    print(f"  Baseline: {baseline_raw['n_runs']} runs found")
    print(f"  E-PUCT: {epuct_raw['n_runs']} runs found")
    print(f"  Lineage: {lineage_raw['n_runs']} runs found")

    if baseline_raw["n_runs"] == 0 and epuct_raw["n_runs"] == 0 and lineage_raw["n_runs"] == 0:
        print("\nNo experiment data found. Run experiments first.")
        return

    # Aggregate data
    baseline_agg = align_and_aggregate(baseline_raw) if baseline_raw["n_runs"] > 0 else None
    epuct_agg = align_and_aggregate(epuct_raw) if epuct_raw["n_runs"] > 0 else None
    lineage_agg = align_and_aggregate(lineage_raw) if lineage_raw["n_runs"] > 0 else None

    # Generate plots
    print("\nGenerating plots...")

    # 1. Comparison plot (all three methods)
    comparison_path = os.path.join(base_dir, "comparison_averaged.png")
    plot_comparison(baseline_agg, epuct_agg, comparison_path, lineage_agg)

    # 2. Detailed plots for each config
    if baseline_raw["n_runs"] > 0:
        baseline_detail_path = os.path.join(base_dir, "baseline_detailed.png")
        plot_detailed(baseline_raw, "Baseline", baseline_detail_path)

    if epuct_raw["n_runs"] > 0:
        epuct_detail_path = os.path.join(base_dir, "epuct_detailed.png")
        plot_detailed(epuct_raw, "E-PUCT", epuct_detail_path)

    if lineage_raw["n_runs"] > 0:
        lineage_detail_path = os.path.join(base_dir, "lineage_detailed.png")
        plot_detailed(lineage_raw, "Lineage", lineage_detail_path)

    # 3. Boxplot of final scores (all three methods)
    boxplot_path = os.path.join(base_dir, "final_scores_boxplot.png")
    plot_final_scores_boxplot(baseline_raw, epuct_raw, boxplot_path, lineage_raw)

    print("\n" + "="*60)
    print("All plots generated!")
    print("="*60)


if __name__ == "__main__":
    main()
