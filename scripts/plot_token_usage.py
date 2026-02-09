#!/usr/bin/env python3
"""
Plot token usage vs performance for OpenEvolve experiments

Usage:
    python scripts/plot_token_usage.py --path /path/to/output_dir
    python scripts/plot_token_usage.py --compare dir1 dir2 dir3 --labels baseline method1 method2
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required for plotting.")
    print("Install with: pip install matplotlib numpy")
    sys.exit(1)


def load_token_stats(path: str) -> Optional[Dict]:
    """Load token statistics from a directory"""
    json_path = os.path.join(path, "token_stats.json")
    if not os.path.exists(json_path):
        print(f"Warning: token_stats.json not found in {path}")
        return None

    with open(json_path, "r") as f:
        return json.load(f)


def plot_single_experiment(output_dir: str, save_path: Optional[str] = None) -> None:
    """Plot token usage for a single experiment"""
    data = load_token_stats(output_dir)
    if not data:
        return

    iterations = data.get("iterations", [])
    if not iterations:
        print("No iteration data found")
        return

    # Extract data
    iter_nums = [s["iteration"] for s in iterations]
    cumulative_tokens = [s["cumulative_total_tokens"] for s in iterations]
    best_scores = [s["best_score"] for s in iterations]
    tokens_per_iter = [s["total_tokens"] for s in iterations]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Token Usage Analysis\n{output_dir}", fontsize=12)

    # Plot 1: Performance vs Iteration
    ax1 = axes[0, 0]
    ax1.plot(iter_nums, best_scores, "b-", linewidth=2)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best Score")
    ax1.set_title("Performance vs Iteration")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Performance vs Cumulative Tokens
    ax2 = axes[0, 1]
    ax2.plot(cumulative_tokens, best_scores, "g-", linewidth=2)
    ax2.set_xlabel("Cumulative Tokens")
    ax2.set_ylabel("Best Score")
    ax2.set_title("Performance vs Token Usage")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Tokens per Iteration
    ax3 = axes[1, 0]
    ax3.bar(iter_nums, tokens_per_iter, alpha=0.7, color="orange")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Tokens")
    ax3.set_title("Tokens per Iteration")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cumulative Token Growth
    ax4 = axes[1, 1]
    ax4.plot(iter_nums, cumulative_tokens, "r-", linewidth=2)
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Cumulative Tokens")
    ax4.set_title("Cumulative Token Usage")
    ax4.grid(True, alpha=0.3)

    # Add summary statistics
    summary = data.get("summary", {})
    summary_text = (
        f"Total iterations: {summary.get('total_iterations', 0)}\n"
        f"Total tokens: {summary.get('cumulative_total_tokens', 0):,}\n"
        f"Avg tokens/iter: {summary.get('avg_tokens_per_iteration', 0):.1f}"
    )
    fig.text(0.02, 0.02, summary_text, fontsize=9, family="monospace",
             verticalalignment="bottom", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_comparison(
    output_dirs: List[str],
    labels: List[str],
    save_path: Optional[str] = None
) -> None:
    """Compare token usage across multiple experiments"""
    # Load all data
    all_data = []
    for path, label in zip(output_dirs, labels):
        data = load_token_stats(path)
        if data:
            all_data.append((label, data))
        else:
            print(f"Skipping {label}: no data found")

    if not all_data:
        print("No valid data to compare")
        return

    # Create figure with comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Method Comparison: Token Efficiency", fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))

    # Plot 1: Performance vs Iteration
    ax1 = axes[0]
    for (label, data), color in zip(all_data, colors):
        iterations = data.get("iterations", [])
        iter_nums = [s["iteration"] for s in iterations]
        best_scores = [s["best_score"] for s in iterations]
        ax1.plot(iter_nums, best_scores, "-", linewidth=2, label=label, color=color)

    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Best Score", fontsize=11)
    ax1.set_title("Performance vs Iteration\n(Same X-axis: iterations)", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Performance vs Cumulative Tokens
    ax2 = axes[1]
    for (label, data), color in zip(all_data, colors):
        iterations = data.get("iterations", [])
        cumulative_tokens = [s["cumulative_total_tokens"] for s in iterations]
        best_scores = [s["best_score"] for s in iterations]
        ax2.plot(cumulative_tokens, best_scores, "-", linewidth=2, label=label, color=color)

    ax2.set_xlabel("Cumulative Tokens", fontsize=11)
    ax2.set_ylabel("Best Score", fontsize=11)
    ax2.set_title("Performance vs Token Usage\n(Same X-axis: resource consumption)", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add summary table
    summary_lines = ["Method Summary:"]
    for label, data in all_data:
        summary = data.get("summary", {})
        total_tokens = summary.get("cumulative_total_tokens", 0)
        avg_tokens = summary.get("avg_tokens_per_iteration", 0)
        iterations = data.get("iterations", [])
        final_score = iterations[-1]["best_score"] if iterations else 0
        summary_lines.append(
            f"  {label}: {total_tokens:,} tokens, "
            f"{avg_tokens:.0f} avg/iter, final={final_score:.4f}"
        )

    fig.text(0.02, 0.02, "\n".join(summary_lines), fontsize=9, family="monospace",
             verticalalignment="bottom", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()


def print_summary(output_dir: str) -> None:
    """Print token usage summary to console"""
    data = load_token_stats(output_dir)
    if not data:
        return

    summary = data.get("summary", {})
    iterations = data.get("iterations", [])

    print("\n" + "=" * 60)
    print("TOKEN USAGE SUMMARY")
    print("=" * 60)
    print(f"Directory: {output_dir}")
    print("-" * 60)
    print(f"Total iterations:      {summary.get('total_iterations', 0)}")
    print(f"Total tokens:          {summary.get('cumulative_total_tokens', 0):,}")
    print(f"  - Prompt tokens:     {summary.get('cumulative_prompt_tokens', 0):,}")
    print(f"  - Completion tokens: {summary.get('cumulative_completion_tokens', 0):,}")
    print(f"Avg tokens/iteration:  {summary.get('avg_tokens_per_iteration', 0):.1f}")
    print("-" * 60)

    if iterations:
        final_score = iterations[-1]["best_score"]
        print(f"Final best score:      {final_score:.4f}")

        # Find iteration where score exceeded certain thresholds
        thresholds = [0.5, 0.7, 0.8, 0.9]
        for thresh in thresholds:
            for s in iterations:
                if s["best_score"] >= thresh:
                    print(f"Score >= {thresh:.1f} at:    "
                          f"iter {s['iteration']}, {s['cumulative_total_tokens']:,} tokens")
                    break

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Plot token usage vs performance for OpenEvolve experiments"
    )
    parser.add_argument(
        "--path", type=str,
        help="Path to single experiment output directory"
    )
    parser.add_argument(
        "--compare", type=str, nargs="+",
        help="Paths to multiple experiment directories for comparison"
    )
    parser.add_argument(
        "--labels", type=str, nargs="+",
        help="Labels for comparison plot (must match number of --compare paths)"
    )
    parser.add_argument(
        "--save", type=str,
        help="Save plot to file instead of displaying"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print text summary to console"
    )

    args = parser.parse_args()

    if args.compare:
        # Comparison mode
        if args.labels and len(args.labels) != len(args.compare):
            print("Error: Number of labels must match number of comparison paths")
            sys.exit(1)

        labels = args.labels or [Path(p).name for p in args.compare]
        plot_comparison(args.compare, labels, args.save)

        if args.summary:
            for path in args.compare:
                print_summary(path)

    elif args.path:
        # Single experiment mode
        if args.summary:
            print_summary(args.path)
        else:
            plot_single_experiment(args.path, args.save)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
