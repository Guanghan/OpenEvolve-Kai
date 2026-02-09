"""
Token usage tracking for OpenEvolve

Tracks LLM token consumption across iterations to enable:
1. Resource usage comparison between methods
2. X-axis options: iterations OR cumulative tokens
3. Cost estimation and optimization
"""

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage for a single LLM call"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage objects"""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            model=self.model or other.model,
        )


@dataclass
class IterationTokenStats:
    """Token statistics for a single iteration"""
    iteration: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0
    cumulative_total_tokens: int = 0
    best_score: float = 0.0
    timestamp: float = 0.0
    model: str = ""


class TokenTracker:
    """
    Tracks token usage across the evolution run

    Features:
    - Per-iteration token tracking
    - Cumulative totals
    - JSON/CSV export for analysis
    - Thread-safe for parallel processing
    """

    def __init__(self):
        # Use RLock (reentrant lock) to allow nested lock acquisitions
        # This prevents deadlock when save_json() calls get_stats() which both acquire lock
        self._lock = threading.RLock()
        self._iteration_stats: List[IterationTokenStats] = []
        self._cumulative_prompt_tokens: int = 0
        self._cumulative_completion_tokens: int = 0
        self._cumulative_total_tokens: int = 0

    def record_iteration(
        self,
        iteration: int,
        token_usage: Optional[TokenUsage] = None,
        best_score: float = 0.0,
        timestamp: float = 0.0,
    ) -> None:
        """
        Record token usage for an iteration

        Args:
            iteration: Iteration number
            token_usage: Token usage for this iteration (None if no LLM call)
            best_score: Best combined_score at this iteration
            timestamp: Unix timestamp
        """
        with self._lock:
            prompt_tokens = token_usage.prompt_tokens if token_usage else 0
            completion_tokens = token_usage.completion_tokens if token_usage else 0
            total_tokens = token_usage.total_tokens if token_usage else 0
            model = token_usage.model if token_usage else ""

            # Update cumulative totals
            self._cumulative_prompt_tokens += prompt_tokens
            self._cumulative_completion_tokens += completion_tokens
            self._cumulative_total_tokens += total_tokens

            stats = IterationTokenStats(
                iteration=iteration,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cumulative_prompt_tokens=self._cumulative_prompt_tokens,
                cumulative_completion_tokens=self._cumulative_completion_tokens,
                cumulative_total_tokens=self._cumulative_total_tokens,
                best_score=best_score,
                timestamp=timestamp,
                model=model,
            )

            self._iteration_stats.append(stats)

            logger.debug(
                f"Iteration {iteration}: tokens={total_tokens} "
                f"(prompt={prompt_tokens}, completion={completion_tokens}), "
                f"cumulative={self._cumulative_total_tokens}"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        with self._lock:
            return {
                "total_iterations": len(self._iteration_stats),
                "cumulative_prompt_tokens": self._cumulative_prompt_tokens,
                "cumulative_completion_tokens": self._cumulative_completion_tokens,
                "cumulative_total_tokens": self._cumulative_total_tokens,
                "avg_tokens_per_iteration": (
                    self._cumulative_total_tokens / len(self._iteration_stats)
                    if self._iteration_stats else 0
                ),
                "avg_prompt_tokens": (
                    self._cumulative_prompt_tokens / len(self._iteration_stats)
                    if self._iteration_stats else 0
                ),
                "avg_completion_tokens": (
                    self._cumulative_completion_tokens / len(self._iteration_stats)
                    if self._iteration_stats else 0
                ),
            }

    def save_json(self, output_path: str) -> None:
        """
        Save token statistics to JSON file

        Output format:
        {
            "summary": {...},
            "iterations": [...]
        }
        """
        with self._lock:
            data = {
                "summary": self.get_stats(),
                "iterations": [asdict(s) for s in self._iteration_stats],
            }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved token statistics to {output_path}")

    def save_csv(self, output_path: str) -> None:
        """
        Save token statistics to CSV file for easy plotting

        Columns: iteration, prompt_tokens, completion_tokens, total_tokens,
                 cumulative_prompt_tokens, cumulative_completion_tokens,
                 cumulative_total_tokens, best_score, timestamp, model
        """
        import csv

        with self._lock:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                # Header
                writer.writerow([
                    "iteration",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "cumulative_prompt_tokens",
                    "cumulative_completion_tokens",
                    "cumulative_total_tokens",
                    "best_score",
                    "timestamp",
                    "model",
                ])
                # Data rows
                for stats in self._iteration_stats:
                    writer.writerow([
                        stats.iteration,
                        stats.prompt_tokens,
                        stats.completion_tokens,
                        stats.total_tokens,
                        stats.cumulative_prompt_tokens,
                        stats.cumulative_completion_tokens,
                        stats.cumulative_total_tokens,
                        stats.best_score,
                        stats.timestamp,
                        stats.model,
                    ])

        logger.info(f"Saved token statistics CSV to {output_path}")

    def save(self, output_dir: str) -> None:
        """Save both JSON and CSV formats"""
        json_path = os.path.join(output_dir, "token_stats.json")
        csv_path = os.path.join(output_dir, "token_stats.csv")

        self.save_json(json_path)
        self.save_csv(csv_path)

    def load(self, json_path: str) -> None:
        """Load token statistics from JSON file (for resume from checkpoint)"""
        if not os.path.exists(json_path):
            logger.warning(f"Token stats file not found: {json_path}")
            return

        with open(json_path, "r") as f:
            data = json.load(f)

        with self._lock:
            self._iteration_stats = [
                IterationTokenStats(**s) for s in data.get("iterations", [])
            ]

            # Recalculate cumulative totals from loaded data
            if self._iteration_stats:
                last_stats = self._iteration_stats[-1]
                self._cumulative_prompt_tokens = last_stats.cumulative_prompt_tokens
                self._cumulative_completion_tokens = last_stats.cumulative_completion_tokens
                self._cumulative_total_tokens = last_stats.cumulative_total_tokens

        logger.info(f"Loaded token statistics from {json_path}")

    def get_iteration_data_for_plotting(self) -> Dict[str, List]:
        """
        Get data formatted for plotting

        Returns dict with lists that can be used directly for matplotlib:
        - iterations: [0, 1, 2, ...]
        - cumulative_tokens: [100, 250, 400, ...]
        - best_scores: [0.1, 0.2, 0.25, ...]
        """
        with self._lock:
            return {
                "iterations": [s.iteration for s in self._iteration_stats],
                "cumulative_tokens": [s.cumulative_total_tokens for s in self._iteration_stats],
                "best_scores": [s.best_score for s in self._iteration_stats],
                "tokens_per_iteration": [s.total_tokens for s in self._iteration_stats],
            }


# Global token tracker instance for use in worker processes
_global_token_tracker: Optional[TokenTracker] = None


def get_global_token_tracker() -> Optional[TokenTracker]:
    """Get the global token tracker instance"""
    return _global_token_tracker


def set_global_token_tracker(tracker: TokenTracker) -> None:
    """Set the global token tracker instance"""
    global _global_token_tracker
    _global_token_tracker = tracker
