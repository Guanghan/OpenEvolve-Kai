"""
Improvements Integration Module for OpenEvolve

This module provides a unified interface for all improvement modules:
- BidirectionalReflectionMemory: Learn from failures and successes
- EPUCTSelector: PUCT-inspired parent selection
- LineageTracker: Ancestry tracking with soft backtracking

Usage:
    from openevolve.improvements import ImprovementsManager, ImprovementsConfig

    config = ImprovementsConfig.from_yaml("config.yaml")
    manager = ImprovementsManager(config)

    # Hook into evolution loop
    manager.on_iteration_start(iteration, parent, context)
    manager.on_iteration_end(iteration, parent, child, success)
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from openevolve.reflection_memory import (
    BidirectionalReflectionMemory,
    FailureLesson,
    ReflectionMemoryConfig,
    SuccessPattern,
)
from openevolve.epuct_selector import (
    EPUCTSelector,
    EPUCTSelectorConfig,
    LowLogProbSelector,
)
from openevolve.lineage_tracker import (
    LineageTracker,
    LineageTrackerConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class ImprovementsConfig:
    """Configuration for all improvements"""

    # Master switch
    enabled: bool = True

    # Reflection Memory config
    reflection_memory: ReflectionMemoryConfig = field(
        default_factory=ReflectionMemoryConfig
    )

    # E-PUCT Selector config
    epuct_selector: EPUCTSelectorConfig = field(
        default_factory=EPUCTSelectorConfig
    )

    # Lineage Tracker config
    lineage_tracker: LineageTrackerConfig = field(
        default_factory=LineageTrackerConfig
    )

    # Low Log Prob Selector for "Hidden Gem" exploration
    low_logprob_selector_enabled: bool = False
    low_logprob_min_threshold: float = -8.0
    low_logprob_max_threshold: float = -3.0

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ImprovementsConfig":
        """Create config from dictionary"""
        # Handle nested configs
        reflection_memory_dict = config_dict.get("reflection_memory", {})
        epuct_selector_dict = config_dict.get("epuct_selector", {})
        lineage_tracker_dict = config_dict.get("lineage_tracker", {})

        return cls(
            enabled=config_dict.get("enabled", True),
            reflection_memory=ReflectionMemoryConfig(**reflection_memory_dict),
            epuct_selector=EPUCTSelectorConfig(**epuct_selector_dict),
            lineage_tracker=LineageTrackerConfig(**lineage_tracker_dict),
            low_logprob_selector_enabled=config_dict.get("low_logprob_selector_enabled", False),
            low_logprob_min_threshold=config_dict.get("low_logprob_min_threshold", -8.0),
            low_logprob_max_threshold=config_dict.get("low_logprob_max_threshold", -3.0),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ImprovementsConfig":
        """Load config from YAML file"""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Extract improvements section if present
        if "improvements" in config_dict:
            config_dict = config_dict["improvements"]

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "enabled": self.enabled,
            "reflection_memory": {
                "enabled": self.reflection_memory.enabled,
                "failure_lessons_enabled": self.reflection_memory.failure_lessons_enabled,
                "success_patterns_enabled": self.reflection_memory.success_patterns_enabled,
                "cross_domain_links_enabled": self.reflection_memory.cross_domain_links_enabled,
                "max_failures": self.reflection_memory.max_failures,
                "max_patterns": self.reflection_memory.max_patterns,
                "retrieval_top_k": self.reflection_memory.retrieval_top_k,
            },
            "epuct_selector": {
                "enabled": self.epuct_selector.enabled,
                "alpha": self.epuct_selector.alpha,
                "beta": self.epuct_selector.beta,
                "gamma": self.epuct_selector.gamma,
                "c_puct": self.epuct_selector.c_puct,
            },
            "lineage_tracker": {
                "enabled": self.lineage_tracker.enabled,
                "soft_backtrack_enabled": self.lineage_tracker.soft_backtrack_enabled,
                "consecutive_failure_threshold": self.lineage_tracker.consecutive_failure_threshold,
            },
            "low_logprob_selector_enabled": self.low_logprob_selector_enabled,
        }


class ImprovementsManager:
    """
    Manager for all improvement modules.

    Provides a unified interface for integrating improvements into the
    OpenEvolve evolution loop without heavily modifying core code.

    Example usage in evolution loop:
        ```python
        manager = ImprovementsManager(config)

        for iteration in range(max_iterations):
            # Get enhanced context for prompt
            context = manager.on_iteration_start(iteration, parent, current_context)

            # Run evolution step...
            child, success = run_evolution_step(parent, context)

            # Record results
            backtrack_target = manager.on_iteration_end(
                iteration, parent, child, success
            )

            if backtrack_target:
                # Soft backtrack to promising ancestor
                parent = get_program(backtrack_target)
        ```
    """

    def __init__(self, config: Optional[ImprovementsConfig] = None):
        self.config = config or ImprovementsConfig()

        # Initialize sub-modules
        self.reflection_memory = None
        self.epuct_selector = None
        self.lineage_tracker = None
        self.low_logprob_selector = None

        if self.config.enabled:
            # Reflection Memory
            if self.config.reflection_memory.enabled:
                self.reflection_memory = BidirectionalReflectionMemory(
                    config=self.config.reflection_memory
                )

            # E-PUCT Selector
            if self.config.epuct_selector.enabled:
                self.epuct_selector = EPUCTSelector(
                    config=self.config.epuct_selector
                )

            # Lineage Tracker
            if self.config.lineage_tracker.enabled:
                self.lineage_tracker = LineageTracker(
                    config=self.config.lineage_tracker
                )

            # Low Log Prob Selector (for Hidden Gem exploration)
            if self.config.low_logprob_selector_enabled:
                self.low_logprob_selector = LowLogProbSelector(
                    min_logprob_threshold=self.config.low_logprob_min_threshold,
                    max_logprob_threshold=self.config.low_logprob_max_threshold,
                )

        logger.info(
            f"Initialized ImprovementsManager (enabled={self.config.enabled}, "
            f"reflection={self.reflection_memory is not None}, "
            f"epuct={self.epuct_selector is not None}, "
            f"lineage={self.lineage_tracker is not None})"
        )

    def on_iteration_start(
        self,
        iteration: int,
        parent_id: str,
        parent_fitness: float,
        parent_code: str,
        feature_coords: Optional[Tuple[int, ...]] = None,
        log_prob: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Called at the start of each iteration.

        Returns context to be used in prompt generation.
        """
        context = {
            "reflections": "",
            "selection_bonus": 0.0,
            "lineage_health": 1.0,
        }

        if not self.config.enabled:
            return context

        # Get failure lessons relevant to current context
        if self.reflection_memory:
            failure_lessons = self.reflection_memory.retrieve_failure_lessons(top_k=3)
            success_patterns = self.reflection_memory.retrieve_success_patterns(top_k=2)

            # Get cross-domain suggestions for "grokking"
            domain_tags = self.reflection_memory._infer_domain_tags(parent_code)
            cross_domain = self.reflection_memory.get_cross_domain_suggestions(
                current_domain_tags=domain_tags,
                top_k=2,
            )

            # Format for prompt injection
            context["reflections"] = self.reflection_memory.format_for_prompt(
                failure_lessons=failure_lessons,
                success_patterns=success_patterns,
                cross_domain_suggestions=cross_domain,
            )

        # Calculate E-PUCT selection score for this parent
        if self.epuct_selector:
            context["selection_bonus"] = self.epuct_selector.compute_selection_score(
                program_id=parent_id,
                fitness=parent_fitness,
                feature_coords=feature_coords,
                log_prob=log_prob,
            )

        # Get lineage health
        if self.lineage_tracker:
            context["lineage_health"] = self.lineage_tracker.get_lineage_health(parent_id)

        return context

    def on_iteration_end(
        self,
        iteration: int,
        parent_id: str,
        parent_fitness: float,
        child_id: Optional[str],
        child_fitness: Optional[float],
        child_code: Optional[str],
        success: bool,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        code_context: Optional[str] = None,
        mutation_type: Optional[str] = None,
        feature_coords: Optional[Tuple[int, ...]] = None,
        metrics_improvement: Optional[Dict[str, float]] = None,
        artifacts: Optional[Dict[str, Any]] = None,  # V2: Add artifacts for config extraction
    ) -> Optional[str]:
        """
        Called at the end of each iteration.

        Records results and returns backtrack target if needed.

        Returns:
            ID of program to backtrack to, or None
        """
        if not self.config.enabled:
            return None

        backtrack_target = None

        # Record failure or success in reflection memory
        if self.reflection_memory:
            if not success and error_type and error_message:
                # Store failure lesson - use child_code as context if available
                self.reflection_memory.store_failure_lesson(
                    error_type=error_type,
                    error_message=error_message,
                    code_context=child_code or code_context or "",  # FIX: Use child_code
                    mutation_type=mutation_type,
                    feature_region=feature_coords,
                )
            elif success and child_fitness and metrics_improvement:
                # Store success pattern if significant
                max_improvement = max(metrics_improvement.values()) if metrics_improvement else 0
                if max_improvement > 0.1:  # Significant improvement
                    # V2: Extract configuration from artifacts for better description
                    config_info = ""
                    if artifacts:
                        config_str = artifacts.get('configuration', '')
                        score = metrics_improvement.get('combined_score', child_fitness)
                        config_info = f" Config: {config_str}. Score: {score:.2f}."

                    self.reflection_memory.store_success_pattern(
                        pattern_type="improvement",
                        description=f"Improved fitness by {max_improvement:.2%}.{config_info}",
                        code_snippet=child_code[:500] if child_code else "",
                        metrics_improvement=metrics_improvement,
                    )

        # Record in E-PUCT selector
        if self.epuct_selector:
            self.epuct_selector.record_offspring_result(
                parent_id=parent_id,
                offspring_fitness=child_fitness or 0.0,
                parent_fitness=parent_fitness,
                success=success,
            )
            if feature_coords:
                self.epuct_selector.update_feature_counts(feature_coords)

        # Record in lineage tracker and check for backtrack
        if self.lineage_tracker:
            if success and child_id and child_fitness:
                # Add child to lineage
                generation = iteration + 1
                self.lineage_tracker.add_program(
                    program_id=child_id,
                    parent_id=parent_id,
                    fitness=child_fitness,
                    generation=generation,
                )

            # Record offspring result and check if we should backtrack
            backtrack_target = self.lineage_tracker.record_offspring_result(
                parent_id=parent_id,
                offspring_id=child_id or "",
                offspring_fitness=child_fitness or 0.0,
                success=success,
            )

        return backtrack_target

    def select_parent(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Select parent(s) using E-PUCT scoring if enabled.

        Falls back to returning candidates as-is if disabled.
        """
        if not self.config.enabled or not self.epuct_selector:
            return candidates[:top_k]

        return self.epuct_selector.select_parent(candidates, top_k)

    def advance_generation(self) -> None:
        """Called at the end of each generation"""
        if self.epuct_selector:
            self.epuct_selector.advance_generation()

    def get_reevaluation_candidates(self, generation: int) -> List[str]:
        """Get programs that should be re-evaluated (Hidden Gem mechanism)"""
        if not self.config.enabled or not self.lineage_tracker:
            return []

        return self.lineage_tracker.should_reevaluate(generation)

    def save(self, output_dir: str) -> None:
        """Save all improvement module states"""
        if not self.config.enabled:
            return

        os.makedirs(output_dir, exist_ok=True)

        if self.reflection_memory:
            self.reflection_memory.save(
                os.path.join(output_dir, "reflection_memory.json")
            )

        # Save stats
        stats = self.get_stats()
        with open(os.path.join(output_dir, "improvements_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved improvements state to {output_dir}")

    def load(self, output_dir: str) -> None:
        """Load all improvement module states"""
        if not self.config.enabled:
            return

        reflection_path = os.path.join(output_dir, "reflection_memory.json")
        if self.reflection_memory and os.path.exists(reflection_path):
            self.reflection_memory.load(reflection_path)

        logger.info(f"Loaded improvements state from {output_dir}")

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all modules"""
        stats = {"enabled": self.config.enabled}

        if self.reflection_memory:
            stats["reflection_memory"] = self.reflection_memory.get_stats()

        if self.epuct_selector:
            stats["epuct_selector"] = self.epuct_selector.get_stats()

        if self.lineage_tracker:
            stats["lineage_tracker"] = self.lineage_tracker.get_stats()

        return stats

    def format_reflections_for_prompt(self) -> str:
        """
        Get formatted reflections to inject into evolution prompt.

        This should be called by the PromptSampler when building prompts.
        """
        if not self.config.enabled or not self.reflection_memory:
            return ""

        # Get relevant reflections
        failure_lessons = self.reflection_memory.retrieve_failure_lessons(top_k=3)
        success_patterns = self.reflection_memory.retrieve_success_patterns(top_k=2)

        return self.reflection_memory.format_for_prompt(
            failure_lessons=failure_lessons,
            success_patterns=success_patterns,
        )


def create_improvements_from_config(config_dict: Dict[str, Any]) -> ImprovementsManager:
    """
    Factory function to create ImprovementsManager from a config dict.

    Can be called from OpenEvolve config loading.
    """
    if "improvements" not in config_dict:
        # Return disabled manager if no improvements config
        return ImprovementsManager(ImprovementsConfig(enabled=False))

    improvements_config = ImprovementsConfig.from_dict(config_dict["improvements"])
    return ImprovementsManager(improvements_config)
