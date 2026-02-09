"""
E-PUCT (Evolution-PUCT) Selector for OpenEvolve

This module implements a PUCT-inspired parent selection mechanism for evolutionary
algorithms. The key insight is that PUCT's essence is credit assignment under
uncertainty, which can be adapted from tree search to evolution.

Formula:
    SelectionScore(p) = α·Fitness(p) + β·Prior(p)·√(TotalGen/(1+Sel(p))) + γ·Novelty(p)

Where:
- Fitness(p): Combined score of program p
- Prior(p): Log probability based prior (exploration guidance)
- Sel(p): Number of times p has been selected as parent
- TotalGen: Total number of generations
- Novelty(p): Diversity bonus (MAP-Elites inspired)
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SelectionStats:
    """Tracks selection statistics for a program"""
    program_id: str
    selection_count: int = 0
    successful_offspring: int = 0  # Children that improved
    failed_offspring: int = 0  # Children that failed
    total_improvement: float = 0.0  # Sum of improvements from children
    log_prob_prior: Optional[float] = None  # Cached log prob prior
    last_selected_generation: int = 0


class EPUCTSelectorConfig:
    """Configuration for E-PUCT selector"""

    def __init__(
        self,
        enabled: bool = True,
        # Weights for the selection formula
        alpha: float = 0.6,  # Weight for fitness
        beta: float = 0.3,   # Weight for exploration bonus
        gamma: float = 0.1,  # Weight for novelty
        # PUCT constant (controls exploration vs exploitation)
        c_puct: float = 1.5,
        # Prior computation
        use_logprob_prior: bool = True,
        default_prior: float = 0.5,
        # Novelty computation
        use_novelty_bonus: bool = True,
        novelty_decay: float = 0.9,  # Decay factor for novelty over time
        # Adaptive parameters
        adaptive_exploration: bool = True,
        min_exploration_bonus: float = 0.1,
        max_exploration_bonus: float = 2.0,
    ):
        self.enabled = enabled
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.c_puct = c_puct
        self.use_logprob_prior = use_logprob_prior
        self.default_prior = default_prior
        self.use_novelty_bonus = use_novelty_bonus
        self.novelty_decay = novelty_decay
        self.adaptive_exploration = adaptive_exploration
        self.min_exploration_bonus = min_exploration_bonus
        self.max_exploration_bonus = max_exploration_bonus


class EPUCTSelector:
    """
    E-PUCT based parent selector for evolutionary algorithms.

    This adapts the PUCT formula from MCTS to evolution:
    - Q(s,a) → Fitness(p): The quality of a program
    - P(s,a) → Prior(p): Log probability based prior from LLM
    - N(s,a) → Sel(p): Selection count of a program
    - N(s) → TotalGen: Total number of generations

    Key insight: PUCT balances exploitation (selecting good parents) with
    exploration (giving underexplored parents a chance), which is exactly
    what we need in evolutionary algorithms.
    """

    def __init__(self, config: Optional[EPUCTSelectorConfig] = None):
        self.config = config or EPUCTSelectorConfig()

        # Selection statistics per program
        self.stats: Dict[str, SelectionStats] = {}

        # Global statistics
        self.total_generations: int = 0
        self.total_selections: int = 0

        # Fitness normalization
        self._min_fitness: float = float('inf')
        self._max_fitness: float = float('-inf')

        # Novelty tracking
        self._feature_counts: Dict[Tuple[int, ...], int] = {}

        logger.info(f"Initialized EPUCTSelector (enabled={self.config.enabled})")

    def _get_or_create_stats(self, program_id: str) -> SelectionStats:
        """Get or create selection stats for a program"""
        if program_id not in self.stats:
            self.stats[program_id] = SelectionStats(program_id=program_id)
        return self.stats[program_id]

    def compute_selection_score(
        self,
        program_id: str,
        fitness: float,
        feature_coords: Optional[Tuple[int, ...]] = None,
        log_prob: Optional[float] = None,
    ) -> float:
        """
        Compute the E-PUCT selection score for a program.

        Args:
            program_id: Unique identifier for the program
            fitness: The fitness/combined_score of the program
            feature_coords: MAP-Elites feature coordinates
            log_prob: Log probability from LLM (if available)

        Returns:
            Selection score (higher = more likely to be selected)
        """
        if not self.config.enabled:
            return fitness

        stats = self._get_or_create_stats(program_id)

        # Update fitness bounds for normalization
        self._min_fitness = min(self._min_fitness, fitness)
        self._max_fitness = max(self._max_fitness, fitness)

        # Normalize fitness to [0, 1]
        fitness_range = self._max_fitness - self._min_fitness
        if fitness_range > 0:
            normalized_fitness = (fitness - self._min_fitness) / fitness_range
        else:
            normalized_fitness = 0.5

        # Compute prior
        if self.config.use_logprob_prior and log_prob is not None:
            # Convert log prob to [0, 1] range
            # Higher log prob = more "natural" mutation = higher prior
            prior = self._logprob_to_prior(log_prob)
            stats.log_prob_prior = prior
            logger.debug(f"E-PUCT using log_prob={log_prob:.4f} -> prior={prior:.4f}")
        elif stats.log_prob_prior is not None:
            prior = stats.log_prob_prior
        else:
            prior = self.config.default_prior
            logger.debug(f"E-PUCT using default prior={prior} (no log_prob available)")

        # Compute exploration bonus (PUCT-style)
        selection_count = stats.selection_count
        if self.total_generations > 0:
            exploration_bonus = (
                self.config.c_puct
                * prior
                * math.sqrt(self.total_generations)
                / (1 + selection_count)
            )
        else:
            exploration_bonus = self.config.c_puct * prior

        # Clamp exploration bonus
        exploration_bonus = max(
            self.config.min_exploration_bonus,
            min(self.config.max_exploration_bonus, exploration_bonus)
        )

        # Compute novelty bonus
        novelty_bonus = 0.0
        if self.config.use_novelty_bonus and feature_coords is not None:
            novelty_bonus = self._compute_novelty(feature_coords)

        # Adaptive exploration: increase exploration if progress is stagnating
        if self.config.adaptive_exploration:
            exploration_bonus *= self._get_exploration_multiplier()

        # Combine scores using configured weights
        score = (
            self.config.alpha * normalized_fitness
            + self.config.beta * exploration_bonus
            + self.config.gamma * novelty_bonus
        )

        return score

    def _logprob_to_prior(self, log_prob: float) -> float:
        """
        Convert log probability to prior in [0, 1].

        Higher log prob (closer to 0) = more confident = higher prior
        Lower log prob (more negative) = less confident = lower prior

        We use sigmoid transformation to map to [0, 1].
        """
        # Typical log probs are in range [-10, 0]
        # Shift and scale to make sigmoid work well
        # sigmoid((log_prob + 5) / 2) maps roughly:
        #   log_prob = 0  -> ~0.92 (high confidence)
        #   log_prob = -5 -> ~0.5  (medium confidence)
        #   log_prob = -10 -> ~0.08 (low confidence)
        x = (log_prob + 5) / 2
        return 1 / (1 + math.exp(-x))

    def _compute_novelty(self, feature_coords: Tuple[int, ...]) -> float:
        """
        Compute novelty bonus based on feature coordinates.

        Programs in less-explored regions of the feature space get higher novelty.
        """
        if feature_coords not in self._feature_counts:
            self._feature_counts[feature_coords] = 0

        count = self._feature_counts[feature_coords]

        # Novelty decreases as region gets more explored
        # Use inverse count with decay
        if count == 0:
            return 1.0  # Maximum novelty for unexplored regions
        else:
            return self.config.novelty_decay ** count

    def _get_exploration_multiplier(self) -> float:
        """
        Get exploration multiplier based on recent progress.

        If we're making good progress, reduce exploration.
        If progress has stagnated, increase exploration.
        """
        if self.total_selections < 10:
            return 1.0

        # Count recent successful vs failed offspring
        recent_success_rate = sum(
            s.successful_offspring / max(1, s.selection_count)
            for s in self.stats.values()
            if s.selection_count > 0
        ) / max(1, len([s for s in self.stats.values() if s.selection_count > 0]))

        # Low success rate = increase exploration
        # High success rate = reduce exploration
        if recent_success_rate < 0.3:
            return 1.5  # Stagnating, explore more
        elif recent_success_rate > 0.7:
            return 0.7  # Good progress, exploit more
        else:
            return 1.0  # Normal

    def select_parent(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Select parent(s) from candidates using E-PUCT scoring.

        Args:
            candidates: List of candidate programs with 'id', 'metrics', 'feature_coords'
            top_k: Number of parents to select

        Returns:
            List of selected parents
        """
        if not self.config.enabled or not candidates:
            # Fall back to random selection
            return random.sample(candidates, min(top_k, len(candidates)))

        # Compute scores for all candidates
        scored_candidates = []
        for candidate in candidates:
            program_id = candidate.get('id', str(id(candidate)))
            fitness = candidate.get('metrics', {}).get('combined_score', 0.0)
            feature_coords = candidate.get('feature_coords')
            log_prob = candidate.get('log_prob')

            score = self.compute_selection_score(
                program_id=program_id,
                fitness=fitness,
                feature_coords=feature_coords,
                log_prob=log_prob,
            )

            scored_candidates.append((candidate, score))

        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Select top_k
        selected = [c[0] for c in scored_candidates[:top_k]]

        # Update selection counts
        for candidate in selected:
            program_id = candidate.get('id', str(id(candidate)))
            stats = self._get_or_create_stats(program_id)
            stats.selection_count += 1
            stats.last_selected_generation = self.total_generations

        self.total_selections += len(selected)

        return selected

    def record_offspring_result(
        self,
        parent_id: str,
        offspring_fitness: float,
        parent_fitness: float,
        success: bool,
    ) -> None:
        """
        Record the result of an offspring from a parent.

        Args:
            parent_id: ID of the parent program
            offspring_fitness: Fitness of the offspring
            parent_fitness: Fitness of the parent
            success: Whether the offspring was successful (valid code)
        """
        stats = self._get_or_create_stats(parent_id)

        if success:
            stats.successful_offspring += 1
            improvement = offspring_fitness - parent_fitness
            stats.total_improvement += max(0, improvement)
        else:
            stats.failed_offspring += 1

    def update_feature_counts(self, feature_coords: Tuple[int, ...]) -> None:
        """Update the count for a feature region"""
        if feature_coords not in self._feature_counts:
            self._feature_counts[feature_coords] = 0
        self._feature_counts[feature_coords] += 1

    def advance_generation(self) -> None:
        """Called at the end of each generation"""
        self.total_generations += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the selector"""
        active_programs = [s for s in self.stats.values() if s.selection_count > 0]

        return {
            "total_generations": self.total_generations,
            "total_selections": self.total_selections,
            "tracked_programs": len(self.stats),
            "active_programs": len(active_programs),
            "explored_regions": len(self._feature_counts),
            "avg_selections_per_program": (
                sum(s.selection_count for s in active_programs) / len(active_programs)
                if active_programs else 0
            ),
            "avg_success_rate": (
                sum(s.successful_offspring / max(1, s.selection_count) for s in active_programs) / len(active_programs)
                if active_programs else 0
            ),
        }


class LowLogProbSelector:
    """
    Special selector that prioritizes LOW log probability mutations.

    This is for the "Hidden Gem" island - exploring unconventional mutations
    that the LLM is less confident about, but might lead to breakthroughs.

    Key insight: High log prob = "comfortable zone" for LLM.
    Low log prob = unconventional, potentially innovative.
    """

    def __init__(
        self,
        min_logprob_threshold: float = -8.0,
        max_logprob_threshold: float = -3.0,
    ):
        self.min_logprob_threshold = min_logprob_threshold
        self.max_logprob_threshold = max_logprob_threshold

        # Track surprise successes
        self.surprise_successes: List[Dict[str, Any]] = []

        logger.info("Initialized LowLogProbSelector for Hidden Gem exploration")

    def select_mutation(
        self,
        mutations: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Select the mutation with the LOWEST log probability.

        Args:
            mutations: List of mutations with 'code', 'log_prob'

        Returns:
            Selected mutation or None
        """
        if not mutations:
            return None

        # Filter mutations with log probs in our target range
        valid_mutations = [
            m for m in mutations
            if m.get('log_prob') is not None
            and self.min_logprob_threshold <= m['log_prob'] <= self.max_logprob_threshold
        ]

        if not valid_mutations:
            # Fall back to any mutation with log prob
            valid_mutations = [m for m in mutations if m.get('log_prob') is not None]

        if not valid_mutations:
            # No log probs available, return random
            return random.choice(mutations) if mutations else None

        # Select the one with LOWEST log prob (most unconventional)
        return min(valid_mutations, key=lambda m: m['log_prob'])

    def record_surprise_success(
        self,
        mutation: Dict[str, Any],
        fitness_improvement: float,
    ) -> None:
        """
        Record when a low log prob mutation succeeds unexpectedly.

        These are potential breakthroughs worth analyzing.
        """
        if mutation.get('log_prob') is not None and fitness_improvement > 0:
            self.surprise_successes.append({
                'log_prob': mutation['log_prob'],
                'fitness_improvement': fitness_improvement,
                'code_snippet': mutation.get('code', '')[:500],
            })

            # Log this as it's potentially interesting
            logger.info(
                f"Surprise success! Low log prob mutation (log_prob={mutation['log_prob']:.2f}) "
                f"achieved {fitness_improvement:.4f} improvement"
            )

    def get_surprise_successes(self) -> List[Dict[str, Any]]:
        """Get all recorded surprise successes"""
        return self.surprise_successes
