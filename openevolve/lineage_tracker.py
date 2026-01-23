"""
Lineage Tracker for OpenEvolve

This module tracks the ancestry of programs, enabling:
- Soft backtracking: Return to promising ancestors when current direction stagnates
- Lineage health monitoring: Identify which lineages are productive
- Hidden Gem discovery: Re-evaluate previously deprioritized programs

Key insight from LATS: Tree structure enables backtracking to promising nodes.
In evolution, we can achieve similar benefits by tracking program lineages.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LineageNode:
    """Represents a node in the program lineage tree"""
    program_id: str
    parent_id: Optional[str] = None
    generation: int = 0
    fitness: float = 0.0
    timestamp: float = field(default_factory=time.time)

    # Lineage statistics
    children_count: int = 0
    successful_children: int = 0
    failed_children: int = 0
    best_descendant_fitness: float = 0.0

    # Health metrics
    consecutive_failures: int = 0
    improvement_rate: float = 0.0  # Average improvement per generation
    is_deprioritized: bool = False  # Soft-pruned but not deleted


@dataclass
class LineageStats:
    """Statistics for a complete lineage (from root to leaves)"""
    root_id: str
    total_nodes: int = 0
    max_depth: int = 0
    total_improvement: float = 0.0
    success_rate: float = 0.0
    is_productive: bool = True


class LineageTrackerConfig:
    """Configuration for lineage tracking"""

    def __init__(
        self,
        enabled: bool = True,
        soft_backtrack_enabled: bool = True,
        # Thresholds for soft backtracking
        consecutive_failure_threshold: int = 3,
        stagnation_generations: int = 5,
        deprioritize_threshold: float = 0.05,  # Min improvement rate
        # Re-evaluation settings
        reevaluate_interval: int = 20,  # Generations between re-evaluations
        max_reevaluations: int = 3,  # Max times to re-evaluate a deprioritized node
        # Lineage health
        min_success_rate: float = 0.2,  # Below this, lineage is unhealthy
        productive_lineage_bonus: float = 0.1,  # Bonus for selecting from productive lineages
    ):
        self.enabled = enabled
        self.soft_backtrack_enabled = soft_backtrack_enabled
        self.consecutive_failure_threshold = consecutive_failure_threshold
        self.stagnation_generations = stagnation_generations
        self.deprioritize_threshold = deprioritize_threshold
        self.reevaluate_interval = reevaluate_interval
        self.max_reevaluations = max_reevaluations
        self.min_success_rate = min_success_rate
        self.productive_lineage_bonus = productive_lineage_bonus


class LineageTracker:
    """
    Tracks program ancestry and enables intelligent backtracking.

    Unlike hard tree pruning in MCTS, evolution uses "soft" approaches:
    - Deprioritize instead of delete
    - Periodic re-evaluation of promising ancestors
    - Lineage-aware parent selection

    This helps solve the "Hidden Gem" problem: promising directions
    that were abandoned too early.
    """

    def __init__(self, config: Optional[LineageTrackerConfig] = None):
        self.config = config or LineageTrackerConfig()

        # Core data structures
        self.nodes: Dict[str, LineageNode] = {}
        self.children_map: Dict[str, List[str]] = defaultdict(list)
        self.roots: Set[str] = set()

        # Deprioritized nodes (soft-pruned)
        self.deprioritized: Set[str] = set()
        self.reevaluation_counts: Dict[str, int] = defaultdict(int)

        # Statistics
        self.current_generation: int = 0
        self.total_backtracks: int = 0
        self.successful_backtracks: int = 0

        logger.info(f"Initialized LineageTracker (enabled={self.config.enabled})")

    def add_program(
        self,
        program_id: str,
        parent_id: Optional[str],
        fitness: float,
        generation: int,
    ) -> None:
        """
        Add a new program to the lineage tree.

        Args:
            program_id: Unique ID of the new program
            parent_id: ID of the parent program (None for initial programs)
            fitness: Fitness score of the new program
            generation: Generation number
        """
        if not self.config.enabled:
            return

        node = LineageNode(
            program_id=program_id,
            parent_id=parent_id,
            generation=generation,
            fitness=fitness,
            best_descendant_fitness=fitness,
        )

        self.nodes[program_id] = node

        if parent_id is None:
            self.roots.add(program_id)
        else:
            self.children_map[parent_id].append(program_id)
            # Update parent statistics
            if parent_id in self.nodes:
                parent_node = self.nodes[parent_id]
                parent_node.children_count += 1

        self.current_generation = max(self.current_generation, generation)

    def record_offspring_result(
        self,
        parent_id: str,
        offspring_id: str,
        offspring_fitness: float,
        success: bool,
    ) -> Optional[str]:
        """
        Record the result of creating an offspring.

        Args:
            parent_id: ID of the parent program
            offspring_id: ID of the offspring
            offspring_fitness: Fitness of the offspring
            success: Whether the offspring was valid

        Returns:
            ID of program to backtrack to, or None
        """
        if not self.config.enabled or parent_id not in self.nodes:
            return None

        parent_node = self.nodes[parent_id]

        if success:
            parent_node.successful_children += 1
            parent_node.consecutive_failures = 0

            # Update best descendant fitness up the tree
            self._propagate_best_fitness(parent_id, offspring_fitness)

            # Calculate improvement rate
            improvement = offspring_fitness - parent_node.fitness
            parent_node.improvement_rate = (
                0.9 * parent_node.improvement_rate + 0.1 * improvement
            )

        else:
            parent_node.failed_children += 1
            parent_node.consecutive_failures += 1

        # Check if we should trigger soft backtrack
        if self.config.soft_backtrack_enabled:
            backtrack_target = self._check_soft_backtrack(parent_id)
            if backtrack_target:
                return backtrack_target

        return None

    def _propagate_best_fitness(self, node_id: str, fitness: float) -> None:
        """Propagate best descendant fitness up the tree"""
        current_id = node_id
        while current_id and current_id in self.nodes:
            node = self.nodes[current_id]
            if fitness > node.best_descendant_fitness:
                node.best_descendant_fitness = fitness
                current_id = node.parent_id
            else:
                break

    def _check_soft_backtrack(self, current_id: str) -> Optional[str]:
        """
        Check if we should backtrack to a previous program.

        Returns:
            ID of program to backtrack to, or None
        """
        if current_id not in self.nodes:
            return None

        current_node = self.nodes[current_id]

        # Check consecutive failures
        if current_node.consecutive_failures >= self.config.consecutive_failure_threshold:
            # Find best ancestor to backtrack to
            backtrack_target = self._find_backtrack_target(current_id)
            if backtrack_target:
                logger.info(
                    f"Soft backtrack triggered: {current_id} -> {backtrack_target} "
                    f"(consecutive failures: {current_node.consecutive_failures})"
                )
                self.total_backtracks += 1
                return backtrack_target

        # Check improvement rate stagnation
        if current_node.improvement_rate < self.config.deprioritize_threshold:
            # Don't backtrack immediately, just deprioritize
            self._deprioritize(current_id)

        return None

    def _find_backtrack_target(self, from_id: str) -> Optional[str]:
        """
        Find the best ancestor to backtrack to.

        Strategy: Find the nearest ancestor that:
        1. Has good success rate
        2. Has unexplored children potential
        3. Is not deprioritized
        """
        if from_id not in self.nodes:
            return None

        current_id = self.nodes[from_id].parent_id
        best_target = None
        best_score = -float('inf')

        while current_id and current_id in self.nodes:
            node = self.nodes[current_id]

            # Skip deprioritized nodes
            if current_id in self.deprioritized:
                current_id = node.parent_id
                continue

            # Calculate backtrack score
            success_rate = (
                node.successful_children / max(1, node.children_count)
            )
            unexplored_potential = 1.0 / (1.0 + node.children_count)

            score = (
                0.5 * success_rate
                + 0.3 * unexplored_potential
                + 0.2 * (node.best_descendant_fitness / max(0.01, node.fitness))
            )

            if score > best_score and success_rate >= self.config.min_success_rate:
                best_score = score
                best_target = current_id

            current_id = node.parent_id

        return best_target

    def _deprioritize(self, node_id: str) -> None:
        """Soft-prune a node (don't delete, just deprioritize)"""
        if node_id not in self.deprioritized:
            self.deprioritized.add(node_id)
            if node_id in self.nodes:
                self.nodes[node_id].is_deprioritized = True
            logger.debug(f"Deprioritized node: {node_id}")

    def should_reevaluate(self, generation: int) -> List[str]:
        """
        Check which deprioritized nodes should be re-evaluated.

        This is the "Hidden Gem" mechanism: periodically reconsider
        abandoned directions.

        Returns:
            List of program IDs to re-evaluate
        """
        if not self.config.enabled or generation % self.config.reevaluate_interval != 0:
            return []

        candidates = []
        for node_id in self.deprioritized:
            if node_id not in self.nodes:
                continue

            # Check re-evaluation count
            if self.reevaluation_counts[node_id] >= self.config.max_reevaluations:
                continue

            node = self.nodes[node_id]

            # Candidates: nodes with good historical fitness but were deprioritized
            if node.fitness > 0 and node.successful_children > 0:
                candidates.append((node_id, node.best_descendant_fitness))

        # Sort by best descendant fitness
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top candidates
        result = [c[0] for c in candidates[:3]]

        for node_id in result:
            self.reevaluation_counts[node_id] += 1

        if result:
            logger.info(f"Re-evaluating {len(result)} deprioritized programs")

        return result

    def restore_from_deprioritized(self, node_id: str) -> None:
        """
        Restore a deprioritized node if it proves valuable.

        Called when a re-evaluation shows the node is actually good.
        """
        if node_id in self.deprioritized:
            self.deprioritized.remove(node_id)
            if node_id in self.nodes:
                self.nodes[node_id].is_deprioritized = False
            self.successful_backtracks += 1
            logger.info(f"Restored node from deprioritized: {node_id}")

    def get_lineage_health(self, node_id: str) -> float:
        """
        Calculate the health score of a lineage.

        Healthy lineages are more likely to produce good offspring.

        Returns:
            Health score in [0, 1]
        """
        if not self.config.enabled or node_id not in self.nodes:
            return 0.5

        node = self.nodes[node_id]

        # Factors for health
        success_rate = (
            node.successful_children / max(1, node.children_count)
            if node.children_count > 0 else 0.5
        )

        improvement_factor = min(1.0, max(0.0, node.improvement_rate + 0.5))

        depth_penalty = 1.0 / (1.0 + node.generation * 0.1)

        health = (
            0.5 * success_rate
            + 0.3 * improvement_factor
            + 0.2 * depth_penalty
        )

        # Penalty for deprioritized
        if node.is_deprioritized:
            health *= 0.5

        return health

    def get_productive_lineages(self, top_k: int = 5) -> List[str]:
        """
        Get the most productive lineages.

        Returns:
            List of root program IDs with best lineages
        """
        if not self.config.enabled:
            return []

        lineage_scores = []
        for root_id in self.roots:
            stats = self._compute_lineage_stats(root_id)
            if stats.is_productive:
                lineage_scores.append((root_id, stats.total_improvement))

        lineage_scores.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in lineage_scores[:top_k]]

    def _compute_lineage_stats(self, root_id: str) -> LineageStats:
        """Compute statistics for a lineage"""
        stats = LineageStats(root_id=root_id)

        visited = set()
        stack = [(root_id, 0)]

        while stack:
            node_id, depth = stack.pop()
            if node_id in visited or node_id not in self.nodes:
                continue

            visited.add(node_id)
            node = self.nodes[node_id]

            stats.total_nodes += 1
            stats.max_depth = max(stats.max_depth, depth)

            # Track improvement
            if node.parent_id and node.parent_id in self.nodes:
                parent_fitness = self.nodes[node.parent_id].fitness
                stats.total_improvement += max(0, node.fitness - parent_fitness)

            # Add children to stack
            for child_id in self.children_map[node_id]:
                stack.append((child_id, depth + 1))

        # Calculate success rate
        total_children = sum(
            self.nodes[nid].children_count for nid in visited
            if nid in self.nodes
        )
        successful_children = sum(
            self.nodes[nid].successful_children for nid in visited
            if nid in self.nodes
        )

        stats.success_rate = (
            successful_children / max(1, total_children)
        )

        stats.is_productive = stats.success_rate >= self.config.min_success_rate

        return stats

    def get_ancestry(self, node_id: str) -> List[str]:
        """Get the complete ancestry of a node (from root to node)"""
        ancestry = []
        current_id = node_id

        while current_id and current_id in self.nodes:
            ancestry.append(current_id)
            current_id = self.nodes[current_id].parent_id

        return list(reversed(ancestry))

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the lineage tracker"""
        return {
            "total_nodes": len(self.nodes),
            "total_roots": len(self.roots),
            "deprioritized_count": len(self.deprioritized),
            "current_generation": self.current_generation,
            "total_backtracks": self.total_backtracks,
            "successful_backtracks": self.successful_backtracks,
            "backtrack_success_rate": (
                self.successful_backtracks / max(1, self.total_backtracks)
            ),
            "average_lineage_depth": (
                sum(n.generation for n in self.nodes.values()) / max(1, len(self.nodes))
            ),
        }
