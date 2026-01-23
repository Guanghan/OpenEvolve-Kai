"""
Bidirectional Reflection Memory for OpenEvolve

This module implements a memory system that learns from both failures and successes,
enabling the evolution process to avoid repeated mistakes and transfer successful
patterns across different contexts.

Key features:
- Failure lessons: Learn from errors to avoid repeating them
- Success patterns: Extract and share successful strategies
- Cross-domain links: Enable "grokking" through unexpected connections
- Embedding-based retrieval: Find relevant reflections efficiently
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FailureLesson:
    """Represents a learned lesson from a failure"""
    id: str
    error_type: str  # e.g., "NameError", "SyntaxError", "RuntimeError"
    error_message: str
    context: str  # Code context where error occurred
    lesson: str  # What to avoid / how to fix
    mutation_type: Optional[str] = None  # Type of mutation that caused failure
    feature_region: Optional[Tuple[int, ...]] = None  # MAP-Elites region
    timestamp: float = field(default_factory=time.time)
    retrieval_count: int = 0
    success_after_application: int = 0  # Times this lesson helped avoid failure


@dataclass
class SuccessPattern:
    """Represents a successful pattern that can be transferred"""
    id: str
    pattern_type: str  # e.g., "algorithm", "optimization", "structure"
    description: str  # What makes this successful
    code_snippet: str  # Example code
    metrics_improvement: Dict[str, float]  # How much it improved
    domain_tags: List[str]  # Tags for cross-domain matching
    timestamp: float = field(default_factory=time.time)
    retrieval_count: int = 0
    cross_domain_successes: int = 0  # Times it worked in different domains


@dataclass
class CrossDomainLink:
    """Links between patterns in different domains"""
    source_pattern_id: str
    target_pattern_id: str
    similarity_score: float  # 0.3-0.8 range for "weak" but useful similarities
    transfer_successes: int = 0
    timestamp: float = field(default_factory=time.time)


class ReflectionMemoryConfig:
    """Configuration for the reflection memory system"""

    def __init__(
        self,
        enabled: bool = True,
        failure_lessons_enabled: bool = True,
        success_patterns_enabled: bool = True,
        cross_domain_links_enabled: bool = True,
        max_failures: int = 200,
        max_patterns: int = 100,
        max_links: int = 500,
        significant_improvement_threshold: float = 0.1,
        retrieval_top_k: int = 3,
        min_similarity_for_link: float = 0.3,
        max_similarity_for_link: float = 0.8,
        decay_factor: float = 0.95,  # For pruning old entries
    ):
        self.enabled = enabled
        self.failure_lessons_enabled = failure_lessons_enabled
        self.success_patterns_enabled = success_patterns_enabled
        self.cross_domain_links_enabled = cross_domain_links_enabled
        self.max_failures = max_failures
        self.max_patterns = max_patterns
        self.max_links = max_links
        self.significant_improvement_threshold = significant_improvement_threshold
        self.retrieval_top_k = retrieval_top_k
        self.min_similarity_for_link = min_similarity_for_link
        self.max_similarity_for_link = max_similarity_for_link
        self.decay_factor = decay_factor


class BidirectionalReflectionMemory:
    """
    A memory system that learns from both failures and successes.

    This implements the core insight from LATS: reflection provides "semantic gradients"
    that are richer than scalar rewards alone.

    Key innovations:
    1. Bidirectional learning: Not just "what to avoid" but also "what to do"
    2. Cross-domain transfer: Success patterns can "grok" across different regions
    3. Contextual retrieval: Find relevant reflections based on current context
    """

    def __init__(
        self,
        config: Optional[ReflectionMemoryConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        self.config = config or ReflectionMemoryConfig()
        self.llm_client = llm_client  # For generating reflections

        # Storage
        self.failure_lessons: Dict[str, FailureLesson] = {}
        self.success_patterns: Dict[str, SuccessPattern] = {}
        self.cross_domain_links: List[CrossDomainLink] = []

        # Indexes for fast retrieval
        self._error_type_index: Dict[str, List[str]] = defaultdict(list)
        self._pattern_type_index: Dict[str, List[str]] = defaultdict(list)
        self._domain_tag_index: Dict[str, List[str]] = defaultdict(list)

        # Statistics
        self.stats = {
            "failures_stored": 0,
            "patterns_stored": 0,
            "retrievals": 0,
            "successful_applications": 0,
            "cross_domain_transfers": 0,
        }

        logger.info(f"Initialized BidirectionalReflectionMemory (enabled={self.config.enabled})")

    def _generate_id(self, content: str) -> str:
        """Generate a unique ID based on content hash"""
        return hashlib.md5(content.encode()).hexdigest()[:12]

    # ==================== Failure Lessons ====================

    def store_failure_lesson(
        self,
        error_type: str,
        error_message: str,
        code_context: str,
        mutation_type: Optional[str] = None,
        feature_region: Optional[Tuple[int, ...]] = None,
    ) -> Optional[str]:
        """
        Store a lesson learned from a failure.

        Args:
            error_type: Type of error (e.g., "NameError")
            error_message: The error message
            code_context: Code that caused the error
            mutation_type: Type of mutation that was attempted
            feature_region: MAP-Elites cell coordinates

        Returns:
            Lesson ID if stored, None if disabled or duplicate
        """
        if not self.config.enabled or not self.config.failure_lessons_enabled:
            return None

        # Generate lesson from error
        lesson = self._generate_failure_lesson(error_type, error_message, code_context)

        # Create unique ID
        content_hash = f"{error_type}:{error_message[:100]}:{code_context[:200]}"
        lesson_id = self._generate_id(content_hash)

        # Check for duplicate
        if lesson_id in self.failure_lessons:
            # Update retrieval count for existing lesson
            self.failure_lessons[lesson_id].retrieval_count += 1
            return lesson_id

        # Create and store lesson
        failure_lesson = FailureLesson(
            id=lesson_id,
            error_type=error_type,
            error_message=error_message,
            context=code_context[:500],  # Truncate for storage
            lesson=lesson,
            mutation_type=mutation_type,
            feature_region=feature_region,
        )

        self.failure_lessons[lesson_id] = failure_lesson
        self._error_type_index[error_type].append(lesson_id)
        self.stats["failures_stored"] += 1

        # Prune if over limit
        if len(self.failure_lessons) > self.config.max_failures:
            self._prune_failures()

        logger.debug(f"Stored failure lesson: {lesson_id} ({error_type})")
        return lesson_id

    def _generate_failure_lesson(
        self,
        error_type: str,
        error_message: str,
        code_context: str,
    ) -> str:
        """
        Generate a lesson from an error - V1: Smart context-aware generation.

        V1 Improvements:
        1. Analyze actual code to find the problematic pattern
        2. Extract specific variable/function names from error
        3. Identify missing imports or definitions
        4. Provide actionable fix suggestions
        """
        import re

        lesson_parts = []

        # === Part 1: Specific error analysis ===
        if "not defined" in error_message.lower() or error_type in ["NameError", "undefined_variable"]:
            var_name = self._extract_name(error_message)

            # Check if it's a common missing import
            common_imports = {
                'random': 'import random',
                'sys': 'import sys',
                'math': 'import math',
                'numpy': 'import numpy as np',
                'np': 'import numpy as np',
                'os': 'import os',
                'json': 'import json',
                're': 'import re',
            }

            if var_name.lower() in common_imports:
                lesson_parts.append(
                    f"Missing import: Add `{common_imports[var_name.lower()]}` at the top of the code."
                )
            elif var_name.isupper():
                # Likely a constant
                lesson_parts.append(
                    f"Undefined constant `{var_name}`: Define it before use, e.g., `{var_name} = ...`"
                )
            else:
                lesson_parts.append(
                    f"Variable `{var_name}` used before definition. Initialize it before the line where it's used."
                )

        elif error_type in ["TypeError", "type"] or "TypeError" in error_message or "not supported between" in error_message:
            # Extract operation from error message (TypeError patterns)
            if "not supported between" in error_message:
                # Comparison error
                types_match = re.search(r"'(\w+)' and '(\w+)'", error_message)
                if types_match:
                    type1, type2 = types_match.groups()
                    lesson_parts.append(
                        f"Cannot compare {type1} with {type2}. Ensure both operands are the same type before comparison. "
                        f"Check if you're comparing objects when you should compare their values (e.g., use dict.get('key') instead of dict)."
                    )
            elif "argument" in error_message.lower():
                lesson_parts.append(
                    "Function received wrong argument type. Check the function signature and ensure you're passing the correct types."
                )
            elif "'NoneType'" in error_message:
                lesson_parts.append(
                    "Operation on None value. Add a check like `if x is not None:` before using the variable."
                )
            else:
                lesson_parts.append(
                    f"Type error: {error_message[:100]}. Verify operand types match the expected operation."
                )

        elif error_type in ["KeyError", "evaluation"] and "KeyError" in error_message:
            key_name = self._extract_name(error_message)
            lesson_parts.append(
                f"Dictionary key `{key_name}` not found. Use `dict.get('{key_name}', default)` for safe access, "
                f"or verify the key exists with `if '{key_name}' in dict:`."
            )

        elif error_type in ["SyntaxError", "syntax"]:
            lesson_parts.append(
                "Syntax error in generated code. Common causes: missing colons after if/for/def, "
                "unmatched parentheses/brackets, or invalid Python syntax. Double-check code structure."
            )

        elif error_type in ["IndentationError", "indentation"]:
            lesson_parts.append(
                "Indentation error. Use consistent 4-space indentation. Ensure all code blocks "
                "(after if/for/def/class) are properly indented."
            )

        elif error_type == "validation":
            lesson_parts.append(
                f"Validation failed: {error_message[:150]}. Ensure the output format matches the expected schema."
            )

        else:
            # Fallback with more context
            lesson_parts.append(
                f"Error ({error_type}): {error_message[:100]}."
            )

        # === Part 2: Code-specific analysis (if code context available) ===
        if code_context and len(code_context) > 50:
            # Check for common issues in the code
            code_issues = []

            # Check for missing imports at top
            if 'random.' in code_context and 'import random' not in code_context:
                code_issues.append("Add `import random` at the top")
            if 'sys.' in code_context and 'import sys' not in code_context:
                code_issues.append("Add `import sys` at the top")
            if 'np.' in code_context and 'import numpy' not in code_context:
                code_issues.append("Add `import numpy as np` at the top")

            # Check for common K-Module specific issues
            if 'config[' in code_context:
                # Accessing config dict - common source of KeyError
                config_keys = re.findall(r"config\['(\w+)'\]", code_context)
                expected_keys = ['loader', 'preprocess', 'algorithm', 'formatter']
                for key in config_keys:
                    if key not in expected_keys:
                        code_issues.append(f"Unexpected config key `{key}`. Valid keys: {expected_keys}")

            if code_issues:
                lesson_parts.append("Code issues found: " + "; ".join(code_issues))

        # === Part 3: Actionable summary ===
        if lesson_parts:
            return " | ".join(lesson_parts)
        else:
            return f"Error occurred during {error_type}. Review the generated code carefully."

    def _extract_name(self, error_message: str) -> str:
        """Extract variable/function name from error message"""
        import re
        match = re.search(r"'([^']+)'", error_message)
        return match.group(1) if match else "unknown"

    def retrieve_failure_lessons(
        self,
        error_type: Optional[str] = None,
        code_context: Optional[str] = None,
        mutation_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[FailureLesson]:
        """
        Retrieve relevant failure lessons.

        Args:
            error_type: Filter by error type
            code_context: Current code context for similarity matching
            mutation_type: Filter by mutation type
            top_k: Number of lessons to retrieve

        Returns:
            List of relevant failure lessons
        """
        if not self.config.enabled or not self.config.failure_lessons_enabled:
            return []

        top_k = top_k or self.config.retrieval_top_k
        candidates = []

        if error_type and error_type in self._error_type_index:
            # Filter by error type
            lesson_ids = self._error_type_index[error_type]
            candidates = [self.failure_lessons[lid] for lid in lesson_ids if lid in self.failure_lessons]
        else:
            # Return all lessons
            candidates = list(self.failure_lessons.values())

        # Filter by mutation type if specified
        if mutation_type:
            candidates = [c for c in candidates if c.mutation_type == mutation_type]

        # Sort by relevance (retrieval count + recency)
        candidates.sort(
            key=lambda x: (x.success_after_application, -x.retrieval_count, -x.timestamp),
            reverse=True
        )

        # Update retrieval counts
        for lesson in candidates[:top_k]:
            lesson.retrieval_count += 1

        self.stats["retrievals"] += 1
        return candidates[:top_k]

    # ==================== Success Patterns ====================

    def store_success_pattern(
        self,
        pattern_type: str,
        description: str,
        code_snippet: str,
        metrics_improvement: Dict[str, float],
        domain_tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Store a successful pattern for potential transfer.

        Args:
            pattern_type: Type of pattern (e.g., "optimization", "algorithm")
            description: What makes this pattern successful
            code_snippet: Example code
            metrics_improvement: How much metrics improved
            domain_tags: Tags for cross-domain matching

        Returns:
            Pattern ID if stored, None if disabled
        """
        if not self.config.enabled or not self.config.success_patterns_enabled:
            return None

        # Check if improvement is significant
        max_improvement = max(metrics_improvement.values()) if metrics_improvement else 0
        if max_improvement < self.config.significant_improvement_threshold:
            return None

        # Create unique ID
        content_hash = f"{pattern_type}:{code_snippet[:200]}"
        pattern_id = self._generate_id(content_hash)

        # Check for duplicate
        if pattern_id in self.success_patterns:
            # Update existing pattern's success count
            existing = self.success_patterns[pattern_id]
            existing.retrieval_count += 1
            return pattern_id

        # Infer domain tags if not provided
        if not domain_tags:
            domain_tags = self._infer_domain_tags(code_snippet)

        # Create and store pattern
        pattern = SuccessPattern(
            id=pattern_id,
            pattern_type=pattern_type,
            description=description,
            code_snippet=code_snippet[:1000],  # Truncate
            metrics_improvement=metrics_improvement,
            domain_tags=domain_tags,
        )

        self.success_patterns[pattern_id] = pattern
        self._pattern_type_index[pattern_type].append(pattern_id)
        for tag in domain_tags:
            self._domain_tag_index[tag].append(pattern_id)

        self.stats["patterns_stored"] += 1

        # Create cross-domain links
        if self.config.cross_domain_links_enabled:
            self._create_cross_domain_links(pattern)

        # Prune if over limit
        if len(self.success_patterns) > self.config.max_patterns:
            self._prune_patterns()

        logger.debug(f"Stored success pattern: {pattern_id} ({pattern_type})")
        return pattern_id

    def _infer_domain_tags(self, code_snippet: str) -> List[str]:
        """Infer domain tags from code content"""
        tags = []
        code_lower = code_snippet.lower()

        # Algorithm patterns
        if "sort" in code_lower:
            tags.append("sorting")
        if "search" in code_lower or "find" in code_lower:
            tags.append("search")
        if "optimize" in code_lower or "minimize" in code_lower or "maximize" in code_lower:
            tags.append("optimization")
        if "random" in code_lower:
            tags.append("stochastic")
        if "gradient" in code_lower or "descent" in code_lower:
            tags.append("gradient-based")
        if "temperature" in code_lower or "anneal" in code_lower:
            tags.append("simulated-annealing")
        if "population" in code_lower or "mutation" in code_lower:
            tags.append("evolutionary")

        # Data structure patterns
        if "list" in code_lower or "array" in code_lower:
            tags.append("array-based")
        if "dict" in code_lower or "hash" in code_lower:
            tags.append("hash-based")
        if "tree" in code_lower or "node" in code_lower:
            tags.append("tree-based")

        # Library patterns
        if "numpy" in code_lower or "np." in code_lower:
            tags.append("numpy")
        if "math" in code_lower:
            tags.append("math")

        return tags if tags else ["general"]

    def retrieve_success_patterns(
        self,
        pattern_type: Optional[str] = None,
        domain_tags: Optional[List[str]] = None,
        for_grokking: bool = False,
        top_k: Optional[int] = None,
    ) -> List[SuccessPattern]:
        """
        Retrieve relevant success patterns.

        Args:
            pattern_type: Filter by pattern type
            domain_tags: Filter by domain tags
            for_grokking: If True, prioritize cross-domain patterns
            top_k: Number of patterns to retrieve

        Returns:
            List of relevant success patterns
        """
        if not self.config.enabled or not self.config.success_patterns_enabled:
            return []

        top_k = top_k or self.config.retrieval_top_k
        candidates = []

        if pattern_type and pattern_type in self._pattern_type_index:
            pattern_ids = self._pattern_type_index[pattern_type]
            candidates = [self.success_patterns[pid] for pid in pattern_ids if pid in self.success_patterns]
        elif domain_tags:
            # Get patterns with matching tags
            matching_ids = set()
            for tag in domain_tags:
                if tag in self._domain_tag_index:
                    matching_ids.update(self._domain_tag_index[tag])
            candidates = [self.success_patterns[pid] for pid in matching_ids if pid in self.success_patterns]
        else:
            candidates = list(self.success_patterns.values())

        # Sort by relevance
        if for_grokking:
            # For grokking, prioritize patterns with cross-domain successes
            candidates.sort(
                key=lambda x: (x.cross_domain_successes, max(x.metrics_improvement.values(), default=0)),
                reverse=True
            )
        else:
            # Standard: prioritize by improvement magnitude
            candidates.sort(
                key=lambda x: (max(x.metrics_improvement.values(), default=0), x.retrieval_count),
                reverse=True
            )

        # Update retrieval counts
        for pattern in candidates[:top_k]:
            pattern.retrieval_count += 1

        self.stats["retrievals"] += 1
        return candidates[:top_k]

    # ==================== Cross-Domain Links ====================

    def _create_cross_domain_links(self, new_pattern: SuccessPattern) -> None:
        """Create links between new pattern and existing patterns"""
        for existing in self.success_patterns.values():
            if existing.id == new_pattern.id:
                continue

            # Calculate similarity based on tags
            common_tags = set(new_pattern.domain_tags) & set(existing.domain_tags)
            all_tags = set(new_pattern.domain_tags) | set(existing.domain_tags)

            if all_tags:
                similarity = len(common_tags) / len(all_tags)

                # Only create links for "weak" similarities (not too similar, not too different)
                if self.config.min_similarity_for_link <= similarity <= self.config.max_similarity_for_link:
                    link = CrossDomainLink(
                        source_pattern_id=new_pattern.id,
                        target_pattern_id=existing.id,
                        similarity_score=similarity,
                    )
                    self.cross_domain_links.append(link)

        # Prune links if over limit
        if len(self.cross_domain_links) > self.config.max_links:
            self._prune_links()

    def get_cross_domain_suggestions(
        self,
        current_domain_tags: List[str],
        top_k: int = 3,
    ) -> List[SuccessPattern]:
        """
        Get patterns from different domains that might help.

        This is the "grokking" mechanism - finding unexpected connections.
        """
        if not self.config.enabled or not self.config.cross_domain_links_enabled:
            return []

        # Find patterns NOT in current domain but linked to it
        current_pattern_ids = set()
        for tag in current_domain_tags:
            if tag in self._domain_tag_index:
                current_pattern_ids.update(self._domain_tag_index[tag])

        # Get linked patterns from other domains
        cross_domain_patterns = []
        for link in self.cross_domain_links:
            if link.source_pattern_id in current_pattern_ids:
                target = self.success_patterns.get(link.target_pattern_id)
                if target and link.target_pattern_id not in current_pattern_ids:
                    cross_domain_patterns.append((target, link.similarity_score, link.transfer_successes))
            elif link.target_pattern_id in current_pattern_ids:
                source = self.success_patterns.get(link.source_pattern_id)
                if source and link.source_pattern_id not in current_pattern_ids:
                    cross_domain_patterns.append((source, link.similarity_score, link.transfer_successes))

        # Sort by potential (transfer successes + inverse similarity for novelty)
        cross_domain_patterns.sort(
            key=lambda x: (x[2], 1 - x[1]),  # Prioritize proven transfers, then novelty
            reverse=True
        )

        self.stats["cross_domain_transfers"] += min(top_k, len(cross_domain_patterns))
        return [p[0] for p in cross_domain_patterns[:top_k]]

    def record_transfer_success(self, pattern_id: str, was_cross_domain: bool = False) -> None:
        """Record that a pattern was successfully applied"""
        if pattern_id in self.success_patterns:
            pattern = self.success_patterns[pattern_id]
            if was_cross_domain:
                pattern.cross_domain_successes += 1
                # Update link success counts
                for link in self.cross_domain_links:
                    if link.source_pattern_id == pattern_id or link.target_pattern_id == pattern_id:
                        link.transfer_successes += 1
            self.stats["successful_applications"] += 1

    # ==================== Pruning ====================

    def _prune_failures(self) -> None:
        """Prune old/unused failure lessons"""
        # Sort by usefulness (successful applications) and recency
        sorted_lessons = sorted(
            self.failure_lessons.values(),
            key=lambda x: (x.success_after_application, x.timestamp),
            reverse=True
        )

        # Keep top max_failures
        keep_ids = {l.id for l in sorted_lessons[:self.config.max_failures]}

        # Remove old ones
        to_remove = [lid for lid in self.failure_lessons if lid not in keep_ids]
        for lid in to_remove:
            lesson = self.failure_lessons.pop(lid)
            if lesson.error_type in self._error_type_index:
                self._error_type_index[lesson.error_type] = [
                    x for x in self._error_type_index[lesson.error_type] if x != lid
                ]

        logger.debug(f"Pruned {len(to_remove)} failure lessons")

    def _prune_patterns(self) -> None:
        """Prune old/unused success patterns"""
        sorted_patterns = sorted(
            self.success_patterns.values(),
            key=lambda x: (x.cross_domain_successes, max(x.metrics_improvement.values(), default=0), x.timestamp),
            reverse=True
        )

        keep_ids = {p.id for p in sorted_patterns[:self.config.max_patterns]}

        to_remove = [pid for pid in self.success_patterns if pid not in keep_ids]
        for pid in to_remove:
            pattern = self.success_patterns.pop(pid)
            if pattern.pattern_type in self._pattern_type_index:
                self._pattern_type_index[pattern.pattern_type] = [
                    x for x in self._pattern_type_index[pattern.pattern_type] if x != pid
                ]
            for tag in pattern.domain_tags:
                if tag in self._domain_tag_index:
                    self._domain_tag_index[tag] = [
                        x for x in self._domain_tag_index[tag] if x != pid
                    ]

        logger.debug(f"Pruned {len(to_remove)} success patterns")

    def _prune_links(self) -> None:
        """Prune old/unused cross-domain links"""
        # Sort by usefulness
        self.cross_domain_links.sort(
            key=lambda x: (x.transfer_successes, x.timestamp),
            reverse=True
        )

        # Keep top max_links
        self.cross_domain_links = self.cross_domain_links[:self.config.max_links]

        logger.debug(f"Pruned cross-domain links to {len(self.cross_domain_links)}")

    # ==================== Prompt Formatting ====================

    def format_for_prompt(
        self,
        failure_lessons: Optional[List[FailureLesson]] = None,
        success_patterns: Optional[List[SuccessPattern]] = None,
        cross_domain_suggestions: Optional[List[SuccessPattern]] = None,
    ) -> str:
        """
        Format reflections for inclusion in LLM prompt.

        Returns:
            Formatted string to inject into the evolution prompt
        """
        sections = []

        # Failure lessons section
        if failure_lessons:
            lessons_text = "\n".join([
                f"- **{l.error_type}**: {l.lesson}"
                for l in failure_lessons
            ])
            sections.append(f"""## Lessons from Previous Failures
The following errors have occurred before. Please avoid these issues:

{lessons_text}
""")

        # Success patterns section
        if success_patterns:
            patterns_text = "\n".join([
                f"- **{p.pattern_type}**: {p.description} (improved metrics by up to {max(p.metrics_improvement.values()):.2%})"
                for p in success_patterns
            ])
            sections.append(f"""## Successful Patterns to Consider
These approaches have worked well in similar contexts:

{patterns_text}
""")

        # Cross-domain suggestions (grokking)
        if cross_domain_suggestions:
            suggestions_text = "\n".join([
                f"- From **{', '.join(p.domain_tags)}** domain: {p.description}"
                for p in cross_domain_suggestions
            ])
            sections.append(f"""## Cross-Domain Insights
These patterns from different domains might provide unexpected inspiration:

{suggestions_text}
""")

        return "\n\n".join(sections)

    # ==================== Serialization ====================

    def save(self, path: str) -> None:
        """Save reflection memory to file"""
        data = {
            "failure_lessons": {k: {
                "id": v.id,
                "error_type": v.error_type,
                "error_message": v.error_message,
                "context": v.context,
                "lesson": v.lesson,
                "mutation_type": v.mutation_type,
                "feature_region": v.feature_region,
                "timestamp": v.timestamp,
                "retrieval_count": v.retrieval_count,
                "success_after_application": v.success_after_application,
            } for k, v in self.failure_lessons.items()},
            "success_patterns": {k: {
                "id": v.id,
                "pattern_type": v.pattern_type,
                "description": v.description,
                "code_snippet": v.code_snippet,
                "metrics_improvement": v.metrics_improvement,
                "domain_tags": v.domain_tags,
                "timestamp": v.timestamp,
                "retrieval_count": v.retrieval_count,
                "cross_domain_successes": v.cross_domain_successes,
            } for k, v in self.success_patterns.items()},
            "cross_domain_links": [{
                "source_pattern_id": l.source_pattern_id,
                "target_pattern_id": l.target_pattern_id,
                "similarity_score": l.similarity_score,
                "transfer_successes": l.transfer_successes,
                "timestamp": l.timestamp,
            } for l in self.cross_domain_links],
            "stats": self.stats,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved reflection memory to {path}")

    def load(self, path: str) -> None:
        """Load reflection memory from file"""
        with open(path, 'r') as f:
            data = json.load(f)

        # Restore failure lessons
        self.failure_lessons = {}
        for k, v in data.get("failure_lessons", {}).items():
            self.failure_lessons[k] = FailureLesson(**v)
            self._error_type_index[v["error_type"]].append(k)

        # Restore success patterns
        self.success_patterns = {}
        for k, v in data.get("success_patterns", {}).items():
            self.success_patterns[k] = SuccessPattern(**v)
            self._pattern_type_index[v["pattern_type"]].append(k)
            for tag in v["domain_tags"]:
                self._domain_tag_index[tag].append(k)

        # Restore links
        self.cross_domain_links = [
            CrossDomainLink(**l) for l in data.get("cross_domain_links", [])
        ]

        # Restore stats
        self.stats = data.get("stats", self.stats)

        logger.info(f"Loaded reflection memory from {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the reflection memory"""
        return {
            **self.stats,
            "total_failures": len(self.failure_lessons),
            "total_patterns": len(self.success_patterns),
            "total_links": len(self.cross_domain_links),
            "unique_error_types": len(self._error_type_index),
            "unique_pattern_types": len(self._pattern_type_index),
            "unique_domain_tags": len(self._domain_tag_index),
        }
