"""Dependency graph and progression recommender for structured catalog paths.

This module provides DAG-based dependency modeling for any domain with ordered
progression: education (structured catalogs), corporate training (certification
paths), rehabilitation (therapy plans), gaming (category trees), and more.
"""
from __future__ import annotations

import heapq
import warnings
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple


class DependencyGraph:
    """Directed acyclic graph of dependencies for progression ordering.

    Models the dependency structure between categories/items where completing
    prerequisites is required before advancing to dependent items. Works for
    any domain: education (structured catalogs), training (cert paths), gaming
    (category trees), rehab (therapy sequences), etc.

    Parameters
    ----------
    edges : list of (str, str), optional
        List of (dependency, dependent) tuples. Each tuple represents
        a directed edge from dependency to dependent item.

    Examples
    --------
    >>> graph = DependencyGraph()
    >>> graph.add_edges([
    ...     ("algebra", "calculus"),
    ...     ("trigonometry", "calculus"),
    ...     ("calculus", "differential_equations"),
    ... ])
    >>> graph.topological_order()
    ['algebra', 'trigonometry', 'calculus', 'differential_equations']
    >>> graph.all_prerequisites_for("differential_equations")
    {'algebra', 'calculus', 'trigonometry'}
    """

    def __init__(self, edges: Optional[List[Tuple[str, str]]] = None) -> None:
        """Initialize empty graph or populate from edge list.

        Parameters
        ----------
        edges : list of (str, str), optional
            List of (prerequisite, dependent) tuples.
        """
        self._edges: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_edges: Dict[str, Set[str]] = defaultdict(set)
        self._vertices: Set[str] = set()

        if edges:
            self.add_edges(edges)

    def __repr__(self) -> str:
        n_nodes = len(self._vertices)
        n_edges = sum(len(deps) for deps in self._edges.values())
        return f"DependencyGraph(nodes={n_nodes}, edges={n_edges})"

    def add_edge(self, prerequisite: str, dependent: str) -> None:
        """Add a dependency relationship.

        Parameters
        ----------
        prerequisite : str
            The node that must be completed first.
        dependent : str
            The node that depends on the prerequisite.

        Raises
        ------
        ValueError
            If adding the edge would create a cycle.

        Notes
        -----
        Cycle detection is performed on-demand. For large graphs,
        use validate() to check acyclicity before operations.
        """
        if prerequisite == dependent:
            raise ValueError(f"Cannot add self-loop: {prerequisite} -> {dependent}")

        # Check for cycle that would be created
        if self._would_create_cycle(prerequisite, dependent):
            raise ValueError(
                f"Adding edge {prerequisite} -> {dependent} would create a cycle"
            )

        self._edges[prerequisite].add(dependent)
        self._reverse_edges[dependent].add(prerequisite)
        self._vertices.add(prerequisite)
        self._vertices.add(dependent)

    def add_edges(self, edges: List[Tuple[str, str]]) -> None:
        """Add multiple prerequisite relationships.

        Parameters
        ----------
        edges : list of (str, str)
            List of (prerequisite, dependent) tuples.

        Raises
        ------
        ValueError
            If any edge would create a cycle.

        Notes
        -----
        All edges are validated before any are added to maintain
        graph consistency on error.
        """
        if not edges:
            return

        # Validate all edges before adding any, considering previously validated edges
        temp_edges = defaultdict(set)
        temp_reverse = defaultdict(set)

        for prereq, dep in edges:
            if prereq == dep:
                raise ValueError(f"Cannot add self-loop: {prereq} -> {dep}")

            # Check if this edge would create a cycle in the current graph + temp edges
            if self._would_create_cycle_with_temp(prereq, dep, temp_edges):
                raise ValueError(
                    f"Adding edge {prereq} -> {dep} would create a cycle"
                )

            # Add to temporary tracking for subsequent validation
            temp_edges[prereq].add(dep)
            temp_reverse[dep].add(prereq)

        # Add all edges to the actual graph
        for prereq, dep in edges:
            self._edges[prereq].add(dep)
            self._reverse_edges[dep].add(prereq)
            self._vertices.add(prereq)
            self._vertices.add(dep)

    def prerequisites_for(self, node: str) -> Set[str]:
        """Return direct prerequisites (parents) of a node.

        Parameters
        ----------
        node : str
            The node to query.

        Returns
        -------
        set of str
            Direct prerequisites. Empty set if none or node doesn't exist.

        Examples
        --------
        >>> graph = DependencyGraph([("a", "b"), ("c", "b")])
        >>> graph.prerequisites_for("b")
        {'a', 'c'}
        """
        return self._reverse_edges.get(node, set()).copy()

    def all_prerequisites_for(self, node: str) -> Set[str]:
        """Return all transitive prerequisites (ancestors) of a node.

        Uses BFS to find everything that must be completed before this node.

        Parameters
        ----------
        node : str
            The node to query.

        Returns
        -------
        set of str
            All transitive prerequisites. Empty set if none or node doesn't exist.
        """
        if node not in self._vertices:
            return set()

        visited = set()
        queue = deque(self._reverse_edges.get(node, set()))

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self._reverse_edges.get(current, set()) - visited)

        return visited

    def dependents_of(self, node: str) -> Set[str]:
        """Return nodes that directly depend on this node.

        Parameters
        ----------
        node : str
            The node to query.

        Returns
        -------
        set of str
            Direct dependents (children). Empty set if none or node doesn't exist.

        Examples
        --------
        >>> graph = DependencyGraph([("a", "b"), ("a", "c")])
        >>> graph.dependents_of("a")
        {'b', 'c'}
        """
        return self._edges.get(node, set()).copy()

    def topological_order(self) -> List[str]:
        """Return nodes in valid progression order (topological sort).

        Uses Kahn's algorithm. All prerequisites appear before their dependents.

        Returns
        -------
        list of str
            Nodes in topological order. Empty list if graph is empty.

        Raises
        ------
        ValueError
            If graph contains a cycle.

        Notes
        -----
        For deterministic output, categories at the same level are ordered
        lexicographically.
        """
        if not self._vertices:
            return []

        # Check for cycles
        self.validate()

        # Compute in-degree for all vertices
        in_degree = {v: len(self._reverse_edges.get(v, set())) for v in self._vertices}

        # Initialize min-heap with vertices having no prerequisites
        heap = sorted([v for v in self._vertices if in_degree[v] == 0])
        heapq.heapify(heap)

        result = []
        while heap:
            # Pop vertex with smallest label for determinism (O(log n) vs O(n log n))
            node = heapq.heappop(heap)
            result.append(node)

            # Process all dependents
            for dependent in self._edges.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    heapq.heappush(heap, dependent)

        return result

    def prerequisites_met(
        self, node: str, completed: Set[str] = None,
        *, mastered: Set[str] = None, succeeded: Set[str] = None,
    ) -> bool:
        """Check if all prerequisites for a node have been completed.

        A node is ready to start when all its direct prerequisites are in
        the completed set and the node itself has not been completed yet.

        Parameters
        ----------
        node : str
            The node to check readiness for.
        completed : set of str
            Set of nodes already completed.
            (Also accepts deprecated aliases ``mastered`` and ``succeeded``.)

        Returns
        -------
        bool
            True if all direct prerequisites are completed and node is not
            yet completed. False otherwise.

        Examples
        --------
        >>> graph = DependencyGraph([("a", "b"), ("b", "c")])
        >>> graph.prerequisites_met("b", {"a"})
        True
        >>> graph.prerequisites_met("c", {"a"})  # "b" not completed
        False
        """
        # Support deprecated keyword aliases
        if completed is None and mastered is not None:
            warnings.warn(
                "mastered is deprecated, use completed instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            completed = mastered
        if completed is None and succeeded is not None:
            warnings.warn(
                "succeeded is deprecated, use completed instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            completed = succeeded
        if completed is None:
            completed = set()

        if node in completed:
            return False

        prerequisites = self.prerequisites_for(node)
        return prerequisites.issubset(completed)

    # Backward-compatible aliases
    is_ready = prerequisites_met

    def available(
        self, completed: Set[str] = None,
        *, mastered: Set[str] = None, succeeded: Set[str] = None,
    ) -> List[str]:
        """Return nodes whose prerequisites are all completed but node itself is not.

        Parameters
        ----------
        completed : set of str
            Set of nodes already completed.
            (Also accepts deprecated aliases ``mastered`` and ``succeeded``.)

        Returns
        -------
        list of str
            Available nodes sorted lexicographically.

        Examples
        --------
        >>> graph = DependencyGraph([("a", "b"), ("b", "c")])
        >>> graph.available({"a"})
        ['b']
        >>> graph.available({"a", "b"})
        ['c']
        """
        # Support deprecated keyword aliases
        if completed is None and mastered is not None:
            warnings.warn(
                "mastered is deprecated, use completed instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            completed = mastered
        if completed is None and succeeded is not None:
            warnings.warn(
                "succeeded is deprecated, use completed instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            completed = succeeded
        if completed is None:
            completed = set()

        result = []
        for v in self._vertices:
            if v in completed:
                continue
            prereqs = self._reverse_edges.get(v, set())
            if prereqs.issubset(completed):
                result.append(v)

        return sorted(result)

    # Backward-compatible aliases
    available_categories = available
    available_skills = available

    def path_to(
        self, target: str, completed: Optional[Set[str]] = None,
        *, mastered: Optional[Set[str]] = None,
        succeeded: Optional[Set[str]] = None,
    ) -> List[str]:
        """Return the ordered progression path to reach a target node.

        Finds the minimal path considering already-completed nodes.

        Parameters
        ----------
        target : str
            The target node to reach.
        completed : set of str, optional
            Set of nodes already completed. Defaults to empty set.
            (Also accepts deprecated aliases ``mastered`` and ``succeeded``.)

        Returns
        -------
        list of str
            Ordered list of nodes to complete, ending with target.
            Empty list if target is already completed or doesn't exist.

        Examples
        --------
        >>> graph = DependencyGraph([
        ...     ("algebra", "calculus"),
        ...     ("trigonometry", "calculus"),
        ... ])
        >>> graph.path_to("calculus", completed={"algebra"})
        ['trigonometry', 'calculus']
        """
        # Support deprecated keyword aliases
        if completed is None and mastered is not None:
            warnings.warn(
                "mastered is deprecated, use completed instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            completed = mastered
        if completed is None and succeeded is not None:
            warnings.warn(
                "succeeded is deprecated, use completed instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            completed = succeeded
        if completed is None:
            completed = set()

        if target not in self._vertices:
            return []

        if target in completed:
            return []

        all_prereqs = self.all_prerequisites_for(target)
        unmet = all_prereqs - completed

        subgraph = unmet | {target}
        path = [s for s in self.topological_order() if s in subgraph]

        return path

    # Backward-compatible alias
    learning_path = path_to

    def validate(self) -> None:
        """Check graph is a DAG (no cycles).

        Uses depth-first search with three colors to detect cycles:
        white (unvisited), gray (visiting), black (visited).

        Raises
        ------
        ValueError
            If the graph contains a cycle.

        Examples
        --------
        >>> graph = PrerequisiteGraph([("a", "b")])
        >>> graph.validate()  # No error
        >>> graph.add_edge("b", "a")  # Would create cycle
        ValueError: Adding edge b -> a would create a cycle
        """
        color = {v: "white" for v in self._vertices}

        def _has_cycle_dfs(node: str) -> bool:
            """Return True if DFS from node finds a cycle."""
            color[node] = "gray"
            for neighbor in self._edges.get(node, set()):
                if color[neighbor] == "gray":
                    # Back edge found (cycle)
                    return True
                if color[neighbor] == "white" and _has_cycle_dfs(neighbor):
                    return True
            color[node] = "black"
            return False

        for vertex in self._vertices:
            if color[vertex] == "white":
                if _has_cycle_dfs(vertex):
                    raise ValueError("Graph contains a cycle (not a DAG)")

    def to_dict(self) -> Dict:
        """Serialize graph to dict.

        Returns
        -------
        dict
            Serializable dictionary with edges list.

        Examples
        --------
        >>> graph = PrerequisiteGraph([("a", "b")])
        >>> data = graph.to_dict()
        >>> data["edges"]
        [('a', 'b')]
        """
        edges = []
        for source, targets in self._edges.items():
            for target in sorted(targets):
                edges.append((source, target))

        return {"edges": sorted(edges)}

    @classmethod
    def from_dict(cls, data: Dict) -> DependencyGraph:
        """Deserialize graph from dict.

        Parameters
        ----------
        data : dict
            Dictionary with "edges" key containing list of (prereq, dependent) tuples.

        Returns
        -------
        PrerequisiteGraph
            Reconstructed graph.

        Raises
        ------
        ValueError
            If edges contain cycles.
        """
        edges = data.get("edges", [])
        return cls(edges=edges)

    def summary(self) -> str:
        """Return human-readable summary of graph structure.

        Generates a text summary with vertex/edge counts and identification
        of root categories (no prerequisites) and leaf categories (no dependents).

        Returns
        -------
        str
            Multi-line summary including vertex count, edge count, and lists
            of root and leaf nodes.

        Examples
        --------
        >>> graph = DependencyGraph([("a", "b"), ("b", "c")])
        >>> print(graph.summary())
        DependencyGraph(vertices=3, edges=2)
          Roots (no dependencies): ['a']
          Leaves (no dependents): ['c']
        """
        if not self._vertices:
            return "Empty graph (0 vertices, 0 edges)"

        root_nodes = sorted([v for v in self._vertices if not self._reverse_edges[v]])
        leaf_nodes = sorted([v for v in self._vertices if not self._edges[v]])

        total_edges = sum(len(targets) for targets in self._edges.values())

        return (
            f"DependencyGraph(vertices={len(self._vertices)}, edges={total_edges})\n"
            f"  Roots (no dependencies): {root_nodes}\n"
            f"  Leaves (no dependents): {leaf_nodes}"
        )

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Check if adding edge source->target would create a cycle.

        Parameters
        ----------
        source : str
            Source of proposed edge.
        target : str
            Target of proposed edge.

        Returns
        -------
        bool
            True if adding this edge would create a cycle.

        Notes
        -----
        A cycle would be created if target can already reach source.
        Uses BFS for efficiency.
        """
        if source == target:
            return True

        # Check if target can reach source (would close a cycle)
        visited = set()
        queue = deque([target])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            if node == source:
                return True

            for next_node in self._edges.get(node, set()):
                if next_node not in visited:
                    queue.append(next_node)

        return False

    def _would_create_cycle_with_temp(
        self, source: str, target: str, temp_edges: Dict[str, Set[str]]
    ) -> bool:
        """Check if adding edge source->target would create a cycle, considering temporary edges.

        Parameters
        ----------
        source : str
            Source of proposed edge.
        target : str
            Target of proposed edge.
        temp_edges : dict[str, set[str]]
            Temporary edges already validated in this batch.

        Returns
        -------
        bool
            True if adding this edge would create a cycle.

        Notes
        -----
        Combines current graph edges with temp edges to detect cycles across
        a batch of edge additions.
        """
        if source == target:
            return True

        # Check if target can reach source using current edges + temp edges
        visited = set()
        queue = deque([target])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            if node == source:
                return True

            # Check both current edges and temp edges
            next_nodes = set(self._edges.get(node, set())) | set(
                temp_edges.get(node, set())
            )
            for next_node in next_nodes:
                if next_node not in visited:
                    queue.append(next_node)

        return False


class ProgressionRecommender:
    """Recommends next items respecting dependency ordering and user competence.

    Combines a DependencyGraph with optional difficulty weighting to produce
    valid progression recommendations. Incorporates stretch zone principles
    by preferring items that match the user's current level.

    Works for any sequenced domain: education (structured catalogs), training
    (cert paths), rehab (therapy plans), gaming (category trees), onboarding
    (feature rollout).

    Parameters
    ----------
    graph : DependencyGraph
        DAG of item dependencies.
    difficulty_map : dict, optional
        Mapping of category to difficulty float in [0, 1]. If provided,
        recommendations are sorted by difficulty preference.

    Attributes
    ----------
    graph : DependencyGraph
        The underlying dependency graph.
    difficulty_map : dict or None
        Optional difficulty scores for categories.

    Examples
    --------
    >>> graph = DependencyGraph([
    ...     ("algebra", "calculus"),
    ...     ("trigonometry", "calculus"),
    ... ])
    >>> difficulty = {"algebra": 0.3, "trigonometry": 0.4, "calculus": 0.7}
    >>> rec = ProgressionRecommender(graph, difficulty_map=difficulty)
    >>> rec.recommend({"algebra"}, n=2)
    ['trigonometry', 'calculus']
    """

    def __init__(
        self,
        graph: DependencyGraph,
        difficulty_map: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize recommender.

        Parameters
        ----------
        graph : DependencyGraph
            The dependency graph defining item dependencies.
        difficulty_map : dict, optional
            Mapping {category: difficulty} where difficulty is float in [0, 1].
            Higher values indicate harder categories.
        """
        self.graph = graph
        self.difficulty_map = difficulty_map or {}

        self._has_difficulty = bool(difficulty_map)

        # Validate difficulty_map if provided
        if self.difficulty_map:
            for category, diff in self.difficulty_map.items():
                if not isinstance(diff, (int, float)) or not 0 <= diff <= 1:
                    raise ValueError(
                        f"Difficulty for '{category}' must be float in [0, 1], got {diff}"
                    )

    def __repr__(self) -> str:
        return (f"ProgressionRecommender(graph={self.graph!r}, "
                f"has_difficulty={self._has_difficulty})")

    def recommend(
        self, completed: Set[str] = None, n: int = 5,
        *, student_mastery: Set[str] = None,
        user_competence: Set[str] = None,
    ) -> List[str]:
        """Recommend next items considering dependencies, progress, and difficulty.

        Returns items that:
        1. Have all prerequisites completed
        2. Are not already completed
        3. Are sorted by difficulty (if difficulty_map provided)

        Parameters
        ----------
        completed : set of str
            Set of items/nodes already completed.
            (Also accepts deprecated aliases ``student_mastery`` and
            ``user_competence``.)
        n : int, optional
            Maximum number of recommendations (default: 5).

        Returns
        -------
        list of str
            Recommended items in order of preference.

        Examples
        --------
        >>> graph = DependencyGraph([("a", "b"), ("b", "c")])
        >>> rec = ProgressionRecommender(graph)
        >>> rec.recommend({"a"}, n=2)
        ['b']
        """
        # Support deprecated keyword aliases
        if completed is None and student_mastery is not None:
            warnings.warn(
                "student_mastery is deprecated, use completed instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            completed = student_mastery
        if completed is None and user_competence is not None:
            warnings.warn(
                "user_competence is deprecated, use completed instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            completed = user_competence
        if completed is None:
            completed = set()

        # Get items whose prerequisites are met
        candidates = self.graph.available(completed)

        # Filter to only valid candidates
        candidates = self.filter_candidates(candidates, completed)

        # Sort by difficulty if map provided
        if self.difficulty_map:
            # Sort by difficulty (ascending), with unmapped categories at end
            candidates = sorted(
                candidates,
                key=lambda s: (
                    self.difficulty_map.get(s, float("inf")),
                    s,  # Tie-breaker: alphabetical
                ),
            )

        return candidates[:n]

    def filter_candidates(
        self, candidates: List[str], completed: Set[str] = None,
        *, mastered: Set[str] = None, succeeded: Set[str] = None,
    ) -> List[str]:
        """Filter candidates to only those whose prerequisites are completed.

        Parameters
        ----------
        candidates : list of str
            Candidate items to filter.
        completed : set of str
            Set of completed items.
            (Also accepts deprecated aliases ``mastered`` and ``succeeded``.)

        Returns
        -------
        list of str
            Filtered list preserving input order.

        Examples
        --------
        >>> graph = DependencyGraph([("a", "b"), ("b", "c")])
        >>> rec = ProgressionRecommender(graph)
        >>> rec.filter_candidates(["b", "c"], {"a"})
        ['b']
        """
        if completed is None and mastered is not None:
            warnings.warn(
                "mastered is deprecated, use completed instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            completed = mastered
        if completed is None and succeeded is not None:
            warnings.warn(
                "succeeded is deprecated, use completed instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            completed = succeeded
        if completed is None:
            completed = set()
        return [c for c in candidates if self.graph.prerequisites_met(c, completed)]


__all__ = [
    "DependencyGraph",
    "ProgressionRecommender",
    # Backward-compatible aliases (deprecated)
    "PrerequisiteGraph",
    "CurriculumRecommender",
    "SkillGraph",
]


# --- Deprecation handling for renamed symbols (PEP 562) ---
_DEPRECATED_NAMES = {
    "PrerequisiteGraph": "DependencyGraph",
    "CurriculumRecommender": "ProgressionRecommender",
    "SkillGraph": "DependencyGraph",
}


def __getattr__(name: str):
    if name in _DEPRECATED_NAMES:
        import warnings
        warnings.warn(
            f"{name} is deprecated, use {_DEPRECATED_NAMES[name]} instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[_DEPRECATED_NAMES[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
