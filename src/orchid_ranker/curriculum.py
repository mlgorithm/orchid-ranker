"""Prerequisite graph support for curriculum ordering and learning path planning.

This module provides DAG-based prerequisite modeling for educational curricula,
enabling pedagogically valid skill sequencing and learning path recommendations.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PrerequisiteGraph:
    """Directed acyclic graph of skill prerequisites for curriculum ordering.

    Models the dependency structure between skills/items where mastering
    prerequisite skills is required before advancing to dependent skills.
    Ensures acyclic structure to enable topological ordering and learning paths.

    Parameters
    ----------
    edges : list of (str, str), optional
        List of (prerequisite, dependent) tuples. Each tuple represents
        a directed edge from prerequisite to dependent skill.

    Attributes
    ----------
    _edges : dict[str, set[str]]
        Adjacency list: {prerequisite: {dependent1, dependent2, ...}}
    _reverse_edges : dict[str, set[str]]
        Reverse adjacency list: {dependent: {prerequisite1, prerequisite2, ...}}
    _vertices : set[str]
        All skills/vertices in the graph.

    Examples
    --------
    >>> graph = PrerequisiteGraph()
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

    def add_edge(self, prerequisite: str, dependent: str) -> None:
        """Add a prerequisite relationship.

        Parameters
        ----------
        prerequisite : str
            The skill that must be mastered first.
        dependent : str
            The skill that depends on the prerequisite.

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

    def prerequisites_for(self, skill: str) -> Set[str]:
        """Return all direct prerequisites for a skill.

        Parameters
        ----------
        skill : str
            The skill to query.

        Returns
        -------
        set of str
            Direct prerequisites (parents) of the skill.
            Empty set if skill has no prerequisites or doesn't exist.
        """
        return self._reverse_edges.get(skill, set()).copy()

    def all_prerequisites_for(self, skill: str) -> Set[str]:
        """Return all transitive prerequisites (ancestors) for a skill.

        Uses BFS to find all skills that must be mastered before this skill.

        Parameters
        ----------
        skill : str
            The skill to query.

        Returns
        -------
        set of str
            All transitive prerequisites (ancestors) of the skill.
            Empty set if skill has no prerequisites or doesn't exist.
        """
        if skill not in self._vertices:
            return set()

        visited = set()
        queue = deque(self._reverse_edges.get(skill, set()))

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            queue.extend(self._reverse_edges.get(node, set()) - visited)

        return visited

    def dependents_of(self, skill: str) -> Set[str]:
        """Return skills that directly depend on this skill.

        Parameters
        ----------
        skill : str
            The skill to query.

        Returns
        -------
        set of str
            Direct dependents (children) of the skill.
            Empty set if skill has no dependents or doesn't exist.
        """
        return self._edges.get(skill, set()).copy()

    def topological_order(self) -> List[str]:
        """Return skills in valid learning order (topological sort).

        Uses Kahn's algorithm to produce a topological ordering.

        Returns
        -------
        list of str
            Skills ordered such that all prerequisites appear before their
            dependent skills. Empty list if graph is empty.

        Raises
        ------
        ValueError
            If graph contains a cycle.

        Notes
        -----
        For deterministic output, skills at the same level are ordered
        lexicographically.
        """
        if not self._vertices:
            return []

        # Check for cycles
        self.validate()

        # Compute in-degree for all vertices
        in_degree = {v: len(self._reverse_edges.get(v, set())) for v in self._vertices}

        # Initialize queue with vertices having no prerequisites
        queue = sorted([v for v in self._vertices if in_degree[v] == 0])

        result = []
        while queue:
            # Process vertex with smallest label for determinism
            node = queue.pop(0)
            result.append(node)

            # Process all dependents
            for dependent in sorted(self._edges.get(node, set())):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
                    queue.sort()

        return result

    def is_ready(self, skill: str, mastered: Set[str]) -> bool:
        """Check if all prerequisites for skill are in mastered set.

        Parameters
        ----------
        skill : str
            The skill to check.
        mastered : set of str
            Set of skills the learner has already mastered.

        Returns
        -------
        bool
            True if all direct prerequisites of skill are in mastered set.
            True if skill has no prerequisites.
            False if skill is already in mastered set.
        """
        if skill in mastered:
            return False

        prerequisites = self.prerequisites_for(skill)
        return prerequisites.issubset(mastered)

    def available_skills(self, mastered: Set[str]) -> List[str]:
        """Return skills whose prerequisites are all mastered but skill itself is not.

        A skill is "available" if:
        1. All its prerequisites are in the mastered set
        2. The skill itself is not in the mastered set

        Parameters
        ----------
        mastered : set of str
            Set of skills already mastered.

        Returns
        -------
        list of str
            Available skills sorted lexicographically for determinism.
        """
        available = []
        for skill in self._vertices:
            if skill not in mastered and self.is_ready(skill, mastered):
                available.append(skill)

        return sorted(available)

    def learning_path(
        self, target_skill: str, mastered: Optional[Set[str]] = None
    ) -> List[str]:
        """Return ordered path from current state to target skill.

        Finds the minimal learning path considering currently mastered skills.
        If the target skill has unmet prerequisites, includes only those prerequisites
        in the returned path.

        Parameters
        ----------
        target_skill : str
            The skill to eventually master.
        mastered : set of str, optional
            Set of skills already mastered. Defaults to empty set.

        Returns
        -------
        list of str
            Ordered list of skills to master, starting from unmastered prerequisites
            and ending with target_skill. Empty list if target_skill is already mastered
            or doesn't exist in the graph.

        Raises
        ------
        ValueError
            If target_skill has unmet prerequisites that form a cycle
            (should not happen if graph is valid).

        Examples
        --------
        >>> graph = PrerequisiteGraph([
        ...     ("algebra", "calculus"),
        ...     ("trigonometry", "calculus"),
        ... ])
        >>> graph.learning_path("calculus", mastered={"algebra"})
        ['trigonometry', 'calculus']
        """
        if mastered is None:
            mastered = set()

        if target_skill not in self._vertices:
            return []

        if target_skill in mastered:
            return []

        # Get all prerequisites
        all_prereqs = self.all_prerequisites_for(target_skill)
        unmet = all_prereqs - mastered

        # Find topological order among unmet prerequisites + target
        subgraph_skills = unmet | {target_skill}
        path = [s for s in self.topological_order() if s in subgraph_skills]

        return path

    def validate(self) -> None:
        """Check graph is a DAG (no cycles).

        Uses DFS with three colors to detect cycles: white (unvisited),
        gray (visiting), black (visited).

        Raises
        ------
        ValueError
            If the graph contains a cycle.
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
    def from_dict(cls, data: Dict) -> PrerequisiteGraph:
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

        Returns
        -------
        str
            Summary including vertex count, edge count, and leaf/root nodes.
        """
        if not self._vertices:
            return "Empty graph (0 vertices, 0 edges)"

        root_nodes = sorted([v for v in self._vertices if not self._reverse_edges[v]])
        leaf_nodes = sorted([v for v in self._vertices if not self._edges[v]])

        total_edges = sum(len(targets) for targets in self._edges.values())

        return (
            f"PrerequisiteGraph(vertices={len(self._vertices)}, edges={total_edges})\n"
            f"  Root skills (no prerequisites): {root_nodes}\n"
            f"  Leaf skills (no dependents): {leaf_nodes}"
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


class CurriculumRecommender:
    """Recommends next items/skills respecting prerequisite ordering and mastery.

    Combines a PrerequisiteGraph with optional difficulty weighting to produce
    pedagogically valid recommendations. Incorporates zone of proximal development
    (ZPD) principles by preferring skills that match learner's current level.

    Parameters
    ----------
    graph : PrerequisiteGraph
        DAG of skill prerequisites.
    difficulty_map : dict, optional
        Mapping of skill to difficulty float in [0, 1]. If provided,
        recommendations are sorted by difficulty preference. Defaults to None
        (all available skills equally preferred).

    Attributes
    ----------
    graph : PrerequisiteGraph
        The underlying prerequisite graph.
    difficulty_map : dict or None
        Optional difficulty scores for skills.

    Examples
    --------
    >>> graph = PrerequisiteGraph([
    ...     ("algebra", "calculus"),
    ...     ("trigonometry", "calculus"),
    ... ])
    >>> difficulty = {"algebra": 0.3, "trigonometry": 0.4, "calculus": 0.7}
    >>> rec = CurriculumRecommender(graph, difficulty_map=difficulty)
    >>> rec.recommend({"algebra"}, n=2)
    ['trigonometry', 'calculus']
    """

    def __init__(
        self,
        graph: PrerequisiteGraph,
        difficulty_map: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize recommender.

        Parameters
        ----------
        graph : PrerequisiteGraph
            The prerequisite graph defining skill dependencies.
        difficulty_map : dict, optional
            Mapping {skill: difficulty} where difficulty is float in [0, 1].
            Higher values indicate harder skills. Used for ZPD-aware ordering.
        """
        self.graph = graph
        self.difficulty_map = difficulty_map or {}

        # Validate difficulty_map if provided
        if self.difficulty_map:
            for skill, diff in self.difficulty_map.items():
                if not isinstance(diff, (int, float)) or not 0 <= diff <= 1:
                    raise ValueError(
                        f"Difficulty for '{skill}' must be float in [0, 1], got {diff}"
                    )

    def recommend(
        self, student_mastery: Set[str], n: int = 5
    ) -> List[str]:
        """Recommend next skills considering prerequisites, mastery, and difficulty.

        Produces an ordered list of skills that:
        1. Have all prerequisites met
        2. Are not already mastered
        3. Are sorted by difficulty preference (if difficulty_map provided)

        Parameters
        ----------
        student_mastery : set of str
            Set of skills the student has mastered.
        n : int, optional
            Maximum number of recommendations to return. Defaults to 5.

        Returns
        -------
        list of str
            Recommended skills in order of preference, limited to n items.
            Empty list if no available skills or graph is empty.

        Examples
        --------
        >>> graph = PrerequisiteGraph([("a", "b"), ("b", "c")])
        >>> rec = CurriculumRecommender(graph)
        >>> rec.recommend({"a"}, n=2)
        ['b', 'c']
        """
        # Get skills whose prerequisites are met
        candidates = self.graph.available_skills(student_mastery)

        # Filter to only valid candidates
        candidates = self.filter_candidates(candidates, student_mastery)

        # Sort by difficulty if map provided
        if self.difficulty_map:
            # Sort by difficulty (ascending), with unmapped skills at end
            candidates = sorted(
                candidates,
                key=lambda s: (
                    self.difficulty_map.get(s, float("inf")),
                    s,  # Tie-breaker: alphabetical
                ),
            )

        return candidates[:n]

    def filter_candidates(self, candidates: List[str], mastered: Set[str]) -> List[str]:
        """Filter candidate items to only those whose prerequisites are met.

        Validates that all candidates have their direct prerequisites satisfied.
        This is a stricter check than available_skills() as it ensures each
        candidate individually is valid.

        Parameters
        ----------
        candidates : list of str
            Initial list of candidate skills.
        mastered : set of str
            Set of mastered skills.

        Returns
        -------
        list of str
            Filtered list containing only candidates with prerequisites met.
            Order is preserved from input.
        """
        filtered = []
        for candidate in candidates:
            if self.graph.is_ready(candidate, mastered):
                filtered.append(candidate)

        return filtered
