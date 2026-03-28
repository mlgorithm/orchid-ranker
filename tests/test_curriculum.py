"""Tests for curriculum modules: PrerequisiteGraph and CurriculumRecommender."""

from orchid_ranker.curriculum import PrerequisiteGraph, CurriculumRecommender


class TestPrerequisiteGraph:
    """Test PrerequisiteGraph for curriculum dependency modeling."""

    def test_initialization_empty(self):
        """Test initialization with no edges."""
        graph = PrerequisiteGraph()
        assert len(graph._vertices) == 0
        assert graph.topological_order() == []

    def test_initialization_with_edges(self):
        """Test initialization with edge list."""
        edges = [('a', 'b'), ('b', 'c')]
        graph = PrerequisiteGraph(edges=edges)
        assert len(graph._vertices) == 3
        assert 'a' in graph._vertices
        assert 'c' in graph._vertices

    def test_add_single_edge(self):
        """Test adding a single edge."""
        graph = PrerequisiteGraph()
        graph.add_edge('algebra', 'calculus')
        assert 'algebra' in graph._vertices
        assert 'calculus' in graph._vertices
        assert 'calculus' in graph.dependents_of('algebra')

    def test_add_edge_self_loop_raises(self):
        """Test that self-loops are rejected."""
        graph = PrerequisiteGraph()
        try:
            graph.add_edge('a', 'a')
            assert False, "Should raise ValueError for self-loop"
        except ValueError as e:
            assert "self-loop" in str(e).lower()

    def test_add_edge_cycle_detection(self):
        """Test that cycles are detected and rejected."""
        graph = PrerequisiteGraph()
        graph.add_edge('a', 'b')
        graph.add_edge('b', 'c')
        try:
            graph.add_edge('c', 'a')
            assert False, "Should raise ValueError for cycle"
        except ValueError as e:
            assert "cycle" in str(e).lower()

    def test_add_multiple_edges(self):
        """Test adding multiple edges at once."""
        edges = [('algebra', 'precalc'), ('trigonometry', 'precalc'), ('precalc', 'calculus')]
        graph = PrerequisiteGraph()
        graph.add_edges(edges)
        assert len(graph._vertices) == 4
        assert graph.prerequisites_for('calculus') == {'precalc'}

    def test_add_edges_validates_before_adding(self):
        """Test that add_edges validates all edges before adding any."""
        graph = PrerequisiteGraph([('a', 'b'), ('b', 'c')])
        # Try to add edges including one that would create a cycle
        try:
            graph.add_edges([('c', 'd'), ('d', 'a')])  # Second creates cycle
            assert False, "Should raise ValueError"
        except ValueError:
            # Graph should not be modified
            assert 'd' not in graph._vertices

    def test_prerequisites_for(self):
        """Test querying direct prerequisites."""
        graph = PrerequisiteGraph([('a', 'c'), ('b', 'c')])
        prereqs = graph.prerequisites_for('c')
        assert prereqs == {'a', 'b'}
        assert graph.prerequisites_for('a') == set()
        assert graph.prerequisites_for('nonexistent') == set()

    def test_all_prerequisites_for(self):
        """Test querying all transitive prerequisites."""
        graph = PrerequisiteGraph([
            ('a', 'b'),
            ('b', 'c'),
            ('c', 'd'),
        ])
        all_prereqs = graph.all_prerequisites_for('d')
        assert all_prereqs == {'a', 'b', 'c'}
        assert graph.all_prerequisites_for('a') == set()

    def test_all_prerequisites_multiple_paths(self):
        """Test transitive closure with multiple paths."""
        graph = PrerequisiteGraph([
            ('a', 'c'),
            ('a', 'b'),
            ('b', 'c'),
            ('c', 'd'),
        ])
        all_prereqs = graph.all_prerequisites_for('d')
        assert all_prereqs == {'a', 'b', 'c'}

    def test_dependents_of(self):
        """Test querying direct dependents."""
        graph = PrerequisiteGraph([('a', 'b'), ('a', 'c')])
        dependents = graph.dependents_of('a')
        assert dependents == {'b', 'c'}
        assert graph.dependents_of('b') == set()

    def test_topological_order_simple(self):
        """Test topological sort on simple graph."""
        graph = PrerequisiteGraph([('a', 'b'), ('b', 'c')])
        order = graph.topological_order()
        assert order == ['a', 'b', 'c']

    def test_topological_order_complex(self):
        """Test topological sort on more complex graph."""
        edges = [
            ('algebra', 'calculus'),
            ('trigonometry', 'calculus'),
            ('calculus', 'differential_equations'),
        ]
        graph = PrerequisiteGraph(edges=edges)
        order = graph.topological_order()
        # All prerequisites must come before dependents
        assert order.index('algebra') < order.index('calculus')
        assert order.index('trigonometry') < order.index('calculus')
        assert order.index('calculus') < order.index('differential_equations')

    def test_is_ready_no_prerequisites(self):
        """Test is_ready when skill has no prerequisites."""
        graph = PrerequisiteGraph([('a', 'b')])
        assert graph.is_ready('a', set()) is True
        assert graph.is_ready('c', set()) is True  # doesn't exist

    def test_is_ready_prerequisites_met(self):
        """Test is_ready when prerequisites are satisfied."""
        graph = PrerequisiteGraph([('a', 'b'), ('c', 'b')])
        assert graph.is_ready('b', {'a', 'c'}) is True
        assert graph.is_ready('b', {'a'}) is False

    def test_is_ready_already_mastered(self):
        """Test is_ready returns False for already mastered skill."""
        graph = PrerequisiteGraph([('a', 'b')])
        assert graph.is_ready('b', {'a', 'b'}) is False

    def test_available_skills_empty_mastery(self):
        """Test available skills with no mastered skills."""
        graph = PrerequisiteGraph([('a', 'b'), ('b', 'c')])
        available = graph.available_skills(set())
        assert available == ['a']

    def test_available_skills_progression(self):
        """Test available skills as mastery progresses."""
        graph = PrerequisiteGraph([('a', 'b'), ('b', 'c')])
        assert graph.available_skills(set()) == ['a']
        assert graph.available_skills({'a'}) == ['b']
        assert graph.available_skills({'a', 'b'}) == ['c']
        assert graph.available_skills({'a', 'b', 'c'}) == []

    def test_learning_path_already_mastered(self):
        """Test learning path for already mastered skill."""
        graph = PrerequisiteGraph([('a', 'b')])
        path = graph.learning_path('b', mastered={'a', 'b'})
        assert path == []

    def test_learning_path_linear(self):
        """Test learning path for linear prerequisite chain."""
        graph = PrerequisiteGraph([('a', 'b'), ('b', 'c')])
        path = graph.learning_path('c', mastered=set())
        assert path == ['a', 'b', 'c']

    def test_learning_path_partial_mastery(self):
        """Test learning path with partial mastery."""
        graph = PrerequisiteGraph([('a', 'b'), ('b', 'c')])
        path = graph.learning_path('c', mastered={'a'})
        assert path == ['b', 'c']

    def test_learning_path_nonexistent(self):
        """Test learning path for non-existent skill."""
        graph = PrerequisiteGraph([('a', 'b')])
        path = graph.learning_path('nonexistent', mastered=set())
        assert path == []

    def test_validate_dag(self):
        """Test that validate passes for valid DAG."""
        graph = PrerequisiteGraph([('a', 'b'), ('b', 'c')])
        graph.validate()  # Should not raise

    def test_validate_cyclic_raises(self):
        """Test that validate detects cycles."""
        graph = PrerequisiteGraph()
        graph._edges['a'].add('b')
        graph._reverse_edges['b'].add('a')
        graph._edges['b'].add('a')
        graph._reverse_edges['a'].add('b')
        graph._vertices = {'a', 'b'}

        try:
            graph.validate()
            assert False, "Should raise ValueError for cycle"
        except ValueError as e:
            assert "cycle" in str(e).lower()

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        graph = PrerequisiteGraph([('a', 'b'), ('b', 'c')])
        data = graph.to_dict()
        assert 'edges' in data
        assert ('a', 'b') in data['edges']
        assert ('b', 'c') in data['edges']

    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        edges = [('a', 'b'), ('b', 'c')]
        original = PrerequisiteGraph(edges=edges)
        data = original.to_dict()
        restored = PrerequisiteGraph.from_dict(data)
        assert restored.topological_order() == original.topological_order()

    def test_summary_non_empty_graph(self):
        """Test summary output for non-empty graph."""
        graph = PrerequisiteGraph([('a', 'b'), ('b', 'c'), ('c', 'd')])
        summary = graph.summary()
        assert 'PrerequisiteGraph' in summary
        assert 'vertices' in summary.lower()
        assert 'edges' in summary.lower()

    def test_summary_empty_graph(self):
        """Test summary output for empty graph."""
        graph = PrerequisiteGraph()
        summary = graph.summary()
        assert 'empty' in summary.lower() or '0 vertices' in summary.lower()


class TestCurriculumRecommender:
    """Test CurriculumRecommender for pedagogical skill recommendations."""

    def test_initialization_valid(self):
        """Test initialization with valid graph."""
        graph = PrerequisiteGraph([('a', 'b')])
        rec = CurriculumRecommender(graph)
        assert rec.graph is graph
        assert rec.difficulty_map == {}

    def test_initialization_with_difficulty_map(self):
        """Test initialization with difficulty scores."""
        graph = PrerequisiteGraph([('a', 'b')])
        difficulty = {'a': 0.3, 'b': 0.7}
        rec = CurriculumRecommender(graph, difficulty_map=difficulty)
        assert rec.difficulty_map == difficulty

    def test_initialization_invalid_difficulty_raises(self):
        """Test that invalid difficulty values raise ValueError."""
        graph = PrerequisiteGraph([('a', 'b')])
        try:
            CurriculumRecommender(graph, difficulty_map={'a': 1.5})
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "difficulty" in str(e).lower()

    def test_recommend_empty_graph(self):
        """Test recommendation on empty graph."""
        graph = PrerequisiteGraph()
        rec = CurriculumRecommender(graph)
        recommendations = rec.recommend({'a'}, n=5)
        assert recommendations == []

    def test_recommend_simple_chain(self):
        """Test recommendation on simple prerequisite chain."""
        graph = PrerequisiteGraph([('a', 'b'), ('b', 'c')])
        rec = CurriculumRecommender(graph)
        # With no mastery, should recommend 'a'
        recommendations = rec.recommend(set(), n=5)
        assert 'a' in recommendations

        # After mastering 'a', should recommend 'b'
        recommendations = rec.recommend({'a'}, n=5)
        assert 'b' in recommendations
        assert 'a' not in recommendations

    def test_recommend_respects_prerequisites(self):
        """Test that recommendations respect prerequisite constraints."""
        graph = PrerequisiteGraph([('a', 'c'), ('b', 'c'), ('c', 'd')])
        rec = CurriculumRecommender(graph)
        # Mastered only 'a', not 'b'
        recommendations = rec.recommend({'a'}, n=5)
        # Should not recommend 'c' (needs both 'a' and 'b')
        assert 'c' not in recommendations
        assert 'b' in recommendations  # Can do 'b'

    def test_recommend_with_difficulty_ordering(self):
        """Test that recommendations are ordered by difficulty."""
        graph = PrerequisiteGraph([('a', 'b'), ('a', 'c'), ('a', 'd')])
        difficulty = {'b': 0.5, 'c': 0.3, 'd': 0.8}
        rec = CurriculumRecommender(graph, difficulty_map=difficulty)
        recommendations = rec.recommend({'a'}, n=3)
        # Should recommend in order of increasing difficulty
        assert len(recommendations) > 0
        if len(recommendations) >= 2:
            idx_c = recommendations.index('c') if 'c' in recommendations else -1
            idx_b = recommendations.index('b') if 'b' in recommendations else -1
            if idx_c >= 0 and idx_b >= 0:
                assert idx_c < idx_b  # c has lower difficulty

    def test_recommend_n_parameter(self):
        """Test that n parameter limits recommendations."""
        graph = PrerequisiteGraph([('a', 'b'), ('a', 'c'), ('a', 'd')])
        rec = CurriculumRecommender(graph)
        recommendations = rec.recommend(set(), n=1)
        assert len(recommendations) == 1

        recommendations = rec.recommend(set(), n=10)
        assert len(recommendations) <= 1  # Only 'a' is available

    def test_recommend_excludes_mastered(self):
        """Test that recommendations exclude mastered skills."""
        graph = PrerequisiteGraph([('a', 'b'), ('b', 'c')])
        rec = CurriculumRecommender(graph)
        recommendations = rec.recommend({'a', 'b'}, n=10)
        assert 'a' not in recommendations
        assert 'b' not in recommendations
        assert 'c' in recommendations

    def test_filter_candidates_valid(self):
        """Test candidate filtering."""
        graph = PrerequisiteGraph([('a', 'b'), ('a', 'c')])
        rec = CurriculumRecommender(graph)
        candidates = ['b', 'c']
        filtered = rec.filter_candidates(candidates, mastered={'a'})
        assert len(filtered) == 2

    def test_filter_candidates_removes_unready(self):
        """Test that filter_candidates removes unmet prerequisites."""
        graph = PrerequisiteGraph([('a', 'b'), ('b', 'c')])
        rec = CurriculumRecommender(graph)
        candidates = ['b', 'c']
        filtered = rec.filter_candidates(candidates, mastered={'a'})
        # 'c' requires 'b', which is not mastered
        assert 'b' in filtered
        assert 'c' not in filtered

    def test_recommend_deterministic(self):
        """Test that recommendations are deterministic."""
        graph = PrerequisiteGraph([('a', 'b'), ('a', 'c'), ('a', 'd')])
        rec = CurriculumRecommender(graph)
        r1 = rec.recommend(set(), n=3)
        r2 = rec.recommend(set(), n=3)
        assert r1 == r2

    def test_recommend_complex_graph(self):
        """Test recommendation on a complex curriculum."""
        edges = [
            ('reading', 'writing'),
            ('math', 'statistics'),
            ('reading', 'literature'),
            ('writing', 'literature'),
            ('statistics', 'research'),
        ]
        graph = PrerequisiteGraph(edges=edges)
        rec = CurriculumRecommender(graph)

        # Start with nothing
        recs = rec.recommend(set(), n=10)
        assert set(recs) == {'reading', 'math'}

        # After reading and math
        recs = rec.recommend({'reading', 'math'}, n=10)
        assert 'writing' in recs
        assert 'statistics' in recs
        assert 'reading' not in recs

    def test_recommend_unmapped_difficulty_skills(self):
        """Test that unmapped difficulty skills sort to the end."""
        graph = PrerequisiteGraph([('a', 'b'), ('a', 'c')])
        difficulty = {'b': 0.3}  # 'c' not mapped
        rec = CurriculumRecommender(graph, difficulty_map=difficulty)
        recommendations = rec.recommend({'a'}, n=2)
        if 'b' in recommendations and 'c' in recommendations:
            # 'b' should come before 'c' (mapped vs unmapped)
            assert recommendations.index('b') < recommendations.index('c')
