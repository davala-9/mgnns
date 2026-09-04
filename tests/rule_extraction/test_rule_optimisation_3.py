import time
import pytest

from src.rule_extraction.rule_optimisation_3 import (
    DFSFrontier,
    BFSFrontier,
    RuleOptimisation3,
)


class FakeSubtree:

    def __init__(self, name, successors=None, sound=False):
        self.name = name
        self._successors = successors or []
        self._sound = sound
        self.soundness_calls = 0

    def check_soundness(self, base_tree, device, model, threshold, pred_position):
        self.soundness_calls += 1
        return self._sound

    def get_successors(self, base_tree):
        return list(self._successors)

    def __eq__(self, other):
        return isinstance(other, FakeSubtree) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"FakeSubtree({self.name!r})"


class FakeBaseTree:
    def __init__(self, initial_subtree):
        self.initial_compact = initial_subtree


@pytest.fixture
def optimiser_factory():
    def _make(base_tree):
        return RuleOptimisation3(
            device="cpu",
            model=object(),
            threshold=0.5,
            pred_position=0,
            base_tree=base_tree,
        )
    return _make


class TestDFSFrontier:
    def test_is_empty_initially(self):
        assert DFSFrontier().is_empty()

    def test_pop_returns_last_pushed(self):
        frontier = DFSFrontier()
        a, b, c = FakeSubtree("a"), FakeSubtree("b"), FakeSubtree("c")
        frontier.push(a)
        frontier.push(b)
        frontier.push(c)
        assert frontier.pop() == c
        assert frontier.pop() == b
        assert frontier.pop() == a
        assert frontier.is_empty()


class TestBFSFrontier:
    def test_is_empty_initially(self):
        assert BFSFrontier().is_empty()

    def test_pop_returns_first_pushed(self):
        frontier = BFSFrontier()
        a, b, c = FakeSubtree("a"), FakeSubtree("b"), FakeSubtree("c")
        frontier.push(a)
        frontier.push(b)
        frontier.push(c)
        assert frontier.pop() == a
        assert frontier.pop() == b
        assert frontier.pop() == c
        assert frontier.is_empty()


class TestGraphSearch:
    def test_returns_initial_subtree_if_already_sound(self, optimiser_factory):
        root = FakeSubtree("root", sound=True)
        opt = optimiser_factory(FakeBaseTree(root))
        result = opt.graph_search(DFSFrontier())
        assert result == root

    def test_finds_sound_successor(self, optimiser_factory):
        target = FakeSubtree("target", sound=True)
        root = FakeSubtree("root", successors=[target], sound=False)
        opt = optimiser_factory(FakeBaseTree(root))
        result = opt.graph_search(BFSFrontier())
        assert result == target

    def test_returns_none_when_no_sound_subtree_exists(self, optimiser_factory):
        leaf = FakeSubtree("leaf", sound=False)
        root = FakeSubtree("root", successors=[leaf], sound=False)
        opt = optimiser_factory(FakeBaseTree(root))
        result = opt.graph_search(DFSFrontier())
        assert result is None

    def test_does_not_revisit_explored_nodes(self, optimiser_factory):
        # A cycle: root -> child -> root
        root = FakeSubtree("root", sound=False)
        child = FakeSubtree("child", successors=[root], sound=False)
        root._successors = [child]
        opt = optimiser_factory(FakeBaseTree(root))
        result = opt.graph_search(BFSFrontier())
        assert result is None
        # Each node's soundness should only be checked once despite the cycle.
        assert root.soundness_calls == 1
        assert child.soundness_calls == 1

    def test_timeout_returns_none(self, optimiser_factory, monkeypatch):
        root = FakeSubtree("root", sound=False)
        opt = optimiser_factory(FakeBaseTree(root))
        times = iter([0.0, 100.0])
        monkeypatch.setattr(time, "monotonic", lambda: next(times, 100.0))
        result = opt.graph_search(DFSFrontier(), timeout=1.0)
        assert result is None


class TestMinimiseRule:
    def test_returns_bfs_result_when_found(self, optimiser_factory, monkeypatch):
        target = FakeSubtree("target", sound=True)
        root = FakeSubtree("root", successors=[target], sound=False)
        opt = optimiser_factory(FakeBaseTree(root))
        called_frontiers = []
        original_graph_search = opt.graph_search
        def spy(frontier, timeout=None):
            called_frontiers.append(type(frontier).__name__)
            return original_graph_search(frontier, timeout=timeout)
        monkeypatch.setattr(opt, "graph_search", spy)
        result = opt.minimise_rule()
        assert result == target
        assert called_frontiers == ["BFSFrontier"]

    def test_falls_back_to_spf_when_bfs_times_out(self, optimiser_factory, capsys):
        target = FakeSubtree("target", sound=True)
        root = FakeSubtree("root", successors=[target], sound=False)
        opt = optimiser_factory(FakeBaseTree(root))
        calls = []
        original_graph_search = opt.graph_search
        def fake_graph_search(frontier, timeout=None):
            calls.append((type(frontier).__name__, timeout))
            if isinstance(frontier, BFSFrontier):
                return None  # simulate a timeout on the first attempt
            return original_graph_search(frontier, timeout=timeout)
        opt.graph_search = fake_graph_search
        result = opt.minimise_rule()
        assert result == target
        assert [name for name, _ in calls] == ["BFSFrontier", "SinglePathFrontier"]
        assert "timed out" in capsys.readouterr().out.lower()

    def test_returns_none_when_both_frontiers_fail(self, optimiser_factory):
        leaf = FakeSubtree("leaf", sound=False)
        root = FakeSubtree("root", successors=[leaf], sound=False)
        opt = optimiser_factory(FakeBaseTree(root))
        result = opt.minimise_rule()
        assert result is None