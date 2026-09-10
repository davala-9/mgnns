import time

from src.rule_extraction import full_program as full_program_module
from src.rule_extraction.full_program import EquivalentProgramExtractor
from src.rule_extraction.lattice_search import DFSFrontier, PriorityFrontier


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

    def is_superset_of(self, other):
        return other.name == self.name

    def __eq__(self, other):
        return isinstance(other, FakeSubtree) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"FakeSubtree({self.name!r})"


class FakeBaseTree:
    def __init__(self, initial_subtree):
        self.initial_compact = initial_subtree


class FakeInverseDict:
    def __getitem__(self, item):
        return f"pred-{item}"


class FakeUnaryPredPositionDict:
    inverse = FakeInverseDict()


class FakeInternalEncoder:
    unary_pred_position_dict = FakeUnaryPredPositionDict()


def make_extractor(predicate_position, initial_subtree):
    extractor = EquivalentProgramExtractor(
        device="cpu", model=object(), threshold=0.5,
        external_encoder=object(), internal_encoder=FakeInternalEncoder(),
    )
    extractor.base_tree[predicate_position] = FakeBaseTree(initial_subtree)
    return extractor


class TestExtractSmallestRulesFor:
    def test_yields_every_inclusion_minimal_sound_node(self):
        a = FakeSubtree("a", sound=True)
        b = FakeSubtree("b", sound=True)
        root = FakeSubtree("root", successors=[a, b], sound=False)
        extractor = make_extractor(0, root)
        results = list(extractor.extract_smallest_rules_for(0, deadline=None))
        assert results == [a, b]

    def test_stops_at_deadline(self):
        a = FakeSubtree("a", sound=True)
        root = FakeSubtree("root", successors=[a], sound=False)
        extractor = make_extractor(0, root)
        results = list(extractor.extract_smallest_rules_for(0, deadline=time.monotonic() - 1))
        assert results == []

    def test_frontier_factory_is_pluggable(self):
        a = FakeSubtree("a", sound=True)
        root = FakeSubtree("root", successors=[a], sound=False)
        extractor = make_extractor(0, root)
        results = list(extractor.extract_smallest_rules_for(0, deadline=None, frontier_factory=DFSFrontier))
        assert results == [a]


class TestHailMary:
    def test_returns_first_sound_node_on_the_greedy_path(self):
        a = FakeSubtree("a", sound=True)
        root = FakeSubtree("root", successors=[a], sound=False)
        extractor = make_extractor(0, root)
        result = next(extractor.hail_mary(0), None)
        assert result == a

    def test_returns_none_when_nothing_is_sound(self):
        leaf = FakeSubtree("leaf", sound=False)
        root = FakeSubtree("root", successors=[leaf], sound=False)
        extractor = make_extractor(0, root)
        result = next(extractor.hail_mary(0), None)
        assert result is None

    def test_accepts_a_custom_frontier_factory(self):
        # get_all_rules() passes hail_mary_frontier (weighted by Optimisation 2's atom weights) here
        # instead of the default SinglePathFrontier -- confirm the plumbing actually reaches search().
        a = FakeSubtree("a", sound=True)
        root = FakeSubtree("root", successors=[a], sound=False)
        extractor = make_extractor(0, root)
        result = next(extractor.hail_mary(0, frontier_factory=DFSFrontier), None)
        assert result == a


class TestAtomWeights:
    def test_computed_once_and_cached(self, monkeypatch):
        calls = []

        def fake_compute_path_weights(model, base_tree, var_layer_mask, predicate_position):
            calls.append(predicate_position)
            return {"weights-for-predicate": predicate_position}

        monkeypatch.setattr(full_program_module, "compute_path_weights", fake_compute_path_weights)
        extractor = make_extractor(0, FakeSubtree("root"))
        extractor.var_layer_mask[0] = {}

        first, second = extractor.atom_weights(0), extractor.atom_weights(0)

        assert first is second
        assert calls == [0]  # only computed once, on the first call


class TestHailMaryFrontier:
    def test_builds_a_priority_frontier_out_of_atom_weights(self, monkeypatch):
        sentinel_weights = {"sentinel": True}
        captured = {}

        monkeypatch.setattr(full_program_module, "compute_path_weights",
                            lambda model, base_tree, var_layer_mask, predicate_position: sentinel_weights)

        def fake_path_weight_score_fn(base_tree, weights):
            captured["base_tree"], captured["weights"] = base_tree, weights
            return lambda node: 0.0

        monkeypatch.setattr(full_program_module, "path_weight_score_fn", fake_path_weight_score_fn)
        extractor = make_extractor(0, FakeSubtree("root"))
        extractor.var_layer_mask[0] = {}

        frontier = extractor.hail_mary_frontier(0)

        assert isinstance(frontier, PriorityFrontier)
        assert captured["base_tree"] is extractor.base_tree[0]
        assert captured["weights"] is sentinel_weights
