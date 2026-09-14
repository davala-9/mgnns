import time

import torch

from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.iclr22 import ICLREncoderDecoder
from src.model.gnn_architectures import GNN
from src.rule_extraction import full_program as full_program_module
from src.rule_extraction.full_program import EquivalentProgramExtractor
from src.rule_extraction.lattice_search import DFSFrontier, GreedyBestSuccessorFrontier


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


class FakeExternalEncoder:
    def default_candidate_filters(self):
        return []


def make_extractor(predicate_position, initial_subtree):
    extractor = EquivalentProgramExtractor(
        device="cpu", model=object(), threshold=0.5,
        external_encoder=FakeExternalEncoder(), internal_encoder=FakeInternalEncoder(),
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
    def test_builds_a_greedy_frontier_out_of_atom_weights(self, monkeypatch):
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

        assert isinstance(frontier, GreedyBestSuccessorFrontier)
        assert captured["base_tree"] is extractor.base_tree[0]
        assert captured["weights"] is sentinel_weights


class TestBinaryPredicateArityBug:
    # KNOWN BUG (reported against real data), NOW FIXED by ICLREncoderDecoder.filter_children_by_variable_role
    # and .filter_features_by_data_arity (see EquivalentProgramExtractor.candidate_filters): full-program
    # extraction could print a body atom like <R>[?X0] even though R is a BINARY predicate in the
    # signature -- unary syntax, one variable, for a predicate that needs two.
    #
    # Root cause: under the ICLR22 encoding, a canonical variable's structural role -- "pair" (stands
    # for a constant pair, e.g. the root of a binary head) vs "single" (stands for one constant) -- is
    # determined purely by which edge colour connects it to its parent. compute_tree_for() offered a
    # variable's candidate FEATURES purely by backpropagate_relevance's weight-matrix sparsity, with no
    # idea of "pair" vs "single" or of a predicate's arity. So nothing stopped a "single"-typed variable
    # from being offered an arity-2 ("unary-for-X") canonical predicate as a candidate feature. If the
    # search picked it (easy for a real, not perfectly sparse model), unfold_all's
    # unfold_variable_for_single() blindly emitted (data_var, TYPE_PRED, data_predicate) for it via
    # data_predicate_for_feature(), which reverse-looks-up the ORIGINAL predicate name with no arity
    # check -- e.g. (X0, TYPE_PRED, "R") -- printed with unary syntax: <R>[?X0].
    #
    # This model has a signature with TWO canonical predicates: "A" (data arity 1) and "unary-for-R"
    # (R, data arity 2) -- root (=R's pair-term) needs its own "unary-for-R" feature, AND a "single"-typed
    # child (reached via col1) needs a feature too. Crucially, the child's candidate mask offers BOTH "A"
    # and "R" with EQUAL weight (lin_self_1 row for the child's own hidden channel has weight 1 for both),
    # so -- without the fix -- backpropagate_relevance can't tell them apart and the search finds BOTH a
    # correctly-typed minimal rule (child=A) and the buggy one (child=R, printed with unary syntax). Root's
    # own contribution is weighted 2x a single child bit so "root off, child=both bits" (2) doesn't
    # accidentally tie with "root on, child=one bit" (1+1) and register as a spurious third minimal rule.
    # With the fix, "R" is excluded from the child's candidates outright (wrong arity for a single-typed
    # variable), leaving exactly the one correctly-typed rule.
    def test_does_not_print_a_binary_predicate_with_unary_syntax_in_the_body(self):
        model = GNN(feature_dimension=2, num_edge_colours=4, aggregation_1="max", aggregation_2="max")
        for c in range(4):
            model.conv1.weights.data[c] = torch.zeros(4, 2)
            model.conv2.weights.data[c] = torch.zeros(2, 4)
        model.lin_self_2.weight.data = torch.zeros(2, 4)
        model.lin_self_2.weight.data[1, 1] = 2.0    # matrix_A(2): root's own path (2x a single child bit)
        model.conv2.weights.data[0][1, 0] = 1.0     # matrix_B(2, col1): root's col1-child matters
        model.lin_self_1.weight.data = torch.zeros(4, 2)
        model.lin_self_1.weight.data[0, 0] = 1.0    # child's own path <- "A" (position 0)
        model.lin_self_1.weight.data[0, 1] = 1.0    # child's own path <- "R" (position 1) too -- the collision
        model.lin_self_1.weight.data[1, 1] = 1.0    # root's own path <- "R"
        model.lin_self_1.bias.data = torch.zeros(4)
        model.lin_self_2.bias.data = torch.tensor([0.0, 7.5])  # root alone isn't enough; needs the child too

        external_encoder = ICLREncoderDecoder(load_from_document=None, unary_predicates=["A"],
                                              binary_predicates=["R"])
        internal_encoder = CanonicalEncoderDecoder(load_from_document=None,
            unary_predicates=external_encoder.canonical_unary_predicates,
            binary_predicates=external_encoder.canonical_binary_predicates)

        extractor = EquivalentProgramExtractor(torch.device("cpu"), model, threshold=0.5,
                                               external_encoder=external_encoder, internal_encoder=internal_encoder)
        pos = extractor.resolve_predicate_position("R")
        extractor.compute_all_upper_bounds([pos])

        results = list(extractor.extract_smallest_rules_for(pos, deadline=time.monotonic() + 5))
        assert len(results) == 1, f"expected exactly one sound minimal rule for R, got {len(results)}"

        rule_body = extractor.base_tree[pos].extract_from_compact(results[0])
        bodies, head = extractor.external_encoder.unfold_all(
            can_conj=rule_body, internal_encoder=extractor.internal_encoder, head_predicate="R")
        assert bodies == [[("X0", "R", "X1"), ("X0", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "A")]]
        assert head == ("X0", "R", "X1")

        for body in bodies:
            for (s, p, o) in body:
                is_unary_syntax = p == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                printed_predicate = o if is_unary_syntax else p
                arity = extractor.external_encoder.data_pred_to_arity.get(printed_predicate)
                assert not (is_unary_syntax and arity == 2), \
                    f"binary predicate {printed_predicate!r} printed with unary syntax: <{printed_predicate}>[?{s}]"
