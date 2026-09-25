import numpy as np
import pytest
import torch

from src.adni.fact_to_matrix import _pick_best_neighbours, _set_matrix_entry, derive_matrix_from_fact, \
    derive_minimal_matrix_from_fact
from src.adni.matrix_to_tree import matrix_to_tree
from src.adni.signature import AdniSignature
from src.adni.sparse_triangular_matrix import SparseTriangularMatrix
from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.identity import IdentityEncoderDecoder
from src.model.cd_graph import CDGraph, TraceCollector
from src.model.gnn_architectures import GNN
from src.model.gnn_transformation import apply_model
from src.utils.utils import TYPE_PRED

# d=3 nodes (node_0, node_1, node_2), values in {1, 2}: unary "node", "node_0", "node_1", "node_2",
# "positive" (positions 0..4); binary "part_of", "1.0", "2.0" (colours 0, 1, 2).
UNARY = ["node", "node_0", "node_1", "node_2", "positive"]
BINARY = ["part_of", "1.0", "2.0"]
NODE_POS, NODE_0_POS, NODE_1_POS, NODE_2_POS, POSITIVE_POS = range(5)
PART_OF_COL, COLOUR_1, COLOUR_2 = range(3)
THRESHOLD = 0.5


def make_internal_encoder():
    return CanonicalEncoderDecoder(unary_predicates=UNARY, binary_predicates=BINARY)


# A patient "P" with three regions P0 (node_0), P1 (node_1), P2 (node_2), all part_of P. Value edges
# (symmetric, as the real ADNI data always has both directions of a colour present): P0 <-1.0-> P1,
# P1 <-2.0-> P2. Weights are set up so that:
#  - root's part_of competition at layer 2 only reads hidden position 0 (see conv2 colour "part_of" below);
#  - among P0/P1/P2, only P0 reaches a high value at position 0 at layer 1, and only because it receives a
#    "1.0" message from P1 (P0's own self-contribution alone is not enough to cross the threshold) --
#    so the message is load-bearing, not just a bonus, and a test that dropped it (e.g. by wiring the
#    reconstructed edge backwards) would actually fail the soundness check below.
#  - the root's own self-path (matrix_A(2)) is zeroed out entirely, so this fixture only exercises the
#    two-step part_of-then-value-colour mechanism described to derive_matrix_from_fact, not the separate
#    "root aggregates children's raw features at layer 1 too" mechanism (see
#    project_adni_gnn_aggregation_mechanics in memory) -- deliberately out of scope here.
def make_model():
    model = GNN(feature_dimension=5, num_edge_colours=3, aggregation_1="max", aggregation_2="max")

    a1 = torch.zeros(10, 5)
    a1[0, NODE_POS] = 1.0
    a1[0, NODE_0_POS] = 3.0  # P0's own self-contribution at hidden position 0: 1 + 3 = 4 (not enough alone)
    model.lin_self_1.weight.data = a1
    model.lin_self_1.bias.data = torch.zeros(10)

    b1_part_of = torch.zeros(10, 5)
    b1_colour1 = torch.zeros(10, 5)
    b1_colour1[0, NODE_1_POS] = 9.0  # a "1.0" message from a node_1 neighbour adds 9 at hidden position 0
    b1_colour2 = torch.zeros(10, 5)
    model.conv1.weights.data[PART_OF_COL] = b1_part_of
    model.conv1.weights.data[COLOUR_1] = b1_colour1
    model.conv1.weights.data[COLOUR_2] = b1_colour2

    model.lin_self_2.weight.data = torch.zeros(5, 10)  # root's own self-path contributes nothing (see above)
    model.lin_self_2.bias.data = torch.zeros(5)

    b2_part_of = torch.zeros(5, 10)
    b2_part_of[POSITIVE_POS, 0] = 1.0  # "positive" only reads hidden position 0 of the winning part_of child
    model.conv2.weights.data[PART_OF_COL] = b2_part_of
    model.conv2.weights.data[COLOUR_1] = torch.zeros(5, 10)
    model.conv2.weights.data[COLOUR_2] = torch.zeros(5, 10)
    return model


def make_cd_graph():
    features = torch.tensor([
        [0., 0., 0., 0., 0.],  # P (root): no unary facts of its own
        [1., 1., 0., 0., 0.],  # P0: node, node_0
        [1., 0., 1., 0., 0.],  # P1: node, node_1
        [1., 0., 0., 1., 0.],  # P2: node, node_2
    ], dtype=torch.float)
    # (source, target) pairs, per CDGraph's docstring: column [i, j] is an edge from row i to row j.
    edges = torch.tensor([
        [1, 2, 3, 2, 1, 2, 3],
        [0, 0, 0, 1, 2, 3, 2],
    ], dtype=torch.long)
    edge_colours = torch.tensor(
        [PART_OF_COL, PART_OF_COL, PART_OF_COL, COLOUR_1, COLOUR_1, COLOUR_2, COLOUR_2], dtype=torch.long)
    node_names = ["P", "P0", "P1", "P2"]
    return CDGraph(col_size=3, delta=5, features=features, edges=edges, edge_colours=edge_colours,
                   node_names=node_names)


def make_trace(model):
    trace = TraceCollector()
    apply_model(make_cd_graph(), torch.device("cpu"), model, trace)
    return trace


def make_external_encoder():
    return IdentityEncoderDecoder(unary_predicates=UNARY, binary_predicates=BINARY)


def get_fact():
    return "P", TYPE_PRED, "positive"


class TestPickBestNeighbours:

    def test_picks_the_argmax_neighbour_per_position(self):
        vectors = np.array([[1.0, 5.0], [3.0, 2.0]])  # neighbours 10, 20
        picks = _pick_best_neighbours([0, 1], [10, 20], vectors, chosen=set())
        assert picks == {0: 20, 1: 10}

    def test_skips_a_position_whose_max_is_zero(self):
        vectors = np.array([[0.0], [0.0]])
        picks = _pick_best_neighbours([0], [10, 20], vectors, chosen=set())
        assert picks == {}

    def test_prefers_an_already_chosen_neighbour_on_a_tie(self):
        # Both neighbours tie at value 5 for position 1; neighbour 10 was already picked for position 0,
        # so position 1 should reuse it instead of bringing in neighbour 20 too.
        vectors = np.array([[1.0, 5.0], [0.0, 5.0]])
        chosen = set()
        picks = _pick_best_neighbours([0, 1], [10, 20], vectors, chosen)
        assert picks == {0: 10, 1: 10}
        assert chosen == {10}

    def test_breaks_a_first_tie_deterministically_by_lowest_index(self):
        vectors = np.array([[5.0], [5.0]])
        picks = _pick_best_neighbours([0], [20, 10], vectors, chosen=set())
        assert picks == {0: 10}


class TestSetMatrixEntry:

    def test_writes_at_the_sorted_position_regardless_of_argument_order(self):
        matrix = SparseTriangularMatrix.empty(d=3, max_value=2)
        matrix = _set_matrix_entry(matrix, 2, 0, 1)  # k1 > k2: still lands at (0, 2)
        assert matrix.to_dict() == {(0, 2): 1}

    def test_rewriting_the_same_value_is_a_no_op(self):
        matrix = SparseTriangularMatrix.empty(d=3, max_value=2).with_value(0, 1, 1)
        matrix = _set_matrix_entry(matrix, 1, 0, 1)
        assert matrix.to_dict() == {(0, 1): 1}

    def test_conflicting_values_for_the_same_cell_raise(self):
        matrix = SparseTriangularMatrix.empty(d=3, max_value=2).with_value(0, 1, 1)
        with pytest.raises(ValueError):
            _set_matrix_entry(matrix, 0, 1, 2)


class TestDeriveMatrixFromFact:

    def test_two_step_derivation_matches_the_hand_traced_path(self):
        model = make_model()
        trace = make_trace(model)
        internal_encoder = make_internal_encoder()
        external_encoder = make_external_encoder()

        # Sanity check: the fixture really does derive "positive" on the root, and really does need P0's
        # value-edge message (drop it and the score would fall well under threshold -- see make_model).
        root_idx = trace.cd_graph.node_names_to_indices["P"]
        assert trace.activations[2][root_idx][POSITIVE_POS] > THRESHOLD

        matrix = derive_matrix_from_fact(
            get_fact(), trace, external_encoder, AdniSignature(internal_encoder), model, THRESHOLD)

        # node_0 (P0, root's winning part_of child) recipient, node_1 (P1, its "1.0" neighbour) sender --
        # matching the real direction the message actually flowed in the trace.
        assert matrix.to_dict() == {(0, 1): 1}

    def test_reconstructed_tree_soundly_rederives_positive(self):
        model = make_model()
        trace = make_trace(model)
        internal_encoder = make_internal_encoder()
        external_encoder = make_external_encoder()

        matrix = derive_matrix_from_fact(
            get_fact(), trace, external_encoder, AdniSignature(internal_encoder), model, THRESHOLD)

        # Soundness is checked against every node_k (matrix_to_tree's default), matching
        # matrix_search.check_soundness: every real patient has all d regions present as part_of children
        # of root regardless of what the candidate matrix asserts, so that's the graph to test against.
        tree = matrix_to_tree(matrix, AdniSignature(internal_encoder))
        output_graph = apply_model(tree.as_cd_graph, torch.device("cpu"), model)
        assert output_graph.features[0][POSITIVE_POS].item() >= THRESHOLD

    def test_dropping_the_value_edge_would_fail_soundness(self):
        # Confirms the fixture's own claim (see make_model's docstring) that the "1.0" message is load
        # bearing: a tree with only node_0, no value edge, does NOT reach the threshold.
        model = make_model()
        internal_encoder = make_internal_encoder()
        from src.adni.sparse_triangular_matrix import SparseTriangularMatrix
        matrix = SparseTriangularMatrix.empty(d=3, max_value=2)
        tree = matrix_to_tree(matrix, AdniSignature(internal_encoder), included_nodes={0})
        output_graph = apply_model(tree.as_cd_graph, torch.device("cpu"), model)
        assert output_graph.features[0][POSITIVE_POS].item() < THRESHOLD

    def test_direction_mismatch_still_writes_at_the_sorted_position(self):
        # Here the real message flows P0 (k=0) -> P1 (k=1): the recipient's canonical index is the *larger*
        # one, the reverse of matrix_to_tree's fixed "smaller index is the recipient" convention. Per
        # feedback_adni_matrix_direction_is_symmetric, this is fine: the colour value is a property of the
        # unordered (node_0, node_1) pair, so it's written at the sorted position regardless.
        model = make_model()
        a1 = torch.zeros(10, 5)
        a1[0, NODE_POS] = 1.0
        a1[0, NODE_1_POS] = 3.0  # P1 (k=1, the recipient here) self-contribution alone: not enough
        model.lin_self_1.weight.data = a1
        b1_colour1 = torch.zeros(10, 5)
        b1_colour1[0, NODE_0_POS] = 9.0  # a "1.0" message from a node_0 neighbour (P0, k=0)
        model.conv1.weights.data[COLOUR_1] = b1_colour1

        features = torch.tensor([
            [0., 0., 0., 0., 0.],  # P (root)
            [1., 1., 0., 0., 0.],  # P0
            [1., 0., 1., 0., 0.],  # P1: the winning part_of child this time
            [1., 0., 0., 1., 0.],  # P2
        ], dtype=torch.float)
        edges = torch.tensor([
            [1, 2, 3, 1],
            [0, 0, 0, 2],  # P0 -> P1, colour "1.0": P1 (k=1) is the recipient, P0 (k=0) the sender
        ], dtype=torch.long)
        edge_colours = torch.tensor([PART_OF_COL, PART_OF_COL, PART_OF_COL, COLOUR_1], dtype=torch.long)
        cd_graph = CDGraph(col_size=3, delta=5, features=features, edges=edges, edge_colours=edge_colours,
                            node_names=["P", "P0", "P1", "P2"])
        trace = TraceCollector()
        apply_model(cd_graph, torch.device("cpu"), model, trace)
        internal_encoder = make_internal_encoder()
        external_encoder = make_external_encoder()

        matrix = derive_matrix_from_fact(
            get_fact(), trace, external_encoder, AdniSignature(internal_encoder), model, THRESHOLD)

        assert matrix.to_dict() == {(0, 1): 1}

    def test_no_part_of_neighbours_returns_an_empty_matrix(self):
        model = make_model()
        internal_encoder = make_internal_encoder()
        external_encoder = make_external_encoder()
        features = torch.zeros((1, 5), dtype=torch.float)
        features[0, POSITIVE_POS] = 0.0
        cd_graph = CDGraph(col_size=3, delta=5, features=features, edges=torch.empty((2, 0), dtype=torch.long),
                            edge_colours=torch.empty((0,), dtype=torch.long), node_names=["Lonely"])
        trace = TraceCollector()
        apply_model(cd_graph, torch.device("cpu"), model, trace)
        # This root won't actually be above threshold (no part_of children at all), so bypass the usual
        # assertion in derive_matrix_from_fact by picking a low threshold -- we only care that an isolated
        # root doesn't crash the part_of lookup and returns nothing.
        matrix = derive_matrix_from_fact(
            ("Lonely", TYPE_PRED, "positive"), trace, external_encoder, AdniSignature(internal_encoder), model, threshold=-1.0)
        assert matrix.to_dict() == {}


# A patient where root's "positive" reads TWO hidden positions (0 and 1) instead of one, and P0 wins both --
# once via a "1.0" message from P1 (boosting position 0), once via a "2.0" message from P2 (boosting
# position 1). Each message alone is already enough to cross the threshold (over-determined), so the raw,
# two-edge matrix derive_matrix_from_fact returns is *not* minimal: one of the two edges is redundant.
def make_over_determined_model():
    model = GNN(feature_dimension=5, num_edge_colours=3, aggregation_1="max", aggregation_2="max")

    a1 = torch.zeros(10, 5)
    a1[0, NODE_POS] = 1.0
    a1[0, NODE_0_POS] = 3.0  # P0 self-contribution at position 0: 1 + 3 = 4
    a1[1, NODE_POS] = 1.0
    a1[1, NODE_0_POS] = 3.0  # P0 self-contribution at position 1: 1 + 3 = 4
    model.lin_self_1.weight.data = a1
    model.lin_self_1.bias.data = torch.zeros(10)

    b1_colour1 = torch.zeros(10, 5)
    b1_colour1[0, NODE_1_POS] = 9.0  # a "1.0" message from a node_1 neighbour boosts position 0: 4+9=13
    b1_colour2 = torch.zeros(10, 5)
    b1_colour2[1, NODE_2_POS] = 7.0  # a "2.0" message from a node_2 neighbour boosts position 1: 4+7=11
    model.conv1.weights.data[PART_OF_COL] = torch.zeros(10, 5)
    model.conv1.weights.data[COLOUR_1] = b1_colour1
    model.conv1.weights.data[COLOUR_2] = b1_colour2

    model.lin_self_2.weight.data = torch.zeros(5, 10)
    model.lin_self_2.bias.data = torch.zeros(5)

    b2_part_of = torch.zeros(5, 10)
    b2_part_of[POSITIVE_POS, 0] = 1.0
    b2_part_of[POSITIVE_POS, 1] = 1.0  # "positive" reads BOTH hidden positions 0 and 1
    model.conv2.weights.data[PART_OF_COL] = b2_part_of
    model.conv2.weights.data[COLOUR_1] = torch.zeros(5, 10)
    model.conv2.weights.data[COLOUR_2] = torch.zeros(5, 10)
    return model


def make_over_determined_cd_graph():
    features = torch.tensor([
        [0., 0., 0., 0., 0.],  # P (root)
        [1., 1., 0., 0., 0.],  # P0
        [1., 0., 1., 0., 0.],  # P1
        [1., 0., 0., 1., 0.],  # P2
    ], dtype=torch.float)
    edges = torch.tensor([
        [1, 2, 3, 1, 2, 1, 3],
        [0, 0, 0, 2, 1, 3, 1],  # P0<->P1 ("1.0"), P0<->P2 ("2.0")
    ], dtype=torch.long)
    edge_colours = torch.tensor(
        [PART_OF_COL, PART_OF_COL, PART_OF_COL, COLOUR_1, COLOUR_1, COLOUR_2, COLOUR_2], dtype=torch.long)
    return CDGraph(col_size=3, delta=5, features=features, edges=edges, edge_colours=edge_colours,
                   node_names=["P", "P0", "P1", "P2"])


class TestDeriveMinimalMatrixFromFact:

    def test_raw_derivation_is_over_determined(self):
        # Sanity check on the fixture itself: both edges really do come back from the raw derivation,
        # and dropping either one alone (not both) would still leave "positive" above threshold.
        model = make_over_determined_model()
        trace = TraceCollector()
        apply_model(make_over_determined_cd_graph(), torch.device("cpu"), model, trace)
        internal_encoder = make_internal_encoder()
        external_encoder = make_external_encoder()

        matrix = derive_matrix_from_fact(
            get_fact(), trace, external_encoder, AdniSignature(internal_encoder), model, THRESHOLD)
        assert matrix.to_dict() == {(0, 1): 1, (0, 2): 2}

        for dropped in [(0, 1), (0, 2)]:
            reduced = SparseTriangularMatrix.empty(d=3, max_value=2)
            for cell, value in matrix.to_dict().items():
                if cell != dropped:
                    reduced = reduced.with_value(*cell, value)
            tree = matrix_to_tree(reduced, AdniSignature(internal_encoder))  # included_nodes=None -> every node_k
            output_graph = apply_model(tree.as_cd_graph, torch.device("cpu"), model)
            assert output_graph.features[0][POSITIVE_POS].item() >= THRESHOLD

    def test_minimises_away_the_weaker_redundant_edge(self):
        model = make_over_determined_model()
        trace = TraceCollector()
        apply_model(make_over_determined_cd_graph(), torch.device("cpu"), model, trace)
        internal_encoder = make_internal_encoder()
        external_encoder = make_external_encoder()

        matrix, included_nodes = derive_minimal_matrix_from_fact(
            get_fact(), trace, external_encoder, AdniSignature(internal_encoder), model, THRESHOLD, torch.device("cpu"))

        # (0, 2) ["2.0", the weaker-heuristic edge] is dropped; (0, 1) ["1.0"] alone still suffices.
        assert matrix.to_dict() == {(0, 1): 1}
        assert included_nodes == {0, 1}  # node_2 (P2) is pruned too: it was never load-bearing

    def test_minimal_rule_still_reproduces_positive(self):
        model = make_over_determined_model()
        trace = TraceCollector()
        apply_model(make_over_determined_cd_graph(), torch.device("cpu"), model, trace)
        internal_encoder = make_internal_encoder()
        external_encoder = make_external_encoder()

        matrix, included_nodes = derive_minimal_matrix_from_fact(
            get_fact(), trace, external_encoder, AdniSignature(internal_encoder), model, THRESHOLD, torch.device("cpu"))

        tree = matrix_to_tree(matrix, AdniSignature(internal_encoder), included_nodes=included_nodes)
        output_graph = apply_model(tree.as_cd_graph, torch.device("cpu"), model)
        assert output_graph.features[0][POSITIVE_POS].item() >= THRESHOLD
