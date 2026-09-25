import torch

from src.adni.matrix_search import greedy_climb, minimise, value_order_table
from src.adni.signature import AdniSignature, colour_predicate_for, node_predicate_for
from src.adni.sparse_triangular_matrix import SparseTriangularMatrix
from src.encodings.canonical import CanonicalEncoderDecoder
from src.model.gnn_architectures import GNN

# d=2 nodes, values in {1, 2}: unary "node", "node_0", "node_1", "positive" (positions 0..3),
# binary "part_of", "1.0", "2.0" (colours 0, 1, 2).
D = 2
MAX_VALUE = 2


def make_internal_encoder():
    unary = ["node", node_predicate_for(0), node_predicate_for(1), "positive"]
    binary = ["part_of", colour_predicate_for(1), colour_predicate_for(2)]
    return CanonicalEncoderDecoder(unary_predicates=unary, binary_predicates=binary)


def make_model():
    # dim_0 = 4 (unary predicates), dim_1 = 8, dim_2 = 4.
    model = GNN(feature_dimension=4, num_edge_colours=3, aggregation_1="max", aggregation_2="max")
    # matrix_A at layer 1: row 0 only, so node_i's contribution is bigger for i=1 (node_1) than i=0 (node_0).
    a1 = torch.zeros(8, 4)
    a1[0, 0] = 1.0  # weight on "node"
    a1[0, 1] = 10.0  # weight on "node_0" (used when i=0)
    a1[0, 2] = 20.0  # weight on "node_1" (used when i=1)
    model.lin_self_1.weight.data = a1
    # colour "1.0": small weights -- always the worse candidate value.
    b1_colour1 = torch.zeros(8, 4)
    b1_colour1[0, 0] = 2.0  # weight on "node"
    b1_colour1[0, 1] = 100.0  # weight on "node_0" (used when j=0)
    b1_colour1[0, 2] = 200.0  # weight on "node_1" (used when j=1)
    model.conv1.weights.data[1] = b1_colour1
    # colour "2.0": bigger weights -- always the better candidate value.
    b1_colour2 = torch.zeros(8, 4)
    b1_colour2[0, 0] = 3.0  # weight on "node"
    b1_colour2[0, 1] = 300.0  # weight on "node_0" (used when j=0)
    b1_colour2[0, 2] = 400.0  # weight on "node_1" (used when j=1)
    model.conv1.weights.data[2] = b1_colour2
    model.conv1.weights.data[0] = torch.zeros(8, 4)
    # colour "part_of": "positive" (row 3) reads hidden position 0 positively, position 1 negatively
    # (must be excluded by heuristic_for_element's >0 relevance filter).
    b2_part_of = torch.zeros(4, 8)
    b2_part_of[3, 0] = 7.0
    b2_part_of[3, 1] = -100.0
    model.conv2.weights.data[0] = b2_part_of
    model.conv2.weights.data[1] = torch.zeros(4, 8)
    model.conv2.weights.data[2] = torch.zeros(4, 8)
    return model


def test_value_order_table_ranks_colour_2_above_colour_1_everywhere():
    encoder = make_internal_encoder()
    model = make_model()
    table = value_order_table(AdniSignature(encoder), model)
    for i in range(D):
        for j in range(i, D):
            assert table[i][j] == [2, 1]


# Hand-computed heuristics for this fixture (heuristic = 7 * (matrix_A's node/node_i contribution +
# matrix_B(value)'s node/node_j contribution)) rank every cell's best (value 2) candidate as:
#   (1, 1, 2) = 2968 > (0, 1, 2) = 2898 > (0, 0, 2) = 2198
# so a greedy climb from the empty matrix always sets (1, 1) -> 2 first, then (0, 1) -> 2, then (0, 0) -> 2.


def test_greedy_climb_stops_as_soon_as_the_first_move_is_sound():
    encoder = make_internal_encoder()
    model = make_model()
    table = value_order_table(AdniSignature(encoder), model)
    matrix = SparseTriangularMatrix.empty(d=D, max_value=MAX_VALUE)
    result = greedy_climb(matrix, table, AdniSignature(encoder), model, check_soundness=lambda m: m.get(1, 1) == 2)
    assert result.to_dict() == {(1, 1): 2}


def test_greedy_climb_takes_the_second_best_move_when_the_first_is_not_enough():
    encoder = make_internal_encoder()
    model = make_model()
    table = value_order_table(AdniSignature(encoder), model)
    matrix = SparseTriangularMatrix.empty(d=D, max_value=MAX_VALUE)
    result = greedy_climb(
        matrix, table, AdniSignature(encoder), model,
        check_soundness=lambda m: m.get(1, 1) == 2 and m.get(0, 1) == 2,
    )
    assert result.to_dict() == {(1, 1): 2, (0, 1): 2}


def test_greedy_climb_returns_none_when_it_runs_out_of_moves():
    encoder = make_internal_encoder()
    model = make_model()
    table = value_order_table(AdniSignature(encoder), model)
    matrix = SparseTriangularMatrix.empty(d=D, max_value=MAX_VALUE)
    result = greedy_climb(matrix, table, AdniSignature(encoder), model, check_soundness=lambda m: False)
    assert result is None


# Same fixture: heuristics for value 2 (the only value minimise ever needs to look up, since these
# tests always start from the dense matrix) rank (0, 0) weakest, then (0, 1), then (1, 1) strongest --
# so minimise always tries removing (0, 0) first, then (0, 1), then (1, 1).


def _dense_matrix():
    return (
        SparseTriangularMatrix.empty(d=D, max_value=MAX_VALUE)
        .with_value(0, 0, 2).with_value(0, 1, 2).with_value(1, 1, 2)
    )


def test_minimise_drops_every_cell_not_needed_for_the_one_that_is():
    encoder = make_internal_encoder()
    model = make_model()
    result = minimise(_dense_matrix(), AdniSignature(encoder), model, check_soundness=lambda m: m.get(1, 1) == 2)
    assert result.to_dict() == {(1, 1): 2}


def test_minimise_keeps_every_cell_that_is_needed():
    encoder = make_internal_encoder()
    model = make_model()
    result = minimise(
        _dense_matrix(), AdniSignature(encoder), model,
        check_soundness=lambda m: m.get(0, 1) == 2 and m.get(1, 1) == 2,
    )
    assert result.to_dict() == {(0, 1): 2, (1, 1): 2}


def test_minimise_keeps_the_weakest_cell_when_it_is_the_one_actually_needed():
    encoder = make_internal_encoder()
    model = make_model()
    result = minimise(_dense_matrix(), AdniSignature(encoder), model, check_soundness=lambda m: m.get(0, 0) == 2)
    assert result.to_dict() == {(0, 0): 2}
