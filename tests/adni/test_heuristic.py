import torch

from src.adni.heuristic import heuristic_for_element, heuristic_for_matrix
from src.adni.matrix_to_tree import colour_predicate_for, node_predicate_for
from src.adni.sparse_triangular_matrix import SparseTriangularMatrix
from src.encodings.canonical import CanonicalEncoderDecoder
from src.model.gnn_architectures import GNN

# d=2 nodes, values in {0, 1}: unary "node", "node_0", "node_1", "positive" (positions 0..3),
# binary "part_of", "1.0" (colours 0, 1).
D = 2
MAX_VALUE = 1


def make_internal_encoder():
    unary = ["node", node_predicate_for(0), node_predicate_for(1), "positive"]
    binary = ["part_of", colour_predicate_for(1)]
    return CanonicalEncoderDecoder(unary_predicates=unary, binary_predicates=binary)


def make_model():
    # dim_0 = 4 (unary predicates), dim_1 = 8, dim_2 = 4.
    model = GNN(feature_dimension=4, num_edge_colours=2, aggregation_1="max", aggregation_2="max")
    # colour "1.0" (colour 1) at layer 1: only hidden position 0 reads "node" (col 0) and "node_1" (col 2).
    b1_colour1 = torch.zeros(8, 4)
    b1_colour1[0, 0] = 2.0  # weight on "node"
    b1_colour1[0, 2] = 3.0  # weight on "node_1"
    b1_colour1[1, 1] = 5.0  # weight on "node_0" -- irrelevant to node_1, must not affect the score
    model.conv1.weights.data[1] = b1_colour1
    model.conv1.weights.data[0] = torch.zeros(8, 4)
    # matrix_A at layer 1: hidden position 0 also reads "node" and either "node_0" or "node_1"
    # (self-update, so this is the i side, as opposed to conv1's j side above).
    a1 = torch.zeros(8, 4)
    a1[0, 0] = 1.0  # weight on "node"
    a1[0, 1] = 4.0  # weight on "node_0"
    a1[0, 2] = 9.0  # weight on "node_1"
    model.lin_self_1.weight.data = a1
    # colour "part_of" (colour 0) at layer 2: "positive" (row 3) reads hidden position 0 positively
    # and hidden position 1 negatively (must be excluded by the >0 relevance filter).
    b2_part_of = torch.zeros(4, 8)
    b2_part_of[3, 0] = 7.0
    b2_part_of[3, 1] = -100.0
    model.conv2.weights.data[0] = b2_part_of
    model.conv2.weights.data[1] = torch.zeros(4, 8)
    return model


def test_heuristic_for_element_matches_hand_computation():
    encoder = make_internal_encoder()
    model = make_model()
    # relevant hidden position is only 0 (part_of_row[0] = 7 > 0; part_of_row[1] = -100 is excluded).
    # contribution at position 0 = B1[0, node] + B1[0, node_1] + A1[0, node] + A1[0, node_0]
    #                            = 2 + 3 + 1 + 4 = 10. heuristic = 7 * 10 = 70.
    assert heuristic_for_element(i=0, j=1, value=1, internal_encoder=encoder, model=model) == 70.0


def test_heuristic_for_element_depends_on_i():
    encoder = make_internal_encoder()
    model = make_model()
    # (0, 1) = 1 uses node_0 for i; the diagonal (1, 1) = 1 uses node_1 for both i and j, so it should
    # score differently even though j and value are the same in both cells.
    matrix = SparseTriangularMatrix.empty(d=D, max_value=MAX_VALUE).with_value(0, 1, 1).with_value(1, 1, 1)
    scores = heuristic_for_matrix(matrix, encoder, model)
    assert scores[(0, 1)] == 70.0
    # contribution at position 0 = B1[0, node] + B1[0, node_1] + A1[0, node] + A1[0, node_1]
    #                            = 2 + 3 + 1 + 9 = 15. heuristic = 7 * 15 = 105.
    assert scores[(1, 1)] == 105.0


def test_heuristic_for_matrix_only_scores_present_cells():
    encoder = make_internal_encoder()
    model = make_model()
    matrix = SparseTriangularMatrix.empty(d=D, max_value=MAX_VALUE).with_value(0, 1, 1)
    scores = heuristic_for_matrix(matrix, encoder, model)
    assert set(scores) == {(0, 1)}
