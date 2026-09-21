from src.adni.matrix_to_tree import colour_predicate_for, matrix_to_tree, node_predicate_for
from src.adni.rule_printing import tree_to_rule
from src.adni.sparse_triangular_matrix import SparseTriangularMatrix
from src.encodings.canonical import CanonicalEncoderDecoder


def make_internal_encoder(d, max_value):
    unary = ["node"] + [node_predicate_for(k) for k in range(d)] + ["positive"]
    binary = ["part_of"] + [colour_predicate_for(v) for v in range(1, max_value + 1)]
    return CanonicalEncoderDecoder(unary_predicates=unary, binary_predicates=binary)


def test_empty_matrix_prints_only_the_part_of_and_node_atoms():
    encoder = make_internal_encoder(d=2, max_value=1)
    tree = matrix_to_tree(SparseTriangularMatrix.empty(d=2, max_value=1), encoder)
    rule = tree_to_rule(tree, encoder, "positive")
    assert rule == (
        "<positive>[?X0] :- <part_of>[?X1,?X0], <part_of>[?X2,?X0], "
        "<node>[?X1], <node_0>[?X1], <node>[?X2], <node_1>[?X2] ."
    )


def test_off_diagonal_edge_keeps_the_same_variable_for_the_shared_node():
    # This is exactly the case that broke identity.IdentityEncoderDecoder.unfold_all: node_1 (X2) is
    # both a part_of child of root AND the target of node_0's colour edge -- it must stay X2 throughout,
    # not fork into a second, disconnected variable.
    encoder = make_internal_encoder(d=2, max_value=1)
    matrix = SparseTriangularMatrix.empty(d=2, max_value=1).with_value(0, 1, 1)
    tree = matrix_to_tree(matrix, encoder)
    rule = tree_to_rule(tree, encoder, "positive")
    assert rule == (
        "<positive>[?X0] :- <part_of>[?X1,?X0], <part_of>[?X2,?X0], "
        "<node>[?X1], <node_0>[?X1], <1.0>[?X2,?X1], <node>[?X2], <node_1>[?X2] ."
    )
    assert rule.count("X2") == 4  # part_of child, edge target, and its two own feature atoms


def test_diagonal_self_loop_uses_one_variable_on_both_sides_of_the_edge():
    encoder = make_internal_encoder(d=1, max_value=1)
    matrix = SparseTriangularMatrix.empty(d=1, max_value=1).with_value(0, 0, 1)
    tree = matrix_to_tree(matrix, encoder)
    rule = tree_to_rule(tree, encoder, "positive")
    assert rule == (
        "<positive>[?X0] :- <part_of>[?X1,?X0], <node>[?X1], <node_0>[?X1], <1.0>[?X1,?X1] ."
    )
