import pytest
import torch

from src.adni.matrix_to_tree import infer_dimensions, is_sound, matrix_to_tree, node_predicate_for, \
    colour_predicate_for
from src.adni.sparse_triangular_matrix import SparseTriangularMatrix
from src.encodings.canonical import CanonicalEncoderDecoder
from src.model.gnn_architectures import GNN


def make_internal_encoder(d, max_value):
    unary = ["node"] + [node_predicate_for(k) for k in range(d)]
    binary = ["part_of"] + [colour_predicate_for(v) for v in range(1, max_value + 1)]
    return CanonicalEncoderDecoder(unary_predicates=unary, binary_predicates=binary)


def test_infer_dimensions_recovers_d_and_max_value():
    encoder = make_internal_encoder(d=4, max_value=3)
    assert infer_dimensions(encoder) == (4, 3)


def test_infer_dimensions_on_an_encoder_with_extra_unrelated_predicates():
    unary = ["node", node_predicate_for(0), node_predicate_for(1), "positive", "some_other_predicate"]
    binary = ["part_of", colour_predicate_for(1), "some_other_colour"]
    encoder = CanonicalEncoderDecoder(unary_predicates=unary, binary_predicates=binary)
    assert infer_dimensions(encoder) == (2, 1)


def test_root_has_no_unary_predicate():
    encoder = make_internal_encoder(d=3, max_value=2)
    tree = matrix_to_tree(SparseTriangularMatrix.empty(d=3, max_value=2), encoder)
    assert tree.features[0].elements() == []

def test_every_node_is_a_child_of_root_via_part_of_and_has_node_and_node_k():
    encoder = make_internal_encoder(d=3, max_value=2)
    tree = matrix_to_tree(SparseTriangularMatrix.empty(d=3, max_value=2), encoder)
    part_of_colour = encoder.binary_pred_colour_dict["part_of"]
    for k in range(3):
        var_id = k + 1  # root is 0, nodes are added in order right after it
        assert tree.parent[var_id] == 0
        assert tree.parent_edge[var_id][1] == part_of_colour
        expected_positions = {
            encoder.unary_pred_position_dict["node"],
            encoder.unary_pred_position_dict[f"node_{k}"],
        }
        assert set(tree.features[var_id].elements()) == expected_positions

def test_matrix_entry_wires_an_edge_from_node_j_to_node_i():
    # (i, j) = (0, 2), value 1 -> fact (node_2, "1.0", node_0): node_2 is node_0's "child" in the
    # tree's children dict (see matrix_to_tree's docstring on the child/parent <-> subject/object convention).
    matrix = SparseTriangularMatrix.empty(d=3, max_value=2).with_value(0, 2, 1)
    encoder = make_internal_encoder(d=3, max_value=2)
    tree = matrix_to_tree(matrix, encoder)
    node_0_id, node_2_id = 1, 3
    value_1_colour = encoder.binary_pred_colour_dict["1.0"]
    assert tree.children[node_0_id] == {(0, value_1_colour, 2): node_2_id}

def test_diagonal_entry_is_a_self_loop():
    matrix = SparseTriangularMatrix.empty(d=3, max_value=2).with_value(1, 1, 2)
    encoder = make_internal_encoder(d=3, max_value=2)
    tree = matrix_to_tree(matrix, encoder)
    node_1_id = 2
    value_2_colour = encoder.binary_pred_colour_dict["2.0"]
    assert tree.children[node_1_id] == {(0, value_2_colour, 1): node_1_id}

def test_as_cd_graph_encodes_child_to_parent_edges():
    matrix = SparseTriangularMatrix.empty(d=2, max_value=1).with_value(0, 1, 1)
    encoder = make_internal_encoder(d=2, max_value=1)
    tree = matrix_to_tree(matrix, encoder)
    cd_graph = tree.as_cd_graph
    part_of_colour = encoder.binary_pred_colour_dict["part_of"]
    value_1_colour = encoder.binary_pred_colour_dict["1.0"]
    node_0_id, node_1_id = 1, 2
    edges = set(zip(cd_graph.edges[0].tolist(), cd_graph.edges[1].tolist(), cd_graph.edge_colours.tolist()))
    # (node_0, part_of, root), (node_1, part_of, root), (node_1, "1.0", node_0)
    assert edges == {
        (node_0_id, 0, part_of_colour),
        (node_1_id, 0, part_of_colour),
        (node_1_id, node_0_id, value_1_colour),
    }

def test_rejects_encoder_missing_a_required_unary_predicate():
    encoder = CanonicalEncoderDecoder(unary_predicates=["node", "node_0"], binary_predicates=["part_of", "1.0"])
    matrix = SparseTriangularMatrix.empty(d=2, max_value=1)  # needs node_0 AND node_1
    with pytest.raises(ValueError):
        matrix_to_tree(matrix, encoder)

def test_rejects_encoder_missing_a_required_binary_predicate():
    encoder = CanonicalEncoderDecoder(unary_predicates=["node", "node_0", "node_1"], binary_predicates=["part_of"])
    matrix = SparseTriangularMatrix.empty(d=2, max_value=1)  # needs "1.0" too
    with pytest.raises(ValueError):
        matrix_to_tree(matrix, encoder)

def test_empty_matrix_has_no_extra_edges_beyond_part_of():
    encoder = make_internal_encoder(d=3, max_value=2)
    tree = matrix_to_tree(SparseTriangularMatrix.empty(d=3, max_value=2), encoder)
    assert tree.children[1] == {}
    assert tree.children[2] == {}
    assert tree.children[3] == {}

def test_included_nodes_excludes_the_others_entirely():
    matrix = SparseTriangularMatrix.empty(d=3, max_value=2).with_value(0, 2, 1)
    encoder = make_internal_encoder(d=3, max_value=2)
    tree = matrix_to_tree(matrix, encoder, included_nodes={0, 2})
    # root (0) plus only node_0 and node_2 -- node_1 is left out entirely.
    assert len(tree) == 3
    node_0_id, node_2_id = 1, 2
    part_of_colour = encoder.binary_pred_colour_dict["part_of"]
    value_1_colour = encoder.binary_pred_colour_dict["1.0"]
    assert tree.parent[node_0_id] == 0 and tree.parent[node_2_id] == 0
    assert tree.children[0] == {(1, part_of_colour, 0): node_0_id, (1, part_of_colour, 2): node_2_id}
    assert tree.children[node_0_id] == {(0, value_1_colour, 2): node_2_id}
    assert set(tree.features[node_0_id].elements()) == {
        encoder.unary_pred_position_dict["node"], encoder.unary_pred_position_dict["node_0"]}

def test_included_nodes_drops_matrix_entries_touching_an_excluded_node():
    matrix = SparseTriangularMatrix.empty(d=3, max_value=2).with_value(0, 1, 1)
    encoder = make_internal_encoder(d=3, max_value=2)
    tree = matrix_to_tree(matrix, encoder, included_nodes={0, 2})  # excludes node_1, the entry's target
    node_0_id = 1
    assert tree.children[node_0_id] == {}


# --- is_sound ---

# d=2, max_value=1, plus "positive" at unary position 3. The model derives "positive" at the root iff the root
# has a part_of child with node_0: hidden[0] = node_0 feature, positive = sigmoid(20 * max part_of hidden[0] - 10).
def make_node_0_detector(encoder):
    model = GNN(feature_dimension=4, num_edge_colours=2, aggregation_1="max", aggregation_2="max")
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.lin_self_1.weight[0, encoder.unary_pred_position_dict[node_predicate_for(0)]] = 1.0
        model.conv2.weights[encoder.binary_pred_colour_dict["part_of"], 3, 0] = 20.0
    return model


def make_encoder_with_positive():
    unary = ["node", node_predicate_for(0), node_predicate_for(1), "positive"]
    binary = ["part_of", colour_predicate_for(1)]
    return CanonicalEncoderDecoder(unary_predicates=unary, binary_predicates=binary)


def test_is_sound_includes_every_node_by_default():
    encoder = make_encoder_with_positive()
    model = make_node_0_detector(encoder)
    matrix = SparseTriangularMatrix.empty(2, 1)
    assert is_sound(matrix, encoder, model, torch.device("cpu"), position=3, threshold=0.5)


def test_is_sound_respects_included_nodes():
    encoder = make_encoder_with_positive()
    model = make_node_0_detector(encoder)
    matrix = SparseTriangularMatrix.empty(2, 1)
    device = torch.device("cpu")
    assert is_sound(matrix, encoder, model, device, position=3, threshold=0.5, included_nodes={0})
    assert not is_sound(matrix, encoder, model, device, position=3, threshold=0.5, included_nodes={1})


@pytest.mark.parametrize("score, expected", [(0.25, False), (0.5, False), (0.75, True)])
def test_is_sound_requires_a_score_strictly_above_the_threshold(monkeypatch, score, expected):
    from src.adni import matrix_to_tree as matrix_to_tree_module
    encoder = make_encoder_with_positive()

    def fake_apply_model(input_graph, device, model):
        output = input_graph.clone()
        output.features = torch.full_like(output.features, score)
        return output
    monkeypatch.setattr(matrix_to_tree_module, "apply_model", fake_apply_model)

    matrix = SparseTriangularMatrix.empty(2, 1)
    assert is_sound(matrix, encoder, None, torch.device("cpu"), position=3, threshold=0.5) == expected
