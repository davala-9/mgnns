import pytest
import torch

from src.model.cd_graph import CDGraph, TraceCollector


def make_graph(**overrides):
    defaults = dict(
        col_size=2,
        delta=3,
        features=torch.zeros((2, 3)),
        edges=torch.tensor([[0], [1]], dtype=torch.long),
        edge_colours=torch.tensor([1], dtype=torch.long),
        node_names=["a", "b"],
    )
    defaults.update(overrides)
    return CDGraph(**defaults)


# --- construction ---

def test_valid_graph_constructs():
    g = make_graph()
    assert g.col_size == 2
    assert g.delta == 3
    assert g.node_names == ["a", "b"]

def test_node_names_to_indices_built_correctly():
    g = make_graph(node_names=["x", "y"])
    assert g.node_names_to_indices == {"x": 0, "y": 1}

def test_col_size_must_be_positive():
    with pytest.raises(AssertionError):
        make_graph(col_size=0)

def test_delta_must_be_positive():
    with pytest.raises(AssertionError):
        make_graph(delta=0)

def test_features_row_count_must_match_node_names():
    with pytest.raises(AssertionError):
        make_graph(features=torch.zeros((3, 3)))  # 3 rows, but 2 node names

def test_features_column_count_must_match_delta():
    with pytest.raises(AssertionError):
        make_graph(features=torch.zeros((2, 4)))  # delta=3 expected

def test_edges_must_have_two_rows():
    with pytest.raises(AssertionError):
        make_graph(edges=torch.tensor([[0, 1]], dtype=torch.long),
                   edge_colours=torch.tensor([1, 0], dtype=torch.long))

def test_edges_and_edge_colours_length_must_match():
    with pytest.raises(AssertionError):
        make_graph(edge_colours=torch.tensor([1, 0], dtype=torch.long))

def test_edge_colour_out_of_range_rejected():
    with pytest.raises(AssertionError):
        make_graph(edge_colours=torch.tensor([2], dtype=torch.long))  # col_size=2, valid range is 0-1

def test_negative_edge_colour_rejected():
    with pytest.raises(AssertionError):
        make_graph(edge_colours=torch.tensor([-1], dtype=torch.long))

def test_out_of_range_edge_index_rejected():
    with pytest.raises(AssertionError):
        make_graph(edges=torch.tensor([[0], [5]], dtype=torch.long))  # only 2 node names (indices 0-1)

def test_empty_edges_are_accepted():
    g = make_graph(edges=torch.empty((2, 0), dtype=torch.long),
                   edge_colours=torch.empty((0,), dtype=torch.long))
    assert g.edges.numel() == 0

def test_duplicate_node_names_rejected():
    with pytest.raises(AssertionError):
        make_graph(node_names=["a", "a"])


# --- node_names is defensively copied ---

def test_constructor_copies_node_names():
    names = ["a", "b"]
    g = make_graph(node_names=names)
    names.append("c")
    assert g.node_names == ["a", "b"]


# --- clone ---

def test_clone_produces_equal_graph():
    g = make_graph()
    g2 = g.clone()

    assert g2.col_size == g.col_size
    assert g2.delta == g.delta
    assert torch.equal(g2.features, g.features)
    assert torch.equal(g2.edges, g.edges)
    assert torch.equal(g2.edge_colours, g.edge_colours)
    assert g2.node_names == g.node_names

def test_clone_features_are_independent():
    g = make_graph()
    g2 = g.clone()
    g2.features[0, 0] = 99.0
    assert g.features[0, 0].item() == 0.0

def test_clone_edges_are_independent():
    g = make_graph()
    g2 = g.clone()
    g2.edges[0, 0] = 1
    assert g.edges[0, 0].item() == 0

def test_clone_edge_colours_are_independent():
    g = make_graph()
    g2 = g.clone()
    g2.edge_colours[0] = 0
    assert g.edge_colours[0].item() == 1

def test_clone_node_names_are_independent():
    g = make_graph()
    g2 = g.clone()
    g2.node_names.append("c")
    assert g.node_names == ["a", "b"]


# --- TraceCollector ---

def test_trace_collector_defaults():
    tc = TraceCollector()
    assert tc.cd_graph is None
    assert tc.activations == {}

def test_trace_collector_activations_default_is_independent_per_instance():
    tc1 = TraceCollector()
    tc2 = TraceCollector()
    tc1.activations[0] = torch.zeros(1)
    assert tc2.activations == {}
