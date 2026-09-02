import os
import tempfile
import torch
import pytest

from src.utils.utils import TYPE_PRED
from src.model.cd_graph import CDGraph
from src.encodings.canonical import CanonicalEncoderDecoder


def test_init_with_predicates():
    enc = CanonicalEncoderDecoder(
        unary_predicates=["A", "B"],
        binary_predicates=["R"]
    )

    assert enc.get_n_unary_predicates() == 2
    assert enc.get_n_binary_predicates() == 1

    assert enc.get_unary_predicate_for_position(0) == "A"
    assert enc.get_binary_predicate_for_colour(0) == "R"


def test_init_with_empty_predicates_adds_dummy():
    enc = CanonicalEncoderDecoder(
        unary_predicates=[],
        binary_predicates=[]
    )

    assert enc.get_n_unary_predicates() == 1
    assert enc.get_n_binary_predicates() == 1

    assert enc.get_unary_predicate_for_position(0) == CanonicalEncoderDecoder.DUMMY_PRED
    assert enc.get_binary_predicate_for_colour(0) == CanonicalEncoderDecoder.DUMMY_COL


def test_save_and_load(tmp_path):
    file_path = tmp_path / "encoder.txt"

    enc1 = CanonicalEncoderDecoder(
        unary_predicates=["A"],
        binary_predicates=["R"]
    )
    enc1.save_to_file(file_path)

    enc2 = CanonicalEncoderDecoder(load_from_document=file_path)

    assert enc2.get_n_unary_predicates() == 1
    assert enc2.get_n_binary_predicates() == 1
    assert enc2.get_unary_predicate_for_position(0) == "A"
    assert enc2.get_binary_predicate_for_colour(0) == "R"


def test_encode_dataset_basic():
    enc = CanonicalEncoderDecoder(
        unary_predicates=["A"],
        binary_predicates=["R"]
    )

    dataset = [
        ("a", TYPE_PRED, "A"),   # unary fact
        ("a", "R", "b"),         # binary fact
    ]

    cd_graph = enc.encode_dataset(dataset)

    # Check graph type
    assert isinstance(cd_graph, CDGraph)

    # Check node names
    assert set(cd_graph.node_names) == {"a", "b"}

    # Check feature dimension
    assert cd_graph.features.shape[1] == enc.get_n_unary_predicates()

    # Check edges exist
    assert cd_graph.edges.shape[0] == 2  # (2, num_edges)


def test_encode_dataset_unknown_unary_raises():
    enc = CanonicalEncoderDecoder(
        unary_predicates=["A"],
        binary_predicates=["R"]
    )

    dataset = [("a", TYPE_PRED, "UNKNOWN")]

    with pytest.raises(ValueError):
        enc.encode_dataset(dataset)


def test_encode_dataset_unknown_binary_raises():
    enc = CanonicalEncoderDecoder(
        unary_predicates=["A"],
        binary_predicates=["R"]
    )

    dataset = [("a", "UNKNOWN", "b")]

    with pytest.raises(ValueError):
        enc.encode_dataset(dataset)


def test_decode_graph():
    enc = CanonicalEncoderDecoder(
        unary_predicates=["A","B"],
        binary_predicates=["R"]
    )

    # Create a simple CDGraph manually with no edges
    features = torch.tensor([[0.9, 0.1],[0.1,0.4]], dtype=torch.float32)
    edges = torch.zeros((2, 0), dtype=torch.long)
    edge_colours = torch.zeros((0,), dtype=torch.long)

    cd_graph = CDGraph(
        col_size=1,
        delta=2,
        features=features,
        edges=edges,
        edge_colours=edge_colours,
        node_names=["a", "b"]
    )

    decoded = enc.decode_graph(cd_graph, threshold=0.5)

    assert len(decoded) == 1
    fact = ("a", TYPE_PRED, "A")
    assert fact in decoded
    assert decoded[fact] == pytest.approx(0.9, rel=1e-5)