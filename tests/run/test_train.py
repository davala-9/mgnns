import torch

from src.encodings.canonical import CanonicalEncoderDecoder
from src.run.train import build_training_labels
from src.utils.utils import TYPE_PRED


def make_encoder_and_graph():
    encoder = CanonicalEncoderDecoder(unary_predicates=["A", "B"], binary_predicates=["R"])
    cd_graph = encoder.encode_dataset({("a", "R", "b"), ("b", TYPE_PRED, "A")})
    return encoder, cd_graph


def test_build_training_labels_marks_each_example():
    encoder, cd_graph = make_encoder_and_graph()
    labels = build_training_labels(cd_graph, encoder, [("a", TYPE_PRED, "B"), ("b", TYPE_PRED, "A")])
    expected = torch.zeros_like(cd_graph.features)
    expected[cd_graph.node_names_to_indices["a"], encoder.unary_pred_position_dict["B"]] = 1
    expected[cd_graph.node_names_to_indices["b"], encoder.unary_pred_position_dict["A"]] = 1
    assert torch.equal(labels, expected)


def test_build_training_labels_drops_examples_about_unseen_constants_and_binary_examples():
    encoder, cd_graph = make_encoder_and_graph()
    labels = build_training_labels(cd_graph, encoder, [("new", TYPE_PRED, "A"), ("a", "R", "b")])
    assert not labels.any()
