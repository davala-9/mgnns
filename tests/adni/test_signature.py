import pytest
import torch

from src.adni.signature import AdniSignature, colour_predicate_for, node_predicate_for
from src.encodings.canonical import CanonicalEncoderDecoder


def make_internal_encoder(d, max_value, extra_unary=(), extra_binary=()):
    unary = ["node"] + [node_predicate_for(k) for k in range(d)] + ["positive", *extra_unary]
    binary = ["part_of"] + [colour_predicate_for(v) for v in range(1, max_value + 1)] + list(extra_binary)
    return CanonicalEncoderDecoder(unary_predicates=unary, binary_predicates=binary)


def test_recovers_d_and_max_value():
    signature = AdniSignature(make_internal_encoder(d=4, max_value=3))
    assert (signature.d, signature.max_value) == (4, 3)


def test_ignores_extra_unrelated_predicates():
    encoder = make_internal_encoder(d=2, max_value=1, extra_unary=["some_other_predicate"],
                                    extra_binary=["some_other_colour"])
    signature = AdniSignature(encoder)
    assert (signature.d, signature.max_value) == (2, 1)


def test_positions_and_colours_match_the_encoder():
    encoder = make_internal_encoder(d=2, max_value=2)
    signature = AdniSignature(encoder)
    unary, binary = encoder.unary_pred_position_dict, encoder.binary_pred_colour_dict
    assert signature.node_pos == unary["node"]
    assert signature.positive_pos == unary["positive"]
    assert signature.part_of_colour == binary["part_of"]
    assert [signature.node_k_pos(k) for k in range(2)] == [unary["node_0"], unary["node_1"]]
    assert [signature.value_colour(v) for v in (1, 2)] == [binary["1.0"], binary["2.0"]]


@pytest.mark.parametrize("unary,binary", [
    (["node_0", "positive"], ["part_of", "1.0"]),              # no "node"
    (["node", "node_0"], ["part_of", "1.0"]),                  # no "positive"
    (["node", "node_0", "positive"], ["1.0"]),                 # no "part_of"
    (["node", "positive"], ["part_of", "1.0"]),                # no node_k at all
    (["node", "node_0", "positive"], ["part_of"]),             # no colour at all
    (["node", "node_0", "node_2", "positive"], ["part_of", "1.0"]),  # gap: node_1 missing
    (["node", "node_1", "positive"], ["part_of", "1.0"]),      # doesn't start at node_0
    (["node", "node_0", "positive"], ["part_of", "1.0", "3.0"]),     # gap: 2.0 missing
    (["node", "node_0", "positive"], ["part_of", "0.0", "1.0"]),     # colour values start at 1, not 0
])
def test_rejects_an_encoder_that_does_not_declare_an_adni_signature(unary, binary):
    encoder = CanonicalEncoderDecoder(unary_predicates=unary, binary_predicates=binary)
    with pytest.raises(ValueError, match="does not declare an ADNI signature"):
        AdniSignature(encoder)


def test_node_index_of_reads_off_the_node_k_bit():
    encoder = make_internal_encoder(d=3, max_value=1)
    signature = AdniSignature(encoder)
    features = torch.zeros(encoder.get_n_unary_predicates())
    features[encoder.unary_pred_position_dict["node"]] = 1.0
    features[encoder.unary_pred_position_dict["node_2"]] = 1.0
    assert signature.node_index_of(features) == 2


def test_node_index_of_raises_when_no_node_k_bit_is_set():
    encoder = make_internal_encoder(d=3, max_value=1)
    signature = AdniSignature(encoder)
    features = torch.zeros(encoder.get_n_unary_predicates())
    features[encoder.unary_pred_position_dict["node"]] = 1.0  # "node" but no node_k
    with pytest.raises(ValueError):
        signature.node_index_of(features)
