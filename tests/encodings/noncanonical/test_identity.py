import os
import tempfile

import torch

from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.identity import IdentityEncoderDecoder
from src.encodings.noncanonical.noncanonical import GroundContext
from src.model.cd_graph import CDGraph
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunctionBuilder
from src.utils.utils import TYPE_PRED
from src.utils.bitset import BitSet

DUMMY_GRAPH = CDGraph(col_size=1, delta=1, features=torch.tensor([[0]]), edges=torch.zeros(2,1),
                       edge_colours=torch.zeros(1, 1), node_names=["dummy"])
EMPTY_DICT = {}

# ------------------------
# Basic functionality tests
# ------------------------

def test_encode_decode_dataset_identity():
    encoder = IdentityEncoderDecoder(
        unary_predicates=["A"],
        binary_predicates=["R"]
    )

    dataset = {("a", "R", "b"), ("a", TYPE_PRED, "A")}

    assert encoder.encode_dataset(dataset) is dataset
    assert encoder.decode_dataset(dataset) is dataset


def test_decode_fact_identity():
    encoder = IdentityEncoderDecoder()
    fact = ("s", "p", "o")

    assert encoder.decode_fact(*fact) == fact


def test_get_canonical_equivalent_identity():
    encoder = IdentityEncoderDecoder()
    fact = ("s", "p", "o")

    assert encoder.get_canonical_equivalent(fact) == fact


def test_init_rejects_overlapping_predicates():
    try:
        IdentityEncoderDecoder(unary_predicates=["A"], binary_predicates=["A"])
        assert False, "expected an AssertionError for overlapping unary/binary predicates"
    except AssertionError:
        pass


# ------------------------
# File loading / saving
# ------------------------

def test_init_from_file():
    content = "A\tA\t1\nR\tR\t2\n"

    with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
        f.write(content)
        file_path = f.name

    encoder = IdentityEncoderDecoder(load_from_document=file_path)

    assert "A" in encoder.canonical_unary_predicates
    assert "R" in encoder.canonical_binary_predicates

    os.remove(file_path)


def test_save_to_file():
    encoder = IdentityEncoderDecoder(
        unary_predicates=["A"],
        binary_predicates=["R"]
    )

    with tempfile.NamedTemporaryFile(delete=False, mode="r+") as f:
        file_path = f.name

    encoder.save_to_file(file_path)

    with open(file_path, "r") as f:
        lines = f.read().strip().splitlines()

    assert "A\tA\t1" in lines
    assert "R\tR\t2" in lines

    os.remove(file_path)


# ------------------------
# unfold_all / unfold_match_ground
# ------------------------
# can_conj trees below are built directly with TreeShapedConjunctionBuilder (var ids are plain
# ints indexing parallel features/children/parent tuples), matching the current TreeShapedConjunction
# API. Edge keys are (level, colour, position) tuples; only the colour (middle element) matters
# to the unfolding logic.

def test_unfold_simple_tree():
    external = IdentityEncoderDecoder(
        unary_predicates=["A", "B"],
        binary_predicates=["R"]
    )
    internal = CanonicalEncoderDecoder(
        unary_predicates=["A", "B"],
        binary_predicates=["R", "S"]
    )
    # internal: unary A=0, B=1 ; binary R=0, S=1

    builder = TreeShapedConjunctionBuilder(n_colours=2)
    var_a = builder.add(features=BitSet.from_subset(2, set()), level=2, parent=-1)                  # root "a": no features
    var_b = builder.add(features=BitSet.from_subset(2, {0}), level=0, parent=var_a, edge=(0, 0, 0))  # colour0=R -> "b": A(b)
    var_c = builder.add(features=BitSet.from_subset(2, {0, 1}), level=1, parent=var_a, edge=(1, 1, 1))  # colour1=S -> "c": A(c), B(c)
    builder.add(features=BitSet.from_subset(2, {1}), level=0, parent=var_c, edge=(0, 0, 1))          # colour0=R -> "d": B(d)
    conj = builder.build()

    ground_data = GroundContext(fact=("a", TYPE_PRED, "A"), graph=DUMMY_GRAPH,
                                canonical_variable_to_constant_index=EMPTY_DICT)  # dummy data, ignored
    data_conj, head = external.unfold_match_ground(can_conj=conj,
                                                    internal_encoder=internal,
                                                    head_predicate="A",
                                                    grounding_context=ground_data)

    # Validate head
    assert head == ("X0", TYPE_PRED, "A")

    # Expected facts:
    # Unfolding happens in a depth-first way, which tells us the order of the variables
    # a->b->c->d
    assert ("X1", "R", "X0") in data_conj
    assert ("X1", TYPE_PRED, "A") in data_conj
    assert ("X2", "S", "X0") in data_conj
    assert ("X2", TYPE_PRED, "A") in data_conj
    assert ("X2", TYPE_PRED, "B") in data_conj
    assert ("X3", "R", "X2") in data_conj
    assert ("X3", TYPE_PRED, "B") in data_conj
    assert len(data_conj) == 7


def test_unfold_empty():
    encoder = IdentityEncoderDecoder()
    internal = CanonicalEncoderDecoder(
        unary_predicates=["A"],
        binary_predicates=["R"]
    )

    builder = TreeShapedConjunctionBuilder(n_colours=1)
    builder.add(features=BitSet.from_subset(1, set()), level=2, parent=-1)  # single node, no features, no children
    conj = builder.build()

    ground_data = GroundContext(fact=("a", "R", "b"), graph=DUMMY_GRAPH,
                                canonical_variable_to_constant_index=EMPTY_DICT)  # dummy data, ignored
    rule, head = encoder.unfold_match_ground(can_conj=conj,
                                              internal_encoder=internal,
                                              head_predicate="R",
                                              grounding_context=ground_data)

    assert rule == []
    assert head == ("X0", TYPE_PRED, "R")
