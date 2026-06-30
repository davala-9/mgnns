import os
import tempfile

from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.identity import IdentityEncoderDecoder
from src.rule_extraction.tree_shaped_conjunction import Variable, TreeShapedConjunction
from src.utils.utils import TYPE_PRED
from src.utils.bitset import BitSet


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


def test_unfold_simple_tree():
    external = IdentityEncoderDecoder(
        unary_predicates=["A", "B"],
        binary_predicates=["R"]
    )
    internal = CanonicalEncoderDecoder(
        unary_predicates=["A", "B"],
        binary_predicates=["R","S"]
    )

    # Simple treelike conjunction with 4 variables
    feature_mask_a = BitSet.from_subset(dimension=2,subset=set())
    feature_mask_b = BitSet.from_subset(dimension=2,subset={0})
    feature_mask_c = BitSet.from_subset(dimension=2,subset={0,1})
    feature_mask_d = BitSet.from_subset(dimension=2,subset={1})
    variable_a = Variable(feature_mask_a,level=2)
    variable_b = Variable(feature_mask_b,level=0)
    variable_c = Variable(feature_mask_c,level=1)
    variable_d = Variable(feature_mask_d,level=0)
    variable_a.children[(0,0,0)] = variable_b
    variable_a.children[(1,1,1)] = variable_c
    variable_c.children[(0,0,1)] = variable_d
    conj = TreeShapedConjunction(variable_a,2)

    data_conj, root_vars = external.unfold(conj, head_is_binary=False, internal_encoder=internal)

    # Validate root variable
    assert root_vars == ["X0"]

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


def test_unfold_empty():
    encoder = IdentityEncoderDecoder()
    internal = CanonicalEncoderDecoder(
        unary_predicates=["A"],
        binary_predicates=["R"]
    )
    feature_mask_a = BitSet.from_subset(dimension=1,subset=set())
    variable_a = Variable(feature_mask_a,2)
    conj = TreeShapedConjunction(variable_a,1)
    rule, head_vars = encoder.unfold(conj, internal_encoder=internal)

    assert rule == []