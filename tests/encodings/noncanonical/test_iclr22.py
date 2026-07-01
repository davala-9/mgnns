import os
import tempfile
import torch
import pytest

from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.iclr22 import ICLREncoderDecoder
from src.model.cd_graph import CDGraph
from src.rule_extraction.fact_explanation import FactExplainer
from src.rule_extraction.tree_shaped_conjunction import Variable, TreeShapedConjunction
from src.utils.utils import TYPE_PRED
from src.utils.bitset import BitSet

@pytest.fixture
def encoder():
    return ICLREncoderDecoder(
        unary_predicates=["A"],
        binary_predicates=["R"]
    )

@pytest.fixture
def sample_dataset():
    return {("a", TYPE_PRED, "A"),("a","R","b")}


def test_initialisation(encoder):

    # Test internal fields
    assert encoder.input_predicate_to_unary_canonical_dict["A"] == "A"
    assert encoder.input_predicate_to_unary_canonical_dict.inverse["A"] == "A"
    assert encoder.input_predicate_to_unary_canonical_dict["R"] == "unary-for-R"
    assert encoder.input_predicate_to_unary_canonical_dict.inverse["unary-for-R"] == "R"
    assert encoder.input_predicate_to_arity["A"] == 1
    assert encoder.input_predicate_to_arity["R"] == 2
    assert encoder.canonical_unary_predicates == ["A","unary-for-R"]
    assert encoder.canonical_binary_predicates == [encoder.col1, encoder.col2, encoder.col3, encoder.col4]

def test_encode_dataset(encoder,sample_dataset):
    cd_dataset = encoder.encode_dataset(sample_dataset)

    assert encoder.pair_term_dict[("a","b")] == "term-for-a-b"
    assert encoder.pair_term_dict.inverse["term-for-a-b"] == ("a","b")

    # Test generated cd_dataset
    assert ("a", TYPE_PRED, "A") in cd_dataset
    assert ("term-for-a-b", TYPE_PRED, "unary-for-R") in cd_dataset
    # Col 1 facts
    assert ("a", encoder.col1, "term-for-a-b") in cd_dataset
    assert ("term-for-a-b", encoder.col1, "a") in cd_dataset
    assert ("term-for-b-a", encoder.col1, "b") in cd_dataset
    assert ("b", encoder.col1, "term-for-b-a") in cd_dataset
    # Col 2 facts
    assert ("b", encoder.col2, "term-for-a-b") in cd_dataset
    assert ("term-for-a-b", encoder.col2, "b") in cd_dataset
    assert ("term-for-b-a", encoder.col2, "a") in cd_dataset
    assert ("a", encoder.col2, "term-for-b-a") in cd_dataset
    # Col 3 facts
    assert ("term-for-a-b", encoder.col3, "term-for-b-a") in cd_dataset
    assert ("term-for-b-a", encoder.col3, "term-for-a-b") in cd_dataset
    # Col4 facts
    assert ("a", encoder.col4, "b") in cd_dataset


def test_decode_binary_fact(encoder):
    with pytest.raises(AssertionError):
        encoder.decode_fact("a",encoder.col1, "b")

def test_decode_unary_facts(encoder,sample_dataset):
    # TODO: consider making "term-for" and "unary-for" and dashes into protected strings
    encoder.encode_dataset(sample_dataset) # Necessary to create the relevant terms
    assert ("a",TYPE_PRED,"A") == encoder.decode_fact("a",TYPE_PRED,"A")
    assert ("a","R","b") == encoder.decode_fact("term-for-a-b",TYPE_PRED,"unary-for-R")

def test_decoder_dataset(encoder,sample_dataset):
    encoder.encode_dataset(sample_dataset) # Necessary to create the relevant terms
    cd_dataset = {("a",TYPE_PRED,"A"),("term-for-a-b",TYPE_PRED,"unary-for-R")}
    assert encoder.decode_dataset(cd_dataset) == {("a",TYPE_PRED,"A"),("a","R","b")}

def test_init_from_file():
    content = "A\tA\t1\nR\tunary-for-R\t2\n"

    with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
        f.write(content)
        file_path = f.name

    encoder = ICLREncoderDecoder(load_from_document=file_path)

    assert "A" in encoder.canonical_unary_predicates
    assert "unary-for-R" in encoder.canonical_unary_predicates

    os.remove(file_path)


def test_save_to_file(encoder):

    with tempfile.NamedTemporaryFile(delete=False, mode="r+") as f:
        file_path = f.name

    encoder.save_to_file(file_path)

    with open(file_path, "r") as f:
        lines = f.read().strip().splitlines()

    assert "A\tA\t1" in lines
    assert "R\tunary-for-R\t2" in lines

    os.remove(file_path)

def test_unfold_unary_head():
    external = ICLREncoderDecoder(
        unary_predicates=["A","B"],
        binary_predicates=["R","S"]
    )
    external.pair_term_dict[("a","b")] = "term-for-a-b"
    external.pair_term_dict[("b","a")] = "term-for-b-a"
    external.pair_term_dict[("d","a")] = "term-for-d-a"
    external.pair_term_dict[("a","c")] = "term-for-a-c" # this one MUST exist if a and c are connected via c4

    internal = CanonicalEncoderDecoder(
        unary_predicates=external.canonical_unary_predicates,
        binary_predicates=external.canonical_binary_predicates
    )
    # Simple treelike conjunction with 4 variables
    feature_mask_a = BitSet.from_subset(dimension=4,subset={0,1})
    variable_a = Variable(feature_mask_a,level=2) # Represents constant a

    feature_mask_b = BitSet.from_subset(dimension=4,subset={2,3})
    variable_b = Variable(feature_mask_b,level=1) # Represents constant ab

    feature_mask_c = BitSet.from_subset(dimension=4,subset={3})
    variable_c = Variable(feature_mask_c,level=0) # Represents constant ba

    feature_mask_d = BitSet.from_subset(dimension=4,subset={1})
    variable_d = Variable(feature_mask_d,level=0) # Represents constant c

    feature_mask_e = BitSet.from_subset(dimension=4,subset={2})
    variable_e = Variable(feature_mask_e,level=0) # Represents constant da

    variable_a.children[(1,0,0)] = variable_b # layer 1, colour 1, position 1 (the position does not matter)
    variable_b.children[(0,2,3)] = variable_c # layer 0, colour 3, position 4
    variable_a.children[(0,3,1)] = variable_d # layer 0, colour 4, position 2
    variable_a.children[(0,1,2)] = variable_e # layer 0, colour 2, position 3
    conj = TreeShapedConjunction(variable_a,2)

    var_const_idx = {variable_a: 0, variable_b: 1, variable_c: 2, variable_d: 3, variable_e: 4}
    # More edges and nodes should exist, but we dont include them because they are unnecessary for the test
    cd_graph = CDGraph(col_size = 1,delta=1,features=torch.ones(6,1),edges=torch.zeros(2,1),
                       edge_colours=torch.zeros(1,1), node_names = ["a",
                                                                    "term-for-a-b",
                                                                    "term-for-b-a",
                                                                    "c",
                                                                    "term-for-d-a",
                                                                    "term-for-a-c"])
    data_conj, root_vars = external.unfold(conj,
                                           internal_encoder=internal,
                                           var_const_idx=var_const_idx,
                                           cd_graph = cd_graph,
                                           head_predicate_arity=1)

    # Validate root variable
    assert root_vars == ["X0"]
    # Expected facts:
    # Unfolding happens in a depth-first way, which tells us the order of the variables
    # a->b->c->d->e
    assert ("X0", TYPE_PRED, "A") in data_conj
    assert ("X0", TYPE_PRED, "B") in data_conj # from features in variable a
    assert ("X0", "R", "X1") in data_conj
    assert ("X0", "S", "X1") in data_conj # from features in variable b
    assert ("X1", "S", "X0") in data_conj # from features in variable c
    assert ("X2", TYPE_PRED, "B") in data_conj # from features in variable d
    assert ("X3", "R", "X0") in data_conj # from features in variable e


def test_unfold_binary_head():
    external = ICLREncoderDecoder(
        unary_predicates=["A", "B"],
        binary_predicates=["R", "S"]
    )
    external.pair_term_dict[("a", "b")] = "term-for-a-b"
    external.pair_term_dict[("c", "a")] = "term-for-c-a"
    external.pair_term_dict[("b", "a")] = "term-for-b-a"
    external.pair_term_dict[("a", "d")] = "term-for-a-d"  # this one MUST exist if a and d are connected via c4
    internal = CanonicalEncoderDecoder(
        unary_predicates=external.canonical_unary_predicates,
        binary_predicates=external.canonical_binary_predicates
    )
    # Simple treelike conjunction with 4 variables
    feature_mask_a = BitSet.from_subset(dimension=4, subset={2, 3})
    variable_a = Variable(feature_mask_a, level=2)  # Represents constant ab, facts R(a,b) S(a,b)

    feature_mask_b = BitSet.from_subset(dimension=4, subset={0})
    variable_b = Variable(feature_mask_b, level=1)  # Represents constant a, fact A(a)

    feature_mask_c = BitSet.from_subset(dimension=4, subset={3})
    variable_c = Variable(feature_mask_c, level=0)  # Represents constant ca, fact S(c,a)

    feature_mask_d = BitSet.from_subset(dimension=4, subset={2})
    variable_d = Variable(feature_mask_d, level=0)  # Represents constant ba, fact R(b,a)

    feature_mask_e = BitSet.from_subset(dimension=4, subset={1})
    variable_e = Variable(feature_mask_e, level=0)  # Represents constant d, fact B(d)

    variable_a.children[(1, 0, 0)] = variable_b  # layer 1, colour 1, position 1 (the position does not matter)
    variable_b.children[(0, 1, 3)] = variable_c  # layer 0, colour 2, position 4
    variable_a.children[(0, 2, 2)] = variable_d  # layer 0, colour 3, position 3
    variable_b.children[(0, 3, 1)] = variable_e  # layer 0, colour 4, position 2
    conj = TreeShapedConjunction(variable_a,2)


    # Unfolding happens in a depth-first way, which tells us the order of the variables: a->b->c->e->d
    var_const_idx = {variable_a: 0, variable_b: 1, variable_c: 2, variable_e: 3, variable_d: 4}
    # More edges, nodes and 1 features should exist, but we dont include them because they are unnecessary for the test
    cd_graph = CDGraph(col_size=1, delta=4, features=torch.tensor([[0,0,0,0],
                                                                   [0,0,0,0],
                                                                   [0,0,0,0],
                                                                   [0,0,0,0],
                                                                   [0,0,0,0],
                                                                   [0,0,1,0]]),
                       edges=torch.zeros(2,1),
                       edge_colours=torch.zeros(1, 1),
                       node_names=["term-for-a-b", "a", "term-for-c-a", "d", "term-for-b-a", "term-for-a-d"])


    data_conj, root_vars = external.unfold(conj,
                                           internal_encoder=internal,
                                           var_const_idx=var_const_idx,
                                           cd_graph = cd_graph,
                                           head_predicate_arity=2)

    # Validate root variable
    assert root_vars == ["X0","X1"]

    # Expected facts:
    assert ("X0", "R", "X1") in data_conj
    assert ("X0", "S", "X1") in data_conj # from features in variable a
    assert("X0", TYPE_PRED, "A") in data_conj # from features in variable b
    assert ("X2", "S", "X0") in data_conj # from features in variable c
    assert ("X0", "R", "X3") in data_conj # this is how we ground the top-pred
    assert ("X3", TYPE_PRED, "B") in data_conj # from features in variable e
    assert ("X1", "R", "X0") in data_conj # from features in variable d

def test_unfold_empty():
    external = ICLREncoderDecoder( unary_predicates=["A"], binary_predicates=["R"] )
    external.pair_term_dict[("a", "b")] = "term-for-a-b"
    internal = CanonicalEncoderDecoder(
        unary_predicates=external.canonical_unary_predicates,
        binary_predicates=external.canonical_binary_predicates
    )
    feature_mask_a = BitSet.from_subset(dimension=2, subset=set())
    variable_a = Variable(feature_mask_a, level=2)
    conj = TreeShapedConjunction(variable_a,1)
    var_const_idx = {variable_a: 0}
    # More edges, nodes and 1 features should exist, but we dont include them because they are unnecessary for the test
    cd_graph = CDGraph(col_size=1, delta=2, features=torch.tensor([[0, 1]]), edges=torch.zeros(2,1),
                       edge_colours=torch.zeros(1, 1), node_names=["term-for-a-b"])
    data_conj, root_vars = external.unfold(conj,
                            internal_encoder=internal,
                            var_const_idx=var_const_idx,
                            cd_graph = cd_graph,
                            head_predicate_arity=2)
    assert root_vars == ["X0","X1"]
    # Expected fact
    assert ("X0", "R", "X1") in data_conj
