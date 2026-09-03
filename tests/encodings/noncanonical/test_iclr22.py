import os
import tempfile
import torch
import pytest

from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.iclr22 import ICLREncoderDecoder
from src.encodings.noncanonical.noncanonical import GroundContext
from src.model.cd_graph import CDGraph
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunctionBuilder
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
    assert encoder.data_pred_to_unary_canonical["A"] == "A"
    assert encoder.data_pred_to_unary_canonical.inverse["A"] == "A"
    assert encoder.data_pred_to_unary_canonical["R"] == "unary-for-R"
    assert encoder.data_pred_to_unary_canonical.inverse["unary-for-R"] == "R"
    assert encoder.data_pred_to_arity["A"] == 1
    assert encoder.data_pred_to_arity["R"] == 2
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


def test_encode_dataset_use_dummy_constants_skips_pair_nodes(encoder, sample_dataset):
    # Dummy constants ('#', '##') must only be paired with genuine data constants ("a", "b"),
    # never with the synthetic pair-term nodes ("term-for-a-b", ...) created by the encoding itself.
    encoder.encode_dataset(sample_dataset, use_dummy_constants=True)
    dummy_pairs = {pair for pair in encoder.pair_term_dict if pair[0] in ("#", "##")}
    assert dummy_pairs == {("#", "a"), ("#", "b"), ("##", "a"), ("##", "b")}


def test_decode_binary_fact(encoder):
    with pytest.raises(AssertionError):
        encoder.decode_fact("a",encoder.col1, "b")

def test_decode_unary_facts(encoder,sample_dataset):
    # TODO: consider making "term-for" and "unary-for" and dashes into protected strings
    encoder.encode_dataset(sample_dataset) # Necessary to create the relevant terms
    assert ("a",TYPE_PRED,"A") == encoder.decode_fact("a",TYPE_PRED,"A")
    assert ("a","R","b") == encoder.decode_fact("term-for-a-b",TYPE_PRED,"unary-for-R")

def test_decode_fact_pair_node_with_unary_arity_predicate_is_filtered(encoder, sample_dataset):
    # "A" is a unary-arity data predicate; predicting it on a node that represents a pair is
    # an invalid/spurious combination and must be filtered out (returns None), not decoded.
    encoder.encode_dataset(sample_dataset)  # registers "term-for-a-b"
    assert encoder.decode_fact("term-for-a-b", TYPE_PRED, "A") is None

def test_decode_fact_single_constant_with_binary_arity_predicate_is_filtered(encoder, sample_dataset):
    # "unary-for-R" has data-arity 2; predicting it on a plain constant node is the symmetric
    # invalid combination and must be filtered out (returns None) as well.
    encoder.encode_dataset(sample_dataset)
    assert encoder.decode_fact("a", TYPE_PRED, "unary-for-R") is None

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


# --- unfold_match_ground -----------------------------------------------------------------------
# can_conj trees below are built directly with TreeShapedConjunctionBuilder (var ids are plain
# ints indexing parallel features/children/parent tuples), matching the current TreeShapedConjunction
# API. Edge keys are (level, colour, position) tuples; only the colour (middle element) matters
# to the unfolding logic.

def test_unfold_match_ground_unary_head_no_children():
    external = ICLREncoderDecoder(unary_predicates=["A"], binary_predicates=["R"])
    internal = CanonicalEncoderDecoder(
        unary_predicates=external.canonical_unary_predicates,
        binary_predicates=external.canonical_binary_predicates,
    )

    builder = TreeShapedConjunctionBuilder(n_colours=4)
    builder.add(features=BitSet.from_subset(2, {0}), level=0, parent=-1)  # single "a": A(a)
    conj = builder.build()

    cd_graph = CDGraph(col_size=4, delta=2, features=torch.zeros(1, 2),
                       edges=torch.zeros(2, 0, dtype=torch.long), edge_colours=torch.zeros(0, dtype=torch.long),
                       node_names=["a"])
    ground = GroundContext(fact=("a", TYPE_PRED, "A"), graph=cd_graph, canonical_variable_to_constant_index={0: 0})

    data_conj, head = external.unfold_match_ground(can_conj=conj, internal_encoder=internal,
                                                    head_predicate="A", grounding_context=ground)

    assert head == ("X0", TYPE_PRED, "A")
    assert set(data_conj) == {("X0", TYPE_PRED, "A")}


def test_unfold_match_ground_binary_head_with_col3_reversal():
    external = ICLREncoderDecoder(unary_predicates=["A"], binary_predicates=["R", "S"])
    external.term_for_pair(("a", "b"))
    external.term_for_pair(("b", "a"))
    internal = CanonicalEncoderDecoder(
        unary_predicates=external.canonical_unary_predicates,
        binary_predicates=external.canonical_binary_predicates,
    )
    # canonical_unary_predicates = ["A", "unary-for-R", "unary-for-S"] -> A=0, unary-for-R=1, unary-for-S=2
    # canonical_binary_predicates = [col1, col2, col3, col4]           -> col3=2

    builder = TreeShapedConjunctionBuilder(n_colours=4)
    root = builder.add(features=BitSet.from_subset(3, {2}), level=1, parent=-1)             # pair (a,b): S(a,b)
    builder.add(features=BitSet.from_subset(3, {1}), level=0, parent=root, edge=(1, 2, 0))  # col3 -> pair (b,a): R(b,a)
    conj = builder.build()

    node_names = ["term-for-a-b", "term-for-b-a"]
    cd_graph = CDGraph(col_size=4, delta=3, features=torch.zeros(2, 3),
                       edges=torch.zeros(2, 0, dtype=torch.long), edge_colours=torch.zeros(0, dtype=torch.long),
                       node_names=node_names)
    ground = GroundContext(fact=("a", "R", "b"), graph=cd_graph,
                           canonical_variable_to_constant_index={0: 0, 1: 1})

    data_conj, head = external.unfold_match_ground(can_conj=conj, internal_encoder=internal,
                                                    head_predicate="R", grounding_context=ground)

    assert head == ("X0", "R", "X1")
    assert set(data_conj) == {
        ("X0", "S", "X1"),  # from the root pair's own feature
        ("X1", "R", "X0"),  # from the col3-reversed child
    }


def test_unfold_match_ground_col4_backfill_preserves_multichar_constants():
    # Regression test: get_data_constants_for_can_variable used to do list(canonical_constant),
    # which silently chopped a multi-character constant name down to its first letter.
    external = ICLREncoderDecoder(unary_predicates=["A"], binary_predicates=["R"])
    external.term_for_pair(("alice", "bob"))
    external.term_for_pair(("bob", "alice"))
    internal = CanonicalEncoderDecoder(
        unary_predicates=external.canonical_unary_predicates,
        binary_predicates=external.canonical_binary_predicates,
    )
    # canonical_unary_predicates = ["A", "unary-for-R"] -> A=0, unary-for-R=1
    # canonical_binary_predicates = [col1, col2, col3, col4] -> col4=3

    builder = TreeShapedConjunctionBuilder(n_colours=4)
    root = builder.add(features=BitSet.from_subset(2, {0}), level=1, parent=-1)             # alice: A(alice)
    builder.add(features=BitSet.from_subset(2, {0}), level=0, parent=root, edge=(1, 3, 0))  # col4 -> bob: A(bob)
    conj = builder.build()

    node_names = ["alice", "bob", "term-for-alice-bob"]
    features = torch.tensor([[0., 0.], [0., 0.], [0., 1.]])  # the alice-bob pair holds canonical "unary-for-R"
    cd_graph = CDGraph(col_size=4, delta=2, features=features,
                       edges=torch.zeros(2, 0, dtype=torch.long), edge_colours=torch.zeros(0, dtype=torch.long),
                       node_names=node_names)
    ground = GroundContext(fact=("alice", TYPE_PRED, "A"), graph=cd_graph,
                           canonical_variable_to_constant_index={0: 0, 1: 1})

    data_conj, head = external.unfold_match_ground(can_conj=conj, internal_encoder=internal,
                                                    head_predicate="A", grounding_context=ground)

    assert head == ("X0", TYPE_PRED, "A")
    # The col4 edge must be backfilled with the real relation between "alice" and "bob" in full,
    # not between truncated single-character stand-ins.
    assert set(data_conj) == {
        ("X0", TYPE_PRED, "A"),
        ("X1", TYPE_PRED, "A"),
        ("X0", "R", "X1"),
    }


# --- unfold_all ----------------------------------------------------------------------------
# unfold_all had no prior test coverage at all. These target the bugs fixed in it: the col3
# branch silently dropping its recursive result, the col4 branch crashing (unhashable list in
# a set), the empty-feature branch never triggering, and duplicate atoms not being removed.

def test_unfold_all_empty_feature_branching_and_dedup():
    external = ICLREncoderDecoder(unary_predicates=["A"], binary_predicates=["R"])
    internal = CanonicalEncoderDecoder(
        unary_predicates=external.canonical_unary_predicates,
        binary_predicates=external.canonical_binary_predicates,
    )
    # canonical_unary_predicates = ["A", "unary-for-R"] -> A=0
    # canonical_binary_predicates = [col1, col2, col3, col4] -> col1=0

    builder = TreeShapedConjunctionBuilder(n_colours=4)
    var0 = builder.add(features=BitSet.from_subset(2, {0}), level=2, parent=-1)                      # single "a": A(a)
    var1 = builder.add(features=BitSet.from_subset(2, set()), level=1, parent=var0, edge=(2, 0, 0))  # col1 -> pair, no relevant predicate
    builder.add(features=BitSet.from_subset(2, {0}), level=0, parent=var1, edge=(1, 0, 0))           # col1 -> single "a" again: A(a)
    conj = builder.build()

    data_conjs, head = external.unfold_all(can_conj=conj, internal_encoder=internal, head_predicate="A")

    assert head == ("X0", TYPE_PRED, "A")
    # With no relevant feature on the pair node, unfolding must branch over every binary predicate
    # in both directions (there's only one predicate here, "R", so exactly 2 alternative bodies).
    assert len(data_conjs) == 2
    bodies = [frozenset(dc) for dc in data_conjs]
    assert frozenset({("X0", TYPE_PRED, "A"), ("X0", "R", "X1")}) in bodies
    assert frozenset({("X0", TYPE_PRED, "A"), ("X1", "R", "X0")}) in bodies
    # The revisited single variable re-asserts the same unary atom; it must be deduplicated.
    for dc in data_conjs:
        assert len(dc) == len(set(dc))


def test_unfold_all_col3_branch_is_preserved():
    external = ICLREncoderDecoder(unary_predicates=["A"], binary_predicates=["R"])
    internal = CanonicalEncoderDecoder(
        unary_predicates=external.canonical_unary_predicates,
        binary_predicates=external.canonical_binary_predicates,
    )
    # canonical_unary_predicates = ["A", "unary-for-R"] -> unary-for-R=1
    # canonical_binary_predicates = [col1, col2, col3, col4] -> col3=2

    builder = TreeShapedConjunctionBuilder(n_colours=4)
    var0 = builder.add(features=BitSet.from_subset(2, {1}), level=1, parent=-1)             # pair (a,b): R(a,b)
    builder.add(features=BitSet.from_subset(2, {1}), level=0, parent=var0, edge=(1, 2, 0))  # col3 -> pair (b,a): R(b,a)
    conj = builder.build()

    data_conjs, head = external.unfold_all(can_conj=conj, internal_encoder=internal, head_predicate="R")

    assert head == ("X0", "R", "X1")
    # A col3 child must not be dropped: this used to silently return no rule bodies at all.
    assert len(data_conjs) == 1
    assert set(data_conjs[0]) == {("X0", "R", "X1"), ("X1", "R", "X0")}


def test_unfold_all_col4_branches_over_binary_predicates():
    external = ICLREncoderDecoder(unary_predicates=["A"], binary_predicates=["R"])
    internal = CanonicalEncoderDecoder(
        unary_predicates=external.canonical_unary_predicates,
        binary_predicates=external.canonical_binary_predicates,
    )
    # canonical_binary_predicates = [col1, col2, col3, col4] -> col4=3

    builder = TreeShapedConjunctionBuilder(n_colours=4)
    var0 = builder.add(features=BitSet.from_subset(2, {0}), level=1, parent=-1)             # single "a": A(a)
    builder.add(features=BitSet.from_subset(2, {0}), level=0, parent=var0, edge=(1, 3, 0))  # col4 -> single "b": A(b)
    conj = builder.build()

    data_conjs, head = external.unfold_all(can_conj=conj, internal_encoder=internal, head_predicate="A")

    assert head == ("X0", TYPE_PRED, "A")
    # A col4 child has no fixed direction/predicate, so unfolding must branch over every option
    # (this used to crash trying to put a list into a set).
    assert len(data_conjs) == 2
    bodies = [frozenset(dc) for dc in data_conjs]
    assert frozenset({("X0", TYPE_PRED, "A"), ("X1", TYPE_PRED, "A"), ("X0", "R", "X1")}) in bodies
    assert frozenset({("X0", TYPE_PRED, "A"), ("X1", TYPE_PRED, "A"), ("X1", "R", "X0")}) in bodies
