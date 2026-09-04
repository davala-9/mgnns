import pytest
import torch

from src.rule_extraction.tree_shaped_conjunction import (
    TreeShapedConjunctionBuilder, CompactSubTree,
)
from src.model.gnn_architectures import GNN
from src.utils.bitset import BitSet


# --- Shared tree fixtures, mirroring the tree shapes from the previous (stale) test file ---

def build_single_root_tree():
    # Just a root node, no children.
    builder = TreeShapedConjunctionBuilder(n_colours=3)
    builder.add(features=BitSet.from_subset(5, {0}), level=0, parent=-1)
    return builder.build()

def build_multi_node_tree():
    # root
    # |- child1 (edge (2,0,0))
    # |   |- grandchild (edge (1,0,0))
    # |- child2 (edge (2,1,0))
    builder = TreeShapedConjunctionBuilder(n_colours=3)
    root_id = builder.add(features=BitSet.from_subset(5, {0}), level=2, parent=-1)
    child1_id = builder.add(features=BitSet.from_subset(5, {1}), level=1, parent=root_id, edge=(2, 0, 0))
    child2_id = builder.add(features=BitSet.from_subset(5, {2}), level=1, parent=root_id, edge=(2, 1, 0))
    grandchild_id = builder.add(features=BitSet.from_subset(5, {3}), level=0, parent=child1_id, edge=(1, 0, 0))
    return builder.build(), (root_id, child1_id, child2_id, grandchild_id)


# --- TreeShapedConjunctionBuilder ---

def test_add_returns_sequential_ids():
    builder = TreeShapedConjunctionBuilder(n_colours=2)
    ids = [builder.add(features=None, level=0, parent=-1) for _ in range(4)]
    assert ids == [0, 1, 2, 3]

def test_add_registers_child_under_parent():
    builder = TreeShapedConjunctionBuilder(n_colours=2)
    root_id = builder.add(features=None, level=1, parent=-1)
    child_id = builder.add(features=None, level=0, parent=root_id, edge=(1, 0, 0))
    assert builder.children[root_id][(1, 0, 0)] == child_id

def test_add_root_registers_no_children_anywhere():
    builder = TreeShapedConjunctionBuilder(n_colours=2)
    builder.add(features=None, level=0, parent=-1)
    assert builder.children == [{}]

def test_num_vars():
    builder = TreeShapedConjunctionBuilder(n_colours=2)
    assert builder.num_vars() == 0
    builder.add(features=None, level=0, parent=-1)
    builder.add(features=None, level=0, parent=0, edge=(0, 0, 0))
    assert builder.num_vars() == 2

def test_build_produces_immutable_tuples():
    builder = TreeShapedConjunctionBuilder(n_colours=2)
    builder.add(features=None, level=0, parent=-1)
    tree = builder.build()
    assert isinstance(tree.features, tuple)
    assert isinstance(tree.levels, tuple)
    assert isinstance(tree.children, tuple)
    assert isinstance(tree.parent, tuple)


# --- TreeShapedConjunction: construction & basics ---

def test_len_single_root():
    tree = build_single_root_tree()
    assert len(tree) == 1

def test_len_multi_node_tree():
    tree, _ = build_multi_node_tree()
    assert len(tree) == 4

def test_parent_tuple():
    tree, (root, child1, child2, grandchild) = build_multi_node_tree()
    assert tree.parent == (-1, root, root, child1)

def test_parent_edge_root_is_none():
    tree = build_single_root_tree()
    assert tree.parent_edge == (None,)

def test_parent_edge_matches_add_calls():
    tree, _ = build_multi_node_tree()
    assert tree.parent_edge == (None, (2, 0, 0), (2, 1, 0), (1, 0, 0))


# --- get_subtree_for ---

def test_get_subtree_for_root():
    tree = build_single_root_tree()
    sub, mapping = tree.get_subtree_for(0)
    assert len(sub) == 1
    assert dict(mapping) == {0: 0}

def test_get_subtree_for_grandchild_excludes_unrelated_branch():
    tree, (root, child1, child2, grandchild) = build_multi_node_tree()
    sub, mapping = tree.get_subtree_for(grandchild)
    assert len(sub) == 3  # root, child1, grandchild -- NOT child2
    assert dict(mapping) == {0: root, 1: child1, 2: grandchild}
    assert sub.parent == (-1, 0, 1)  # renumbered chain

def test_get_subtree_for_invalid_var_id_raises():
    tree = build_single_root_tree()
    with pytest.raises(ValueError):
        tree.get_subtree_for(5)


# --- initial_compact ---

def test_initial_compact_is_root_only():
    tree, (root, *_) = build_multi_node_tree()
    compact = tree.initial_compact
    assert compact.var_ids == (root,)

def test_initial_compact_mask_is_empty_and_compressed():
    tree = build_single_root_tree()  # root features = {0} out of dimension 5 -> 1 relevant position
    compact = tree.initial_compact
    assert compact.masks[0].dimension == 1  # compressed to the number of relevant (set) positions
    assert compact.masks[0].is_empty()


# --- as_cd_graph ---

def test_as_cd_graph_basic_shape():
    tree, _ = build_multi_node_tree()
    g = tree.as_cd_graph
    assert g.col_size == 3
    assert g.delta == 5
    assert g.node_names == ["dummynode_0", "dummynode_1", "dummynode_2", "dummynode_3"]
    assert g.features.shape == (4, 5)
    assert g.features.sum().item() == 0  # all-zero placeholder features

def test_as_cd_graph_edges_reflect_child_to_parent_structure():
    tree, (root, child1, child2, grandchild) = build_multi_node_tree()
    g = tree.as_cd_graph
    edges = set(zip(g.edges[0].tolist(), g.edges[1].tolist(), g.edge_colours.tolist()))
    assert edges == {
        (child1, root, 0),
        (child2, root, 1),
        (grandchild, child1, 0),
    }


# --- extract_from_compact ---

def test_extract_from_compact_keeps_only_selected_nodes():
    tree, (root, child1, child2, grandchild) = build_multi_node_tree()
    # Keep root, child1, grandchild (skip child2), each with a "keep everything" compressed mask
    compact = CompactSubTree(
        var_ids=(root, child1, grandchild),
        masks=(BitSet.from_subset(1, {0}), BitSet.from_subset(1, {0}), BitSet.from_subset(1, {0})),
    )
    result = tree.extract_from_compact(compact)
    assert len(result) == 3
    assert result.parent == (-1, 0, 1)  # root keeps -1 as its parent, no KeyError
    assert result.children == ({(2, 0, 0): 1}, {(1, 0, 0): 2}, {})

def test_extract_from_compact_recovers_original_features():
    tree, (root, child1, child2, grandchild) = build_multi_node_tree()
    compact = CompactSubTree(
        var_ids=(root, child1, grandchild),
        masks=(BitSet.from_subset(1, {0}), BitSet.from_subset(1, {0}), BitSet.from_subset(1, {0})),
    )
    result = tree.extract_from_compact(compact)
    assert result.features[0] == tree.features[root]
    assert result.features[1] == tree.features[child1]
    assert result.features[2] == tree.features[grandchild]

def test_extract_from_compact_root_only():
    tree, _ = build_multi_node_tree()
    result = tree.extract_from_compact(tree.initial_compact)
    assert len(result) == 1
    assert result.parent == (-1,)


# --- CompactSubTree.get_successors ---

def test_get_successors_adds_each_direct_child_of_the_root():
    tree, (root, child1, child2, grandchild) = build_multi_node_tree()
    successors = list(tree.initial_compact.get_successors(tree))
    var_id_sets = {s.var_ids for s in successors}
    assert (root, child1) in var_id_sets
    assert (root, child2) in var_id_sets

def test_get_successors_does_not_add_grandchildren_directly():
    tree, (root, child1, child2, grandchild) = build_multi_node_tree()
    successors = list(tree.initial_compact.get_successors(tree))
    var_id_sets = {s.var_ids for s in successors}
    assert not any(grandchild in var_ids for var_ids in var_id_sets)

def test_get_successors_also_flips_bits_on_existing_nodes():
    tree, (root, *_) = build_multi_node_tree()
    successors = list(tree.initial_compact.get_successors(tree))
    # The root's own (still-empty) mask should get a "flip a 0 to a 1" successor too
    assert any(s.var_ids == (root,) for s in successors)


# --- CompactSubTree.get_predecessors ---

def test_get_predecessors_removes_empty_leaf():
    tree, (root, child1, child2, grandchild) = build_multi_node_tree()
    compact = CompactSubTree(
        var_ids=(root, child1),
        masks=(BitSet.from_subset(1, {0}), BitSet.from_subset(1, set())),  # child1's mask is empty
    )
    var_id_sets = {p.var_ids for p in compact.get_predecessors(tree)}
    assert (root,) in var_id_sets  # child1 can be dropped: it's a leaf with an empty mask

def test_get_predecessors_does_not_remove_node_with_children_present():
    tree, (root, child1, child2, grandchild) = build_multi_node_tree()
    compact = CompactSubTree(
        var_ids=(root, child1, grandchild),
        masks=(BitSet.from_subset(1, {0}), BitSet.from_subset(1, set()), BitSet.from_subset(1, set())),
    )
    var_id_sets = {p.var_ids for p in compact.get_predecessors(tree)}
    # child1 has an empty mask but its child (grandchild) is present in the subtree, so it must NOT
    # be droppable by the "remove an empty leaf" rule (regression test for the .values() fix).
    assert (root, grandchild) not in var_id_sets
    assert (root, child1) in var_id_sets  # grandchild itself, a genuine empty leaf, can still be dropped


# --- from_subtree ---

def test_from_subtree_closes_under_ancestors():
    tree, (root, child1, child2, grandchild) = build_multi_node_tree()
    result = tree.from_subtree([grandchild], [BitSet.from_subset(5, {3})])
    # root and child1 get pulled in as ancestors, but not child2 (unrelated branch)
    assert len(result) == 3
    assert result.parent == (-1, 0, 1)

def test_from_subtree_gives_unset_ancestors_empty_features():
    tree, (root, child1, child2, grandchild) = build_multi_node_tree()
    result = tree.from_subtree([grandchild], [BitSet.from_subset(5, {3})])
    assert result.features[0].is_empty()  # root: not explicitly given a feature
    assert result.features[1].is_empty()  # child1: not explicitly given a feature
    assert result.features[2].elements() == [3]  # grandchild: the one we actually passed in

def test_from_subtree_root_alone_has_no_parent():
    tree, (root, *_) = build_multi_node_tree()
    result = tree.from_subtree([root], [BitSet.from_subset(5, {0})])
    assert len(result) == 1
    assert result.parent == (-1,)

def test_from_subtree_explicit_ancestor_and_descendant():
    tree, (root, child1, child2, grandchild) = build_multi_node_tree()
    result = tree.from_subtree([root, grandchild],
                               [BitSet.from_subset(5, {0}), BitSet.from_subset(5, {3})])
    assert len(result) == 3
    assert result.parent == (-1, 0, 1)


# --- CompactSubTree.check_soundness ---

def test_check_soundness_runs_and_returns_a_boolean_tensor():
    torch.manual_seed(0)
    tree, _ = build_multi_node_tree()
    model = GNN(feature_dimension=5, num_edge_colours=3, aggregation_1="max", aggregation_2="max")
    device = torch.device("cpu")

    result = tree.initial_compact.check_soundness(tree, device, model, threshold=0.5, pred_position=0)
    assert result.dtype == torch.bool
    assert result.dim() == 0
