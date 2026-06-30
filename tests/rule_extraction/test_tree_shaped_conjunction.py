import numpy as np
import pytest

from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, Variable, walk
from src.utils.bitset import BitSet


class TestVariable:
    def test_variable_initialization(self):
        fm = BitSet.from_subset(5, {1, 4})
        var = Variable(fm, level=2)
        assert var.features is fm
        assert var.level == 2
        assert var.children == {}

    def test_get_feature_list(self):
        fm = BitSet.from_subset(5, {1, 4})
        var = Variable(fm)
        assert set(var.get_feature_list()) == {1, 4}


class TestWalk:

    def test_walk_single_node(self):
        root = Variable(BitSet.from_subset(3, {0}))
        nodes = list(walk(root))
        assert nodes == [root]

    def test_walk_depth_first(self):
        root = Variable(BitSet.from_subset(3, {0}))
        child1 = Variable(BitSet.from_subset(3, {1}))
        child2 = Variable(BitSet.from_subset(3, {2}))
        grandchild = Variable(BitSet.from_subset(3, {0, 2}))
        root.children[(0,0,0)] = child1
        root.children[(0,0,1)] = child2
        child1.children[(0,0,0)] = grandchild
        nodes = list(walk(root))
        assert nodes == [
            root,
            child1,
            grandchild,
            child2,
        ]

    def test_walk_is_stable_when_children_added_during_iteration(self):
        root = Variable(BitSet.from_subset(3, {0}))
        child = Variable(BitSet.from_subset(3, {1}))
        root.children[(0,0,0)] = child
        seen = []
        for node in walk(root):
            seen.append(node)
            if node is root:
                root.children[(0,0,1)] = Variable(
                    BitSet.from_subset(3, {2})
                )
        assert seen == [root, child]


class TestTreeShapedConjunction:

    def test_tree_with_only_root(self):
        root = Variable(BitSet.from_subset(5, {0}))
        tree = TreeShapedConjunction(root, n_colours=3)
        assert len(tree) == 1
        assert list(tree.walk()) == [root]

    def test_len_counts_all_nodes(self):
        root = Variable(BitSet.from_subset(5, {0}))
        tree = TreeShapedConjunction(root, n_colours=3)
        child1 = Variable(BitSet.from_subset(5, {1}))
        child2 = Variable(BitSet.from_subset(5, {2}))
        grandchild = Variable(BitSet.from_subset(5, {3}))
        root.children[(0,0,0)] = child1
        root.children[(0,0,1)] = child2
        child1.children[(0,0,0)] = grandchild
        tree.root_node = root
        assert len(tree) == 4

    def test_walk_delegates_to_root_node(self):
        root = Variable(BitSet.from_subset(5, {0}))
        child = Variable(BitSet.from_subset(5, {1}))
        root.children[(0,0,0)] = child
        tree = TreeShapedConjunction(root, n_colours=3)
        assert list(tree.walk()) == [root, child]