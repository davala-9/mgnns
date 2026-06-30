from src.utils.bitset import BitSet
import numpy as np


# TODO: combine these two classes. No point in having them separate
class TreeShapedConjunction:
    def __init__(self, root_node, n_colours):
        self.n_colours = n_colours
        self.root_node = root_node

    def walk(self):
        return walk(self.root_node)

    def __len__(self):
        return  sum(1 for _ in self.walk())


# Represents a variable, with a given level and set of features (becoming unary predicates when unfolding the rule)
class Variable:
    def __init__(self, features: BitSet=None, level=int):
        self.features = features
        self.level = level
        self.children: dict[tuple[int,int,int], Variable] = {}  # Maps triple (l,col,j) to the relevant node.

    def get_feature_list(self):
        return self.features.elements()

# Note that this is a generator function
def walk(node: Variable):
    children_snapshot = list(node.children.values())  # snapshot BEFORE yielding, in case it's modified.
    yield node
    for child in children_snapshot:
        yield from walk(child)





