import torch
from typing import Tuple

from src.model.cd_graph import CDGraph
from src.model.gnn_transformation import apply_model
from src.utils.bitset import BitSet
from functools import cached_property

class TreeShapedConjunction:
    def __init__(self, root_node, n_colours):
        self.n_colours = n_colours
        self.root_node = root_node

    def walk(self):
        return walk(self.root_node)

    @cached_property
    def index_tree(self):
        id_to_node = []
        node_to_id = {}
        children_ids = []
        parent_ids = []
        def dfs(node, parent_id=None):
            node_id = len(id_to_node)
            node_to_id[node] = node_id
            id_to_node.append(node)
            children_ids.append([])
            parent_ids.append(parent_id)
            for child in node.children.values():
                child_id = dfs(child, node_id)
                children_ids[node_id].append(child_id)
            return node_id
        dfs(self.root_node)
        return id_to_node, node_to_id, children_ids, parent_ids

    @cached_property
    def initial_subtree(self):
        return CompactSubTree(variable=self.root_node,
                              nodes=(self.index_tree[1][self.root_node],),  # this is: node_to_id[self.root_node]
                              masks=(self.root_node.features.to_empty_compressed(),))

    @cached_property
    def as_cd_graph(self):
        id_to_node, node_to_id, children_ids, parent_ids = self.index_tree
        features = torch.zeros((len(id_to_node),self.root_node.features.dimension))
        edges = [[], []]
        edge_colours = []
        node_names = ["dummy_name"] * len(id_to_node) # The node names do not matter for this.
        for node_id, node in enumerate(id_to_node):
            features[node_id] = torch.tensor(node.features.as_vector())
            for (_, col, _), child_var in node.children.items():
                child_id = node_to_id[child_var]
                edges[0].append(child_id)
                edges[1].append(node_id)
                edge_colours.append(col)
        return CDGraph(col_size=self.n_colours,
                       delta=self.root_node.features.dimension,
                       features = features,
                       edges = torch.tensor(edges),
                       edge_colours = torch.tensor(edge_colours),
                       node_names = node_names)

    def __len__(self):
        return  sum(1 for _ in self.walk())

# Represents a variable, with a given level and set of features (becoming unary predicates when unfolding the rule)
class Variable:
    def __init__(self, features: BitSet=None, level: int=0):
        self.features = features
        self.level = level
        self.children: dict[tuple[int,int,int], Variable] = {}  # Maps triple (l,col,j) to the relevant node.

    def get_feature_list(self):
        return self.features.elements()

# This class is very lean. The downside is that there are no checks to ensure it consistently represents what it should
# This class represents a subtree of the variable. It does so by storing:
# -- a tuple of nodes of the tree (as indexed by the index_tree method of variable)
# -- a matching tuple of masks (bitsets that are compressed versions of sub-bitsets of each variables' feature)
class CompactSubTree:

    def __init__(self, base_tree: TreeShapedConjunction, nodes: Tuple[int,...] = (), masks: Tuple[BitSet,...] = ()):
        self.base_tree = base_tree
        if any(a >= b for a, b in zip(nodes, nodes[1:])):
            raise ValueError("Values must be strictly increasing")
        self.nodes = nodes  # A frozen tuple containing the ids of the nodes that are present IN INCREASING ORDER
        self.masks = masks # A frozen tuple containing the compressed masks of the nodes that are present
        assert len(self.nodes) == len(self.masks) # Element i of self.nodes matches element i of self.labels

    def get_successors(self):
        # First get successors obtained by flipping a 0 to a 1 in the max of an existing node
        for index in range(len(self.masks)):
            for new_mask in self.masks[index].successors():
                yield CompactSubTree(self.base_tree, self.nodes,
                                     self.masks[:index] + (new_mask,) + self.masks[index + 1:])
        # Next, get successors obtained by adding a NEW child
        added_children = set() # First we extract all new nodes that need to be added
        id_to_node, _, children_ids, _ = self.base_tree.index_tree
        added_children.update(child for node in self.nodes for child in children_ids[node])
        added_children.difference_update(self.nodes) # Remove those that are already present in the subtree
        added_children = sorted(added_children)
        # Efficient generation of the new CompactSubTrees in one pass through both tuples
        j = 0
        n = len(self.nodes)
        for new_node in added_children:
            while j < n and self.nodes[j] < new_node:
                j += 1
            new_nodeset = self.nodes[:j] + (new_node,) + self.nodes[j:]
            new_mask = id_to_node[new_node].features.to_empty_compressed() # We work with compressed sub-bitsets
            new_maskset = self.masks[:j] + new_mask + self.masks[j:]
            yield CompactSubTree(self.base_tree, new_nodeset, new_maskset)

    def check_soundness(self, device, model, threshold, pred_position):
        input_graph = self.base_tree.as_cd_graph.clone()
        id_to_node, node_to_id, children_ids, parent_ids = self.base_tree.index_tree
        input_graph.features =  torch.zeros_like(input_graph.features) # Return all features to zero, like GER
        for j, node_id in enumerate(self.nodes):
            input_graph.features[node_id] = (
                torch.tensor(id_to_node[node_id].features.from_compressed(self.masks[j]).as_vector()))
        output_graph = apply_model(input_graph,device,model)
        root_id = node_to_id[self.base_tree.root_node]
        return output_graph.features[root_id][pred_position] >= threshold


# TODO: wouldn't it make sense to push this into Variable?
# Note that this is a generator function
def walk(node: Variable):
    children_snapshot = list(node.children.values())  # snapshot BEFORE yielding, in case it's modified.
    yield node
    for child in children_snapshot:
        yield from walk(child)





# class PartialVariable:
#     def __init__(self, full_variable: Variable, active_features: BitSet=None,
#                  active_children : dict[tuple[int,int,int], Variable] = None ):
#         self.full_variable = full_variable
#         if active_features is None:
#             self.active_features = BitSet.from_subset(full_variable.features.dimension,set())
#         else:
#             self.active_features = active_features
#         assert self.active_features.subsetOf(full_variable.features)
#         if active_children is None:
#             self.active_children = {}
#         else:
#             self.active_children = active_children
#         for (l, col, j) in self.active_children:
#             assert (l, col, j) in full_variable.children
#
#     def clone(self):
#
#         cloned_var = PartialVariable(full_variable=self.full_variable,
#                                      active_features=self.active_features.clone())
#         for (l, col, j), child in self.active_children.items():
#             cloned_var.active_children[(l,col,j)] =
#
#
#
#     def getSuccessors(self):
#         for j in self.active_features.new_elements(self.full_variable.features):
#             new_feature_set = self.active_features.clone().add(j)
#             yield PartialVariable(full_variable=self.full_variable, active_features=new_feature_set,
#                                   active_children=self.active_children)
