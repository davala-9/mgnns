from bidict import bidict
import torch
from typing import Tuple

from src.model.cd_graph import CDGraph
from src.model.gnn_transformation import apply_model
from src.utils.bitset import BitSet
from functools import cached_property

class TreeShapedConjunctionBuilder:
    def __init__(self, n_colours):
        self.n_colours = n_colours
        self.features, self.levels, self.children, self.parent = [], [], [], []

    def add(self, features, level, parent=-1, edge=None):
        i = self.num_vars() # new id equals current num of variables e.g. id 3 for current nodes 0, 1, 2
        self.features.append(features)
        self.levels.append(level)
        self.children.append({})
        self.parent.append(parent)
        if parent >= 0:
            self.children[parent][edge] = i
        return i

    def num_vars(self):
            return len(self.features)

    def build(self):
        return TreeShapedConjunction( self.n_colours,
            tuple(self.features), tuple(self.levels),
            tuple(self.children), tuple(self.parent))

# Immutable
class TreeShapedConjunction:
    def __init__(self, n_colours, features = (), levels = (), children = (), parent = ()):
        self.n_colours = n_colours
        self.features = features
        self.levels = levels
        self.children = children
        self.parent = parent

        self.parent_edge = [None,] # Auxiliary mapping from a node to its incoming edge
        for var_id in range(1,len(self)):
            par_id = self.parent[var_id]
            for edge, child_id  in self.children[par_id].items():
                if child_id == var_id:
                   self.parent_edge.append(edge)
                   break
            else:
                raise AssertionError(f"variable {var_id} not found among the children of its parent {par_id}")
        self.parent_edge = tuple(self.parent_edge)

    # Takes a variable and returns a TreeShapedConjunction that is the minimal subtree connecting it to the root
    def get_subtree_for(self, var_id: int, ):
        if not 0 <= var_id < len(self):
            raise ValueError(f"variable {var_id} does not appear to exist in the tree")
        # Construct backwards a list of variable ids, from the root variable to the variable in the input
        subtree_var_id_list = [var_id]
        while subtree_var_id_list[0] != 0:
            subtree_var_id_list.insert(0, self.parent[subtree_var_id_list[0]])
        builder = TreeShapedConjunctionBuilder(self.n_colours)
        sub_ids_2_original_ids = bidict()
        for new_id, var_id in enumerate(subtree_var_id_list):
            sub_ids_2_original_ids[new_id] = var_id
            builder.add(features=None,level=self.levels[var_id],parent=new_id - 1,edge=self.parent_edge[var_id])
        new_tree = builder.build()
        return new_tree, sub_ids_2_original_ids

    # Returns the compact subtree representing the empty subtree (root node only, no unary flags)
    @cached_property
    def initial_compact(self):
        return CompactSubTree(var_ids=(0,), masks=(self.features[0].to_empty_compressed(),))

    @cached_property
    # Pass this as a cd graph where nodes are ordered according to id and labelled "dummynode-id"
    def as_cd_graph(self):
        initial_features = torch.tensor([f.as_vector() for f in self.features], dtype=torch.float)
        edges = [[], []]
        edge_colours = []
        node_names = [f"dummynode_{i}" for i in range(len(self))]
        for var_id in range(len(self)):
            for (_, col, _), child_id in self.children[var_id].items():
                edges[0].append(child_id) # This order: TSCs upper nodes occur later in the computation
                edges[1].append(var_id)
                edge_colours.append(col)
        return CDGraph(col_size=self.n_colours,
                       delta=self.features[0].dimension,
                       features = initial_features,
                       edges = torch.tensor(edges, dtype=torch.long),
                       edge_colours = torch.tensor(edge_colours, dtype=torch.long),
                       node_names = node_names)

    # Returns another TreeShapedConjunction corresponding to the subtree given by a compact tree.
    def extract_from_compact(self, compact: "CompactSubTree"):
        new_features, new_levels, new_children, new_parent = [], [], [], []
        var_id_to_compact_index = {} # Match original var_id to the index of that var_id in the compact tree list
        for new_id, var_id in enumerate(compact.var_ids):
            var_id_to_compact_index[var_id] = new_id
            new_features.append(self.features[var_id].from_compressed(compact.masks[new_id]))
            new_levels.append(self.levels[var_id])
            new_children.append({})
            if self.parent[var_id] == -1:  # root: no parent to register
                new_parent.append(-1)
            else:
                parent_new_id = var_id_to_compact_index[self.parent[var_id]] # new id of the parent
                new_parent.append(parent_new_id)
                new_children[parent_new_id][self.parent_edge[var_id]] = new_id

        return TreeShapedConjunction(self.n_colours, tuple(new_features),
                                     tuple(new_levels), tuple(new_children), tuple(new_parent))

    # Returns another TreeShapedConjunction corresponding to a subtree given by a list of nodes and its features.
    # This is similar to a compact tree but the node labels are not compact.
    def from_subtree(self, subtree_nodes: list[int], subtree_features: list[BitSet]):
        assert len(subtree_nodes) > 0
        assert len(subtree_nodes)==len(subtree_features)

        # close under ancestors, adding missing predecessors
        keep = set()
        for node in subtree_nodes:
            while node != -1 and node not in keep:
                keep.add(node)
                node = self.parent[node]

        # renumber kept nodes; sort so ancestors get lower ids
        order = sorted(keep)
        old_id_to_new_id = {old: i for i, old in enumerate(order)}
        feat_dim = subtree_features[0].dimension

        # construct new tree
        subtree_node_to_feature = dict(zip(subtree_nodes, subtree_features))
        new_features = [subtree_node_to_feature.get(old, BitSet.from_subset(feat_dim,set())) for old in order]
        new_parent = [old_id_to_new_id.get(self.parent[old], -1) for old in order]
        new_level = [self.levels[old] for old in order]
        new_edges = [
            {label: old_id_to_new_id[child_id] for label, child_id in self.children[old].items() if child_id in keep}
            for old in order
        ]
        return TreeShapedConjunction(self.n_colours,features=tuple(new_features), levels = tuple(new_level),
                                     children=tuple(new_edges), parent=tuple(new_parent))

    # Returns a new TreeShapedConjunction, identical to this one except that var_id's feature is replaced.
    def with_feature(self, var_id: int, feature: BitSet) -> "TreeShapedConjunction":
        if not 0 <= var_id < len(self):
            raise ValueError(f"variable {var_id} does not appear to exist in the tree")
        new_features = self.features[:var_id] + (feature,) + self.features[var_id + 1:]
        return TreeShapedConjunction(self.n_colours, new_features, self.levels, self.children, self.parent)

    def __len__(self):
        return len(self.features)

# A compact representation of a subtree. The goal is to be as lean as possible.
# Represent only nodes and a compact version of each label: if the original variable has n relevant positions,
# the compact subtree used a bitset of dimension n (not the original total number of unary predicates)
class CompactSubTree:

    def __init__(self, var_ids: Tuple[int,...] = (), masks: Tuple[BitSet,...] = ()):
        self.var_ids = var_ids  # A frozen tuple containing the ids of the nodes that are present IN INCREASING ORDER
        self.masks = masks # A frozen tuple containing the compressed masks of the nodes that are present

    def __eq__(self, other):
        return isinstance(other, CompactSubTree) and self.var_ids == other.var_ids and self.masks == other.masks

    def __hash__(self):
        return hash((self.var_ids, self.masks))

    def get_successors(self, base_tree):
        # First get successors obtained by flipping a 0 to a 1 in the max of an existing node
        for var_id in range(len(self.var_ids)):
            for new_mask in self.masks[var_id].successors():
                yield CompactSubTree(self.var_ids, self.masks[:var_id] + (new_mask,) + self.masks[var_id + 1:])
        # Next, get successors obtained by adding a NEW child
        added_children = set() # First we extract all new nodes that need to be added
        added_children.update(child_id for var_id in self.var_ids for child_id in base_tree.children[var_id].values())
        added_children.difference_update(self.var_ids) # Remove those that are already present in the subtree
        added_children = sorted(added_children) # Sorted in ascending order
        # Efficient generation of the new CompactSubTrees in one pass through both tuples
        j = 0 # iteration over var_ids in this tree
        for child_id in added_children:
            while j < len(self.var_ids) and self.var_ids[j] < child_id:
                j += 1
            new_var_ids = self.var_ids[:j] + (child_id,) + self.var_ids[j:]
            new_mask = base_tree.features[child_id].to_empty_compressed() # We work with compressed sub-bitsets
            new_maskset = self.masks[:j] + (new_mask,) + self.masks[j:]
            yield CompactSubTree(new_var_ids, new_maskset)

    def get_predecessors(self, base_tree):
        # First, get predecessors obtained by flipping a 1 to a 0
        for index in range(len(self.masks)):
            for new_mask in self.masks[index].predecessors():
                yield CompactSubTree( self.var_ids, self.masks[:index] + (new_mask,) + self.masks[index + 1:])
        # Next, predecessors obtained by removing an empty child
        node_set = set(self.var_ids)
        for index, (var_id, mask) in enumerate(zip(self.var_ids, self.masks)):
            # Only remove nodes with an empty mask
            if not mask.is_empty():
                continue
            # Check there's no child of 'var_id' in the current subtree, before removing it; i.e. we only remove leaves
            if any(child in node_set for child in base_tree.children[var_id].values()):
                continue
            yield CompactSubTree(self.var_ids[:index] + self.var_ids[index + 1:],
                                 self.masks[:index] + self.masks[index + 1:])

    # Whether this subtree includes every node/bit of 'other' (and possibly more).
    # Both subtrees must be compact w.r.t. the same base_tree for the masks to be comparable.
    def is_superset_of(self, other: "CompactSubTree") -> bool:
        i = 0  # index into self.var_ids, advanced in step with other.var_ids (both sorted ascending)
        for var_id, other_mask in zip(other.var_ids, other.masks):
            while i < len(self.var_ids) and self.var_ids[i] < var_id:
                i += 1
            if i >= len(self.var_ids) or self.var_ids[i] != var_id:
                return False
            if not other_mask.subsetOf(self.masks[i]):
                return False
            i += 1
        return True

    def check_soundness(self, base_tree, device, model, threshold, pred_position):
        input_graph = base_tree.as_cd_graph.clone()
        input_graph.features =  torch.zeros_like(input_graph.features) # Return all features to zero, like GER
        for j, var_id in enumerate(self.var_ids):
            input_graph.features[var_id] = (
                torch.tensor(base_tree.features[var_id].from_compressed(self.masks[j]).as_vector()))
        output_graph = apply_model(input_graph,device,model)
        return output_graph.features[0][pred_position] >= threshold