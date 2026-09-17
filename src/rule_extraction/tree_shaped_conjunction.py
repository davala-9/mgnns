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
        self.features, self.levels, self.children, self.parent, self.edge_in = [], [], [], [], []

    def add(self, features, level, parent=-1, edge=None):
        i = self.num_vars() # new id equals current num of variables e.g. id 3 for current nodes 0, 1, 2
        self.features.append(features)
        self.levels.append(level)
        self.children.append({})
        self.parent.append(parent)
        self.edge_in.append(edge)  # incoming edge (l, colour, j), or None for the root
        if parent >= 0:
            self.children[parent][edge] = i
        return i

    def num_vars(self):
            return len(self.features)

    # typed_constraints/var_types: see TreeShapedConjunction -- passed in at build time (rather than
    # tracked by the builder itself) since compute_tree_for already assembles var_types as it spawns
    # each variable.
    def build(self, typed_constraints: tuple = (), var_types: dict = None):
        return TreeShapedConjunction( self.n_colours,
            tuple(self.features), tuple(self.levels),
            tuple(self.children), tuple(self.parent),
            typed_constraints, var_types)

# Immutable
class TreeShapedConjunction:
    def __init__(self, n_colours, features = (), levels = (), children = (), parent = (),
                 typed_constraints: tuple = (), var_types: dict = None):
        self.n_colours = n_colours
        self.features = features
        self.levels = levels
        self.children = children
        self.parent = parent
        # The TypedConstraints this tree was built under, and each var_id's type under every one of them
        # (same order, one type per constraint) -- empty/{} when the tree was built without any. Used by
        # CompactSubTree.get_successors to prune the lattice search (see feature_exclusivity_groups and
        # edge_exclusivity_groups below), not by compute_tree_for itself.
        self.typed_constraints = typed_constraints
        self.var_types = var_types if var_types is not None else {}
        # Caches for the four methods below: each depends only on var_id/pair plus typed_constraints/
        # var_types, both fixed for this tree's lifetime, so there's no point recomputing them every time
        # CompactSubTree.get_successors is called for the same var_id/pair across many search steps.
        self._feature_exclusivity_groups_cache = {}
        self._edge_exclusivity_groups_cache = {}
        self._feature_always_one_positions_cache = {}
        self._edge_always_one_colours_cache = {}

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

    # Every feature-position exclusivity group applicable to var_id (see TypedConstraint.
    # feature_exclusivity_groups), aggregated across every typed_constraint via its own type for var_id.
    # Empty if var_id has no entry in var_types (e.g. the tree was built without any typed_constraints).
    def feature_exclusivity_groups(self, var_id) -> list[frozenset]:
        if var_id not in self._feature_exclusivity_groups_cache:
            self._feature_exclusivity_groups_cache[var_id] = [
                g for tc, node_type in zip(self.typed_constraints, self.var_types.get(var_id, []))
                for g in tc.feature_exclusivity_groups(node_type)]
        return self._feature_exclusivity_groups_cache[var_id]

    # Every colour exclusivity group applicable to an edge from from_var_id to to_var_id (see
    # TypedConstraint.edge_exclusivity_groups), aggregated the same way as feature_exclusivity_groups.
    def edge_exclusivity_groups(self, from_var_id, to_var_id) -> list[frozenset]:
        key = (from_var_id, to_var_id)
        if key not in self._edge_exclusivity_groups_cache:
            from_types = self.var_types.get(from_var_id, [])
            to_types = self.var_types.get(to_var_id, [])
            self._edge_exclusivity_groups_cache[key] = [
                g for tc, from_type, to_type in zip(self.typed_constraints, from_types, to_types)
                for g in tc.edge_exclusivity_groups(from_type, to_type)]
        return self._edge_exclusivity_groups_cache[key]

    # Every feature position that's always_one for var_id, aggregated across every typed_constraint via
    # its own type for var_id. Used only to prioritise CompactSubTree.get_successors' bit-flip order
    # (an always_one position is guaranteed true, so trying it first tends to reach a sound state
    # sooner), never to filter which positions are candidates at all.
    def feature_always_one_positions(self, var_id) -> frozenset:
        if var_id not in self._feature_always_one_positions_cache:
            self._feature_always_one_positions_cache[var_id] = frozenset(
                p for tc, node_type in zip(self.typed_constraints, self.var_types.get(var_id, []))
                for p in tc.get_feature_constraints(node_type).always_one)
        return self._feature_always_one_positions_cache[var_id]

    # Every colour that's always_one for an edge from from_var_id to to_var_id, aggregated the same way
    # as feature_always_one_positions. Used only to prioritise which new child get_successors offers first.
    def edge_always_one_colours(self, from_var_id, to_var_id) -> frozenset:
        key = (from_var_id, to_var_id)
        if key not in self._edge_always_one_colours_cache:
            from_types = self.var_types.get(from_var_id, [])
            to_types = self.var_types.get(to_var_id, [])
            self._edge_always_one_colours_cache[key] = frozenset(
                c for tc, from_type, to_type in zip(self.typed_constraints, from_types, to_types)
                for c in tc.get_edge_constraints(from_type, to_type).always_one)
        return self._edge_always_one_colours_cache[key]

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
        return TreeShapedConjunction(self.n_colours, new_features, self.levels, self.children, self.parent,
                                     self.typed_constraints, self.var_types)

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
        var_id_set = set(self.var_ids)
        # First get successors obtained by flipping a 0 to a 1 in the mask of an existing node -- skipped
        # when it would set a second member of a feature exclusivity group that already has a member set
        # (see TypedConstraint.feature_exclusivity_groups): that combination is structurally impossible,
        # so there's no point spending a soundness check on it. Among the rest, a flip that sets an
        # always_one position (guaranteed true, so likely to move towards a sound state -- see
        # TreeShapedConjunction.feature_always_one_positions) is yielded before the others. Both of these
        # only reorder/remove what's already offered -- never more than one atom added per step, so
        # get_predecessors/_new_atom_weight need no changes.
        for index, var_id in enumerate(self.var_ids):
            groups = base_tree.feature_exclusivity_groups(var_id)
            always_one_positions = base_tree.feature_always_one_positions(var_id)
            if groups:
                current_positions = set(base_tree.features[var_id].from_compressed(self.masks[index]).elements())
            priority, rest = [], []
            for new_mask in self.masks[index].successors():
                new_compact_pos = self.masks[index].new_elements(new_mask)[0]
                new_position = base_tree.features[var_id].elements()[new_compact_pos]
                if groups and any(new_position in g and current_positions & g for g in groups):
                    continue
                successor = CompactSubTree(self.var_ids, self.masks[:index] + (new_mask,) + self.masks[index + 1:])
                (priority if new_position in always_one_positions else rest).append(successor)
            yield from priority
            yield from rest
        # Next, get successors obtained by adding a NEW child -- skipped when it would connect a second
        # child within an edge exclusivity group that already has a connected child (see TypedConstraint.
        # edge_exclusivity_groups). Among the rest, a new child connected via an always_one colour (see
        # TreeShapedConjunction.edge_always_one_colours) is yielded before the others, for the same reason
        # as above.
        added_children = set() # First we extract all new nodes that need to be added
        added_children.update(child_id for var_id in self.var_ids for child_id in base_tree.children[var_id].values())
        added_children.difference_update(self.var_ids) # Remove those that are already present in the subtree
        added_children = sorted(added_children) # Sorted in ascending order
        existing_colours_by_parent = {}  # parent var_id -> colours of its children already in this subtree
        priority, rest = [], []
        # Efficient generation of the new CompactSubTrees in one pass through both tuples (buffered into
        # priority/rest so the always_one reordering doesn't need a second pass, just a delayed yield)
        j = 0 # iteration over var_ids in this tree
        for child_id in added_children:
            while j < len(self.var_ids) and self.var_ids[j] < child_id:
                j += 1
            parent_id = base_tree.parent[child_id]
            colour = base_tree.parent_edge[child_id][1]
            groups = base_tree.edge_exclusivity_groups(parent_id, child_id)
            if groups:
                if parent_id not in existing_colours_by_parent:
                    existing_colours_by_parent[parent_id] = {
                        base_tree.parent_edge[cid][1] for cid in base_tree.children[parent_id].values()
                        if cid in var_id_set
                    }
                if any(colour in g and existing_colours_by_parent[parent_id] & g for g in groups):
                    continue
            new_var_ids = self.var_ids[:j] + (child_id,) + self.var_ids[j:]
            new_mask = base_tree.features[child_id].to_empty_compressed() # We work with compressed sub-bitsets
            new_maskset = self.masks[:j] + (new_mask,) + self.masks[j:]
            successor = CompactSubTree(new_var_ids, new_maskset)
            is_always_one = colour in base_tree.edge_always_one_colours(parent_id, child_id)
            (priority if is_always_one else rest).append(successor)
        yield from priority
        yield from rest

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