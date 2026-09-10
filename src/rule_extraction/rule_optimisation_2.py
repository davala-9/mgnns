from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction
from src.model.gnn_transformation import apply_model
from src.utils.bitset import BitSet
from src.utils.utils import find_index_and_insert

if TYPE_CHECKING:  # avoid a circular import: fact_explanation.py imports this module back
    from src.rule_extraction.fact_explanation import FactExplainer


# OPTIMISATION 2
# Computes an *analytic* influence value for each candidate atom: the product of weight-matrix entries along the
# path connecting that atom to the head prediction. Atoms are then greedily added in decreasing order of this weight
# until the threshold is met.

def compute_path_weights(model, rule_body: TreeShapedConjunction, var_layer_mask, predicate_position: int):
    """Returns {(var_id, pos): weight} for every atom (var_id, pos) in rule_body, where `weight` is the
    analytic, weight-product-based contribution of that atom to the head prediction position, computed
    without running the model."""
    root_level = rule_body.levels[0]
    assert root_level == model.num_layers

    # self_vec[(var_id, l)] is the propagated weight vector at layer l: for every position k, how much
    # analytic weight reaches k, starting from the head prediction position and following var_id's own
    # path (tree edges down to var_id, then self-hops within var_id down to layer l).
    self_vec = {}
    root_vec = torch.zeros(model.layer_dimension(root_level))
    root_vec[predicate_position] = 1.0
    self_vec[(0, root_level)] = root_vec

    for var_id in range(len(rule_body)):
        level = rule_body.levels[var_id]
        if var_id != 0:
            parent = rule_body.parent[var_id]
            edge_layer, colour, child_pos = rule_body.parent_edge[var_id]
            matrix = model.matrix_A(edge_layer) if colour == -1 else model.matrix_B(edge_layer, colour)
            parent_vec = self_vec[(parent, edge_layer)]
            edge_weight = sum(
                parent_vec[j].item() * matrix[j, child_pos].item()
                for j in var_layer_mask[(parent, edge_layer)].elements()
            )
            start_vec = torch.zeros(model.layer_dimension(level))
            start_vec[child_pos] = edge_weight
            self_vec[(var_id, level)] = start_vec
        # Propagate var_id's own vector down to layer 0 via repeated self-hops (matrix_A only: neighbour
        # aggregation is handled separately, via the tree's own children, not as a self-hop).
        for l in range(level, 0, -1):
            current = self_vec[(var_id, l)]
            matrix = model.matrix_A(l)
            next_vec = torch.zeros(model.layer_dimension(l - 1))
            for j in var_layer_mask[(var_id, l)].elements():
                next_vec += current[j].item() * matrix[j, :]
            self_vec[(var_id, l - 1)] = next_vec

    contributions = {}
    for var_id in range(len(rule_body)):
        base_vec = self_vec[(var_id, 0)]
        for pos in rule_body.features[var_id].elements():
            contributions[(var_id, pos)] = base_vec[pos].item()
    return contributions


# Turns compute_path_weights' per-atom weights into a lattice_search.PriorityFrontier score_fn: the
# score of a CompactSubTree is the sum of the weights of every atom (var_id, pos) it currently includes
# -- an estimate of how close it is to being sound, without ever having to run the model on it.
#
# Computed incrementally rather than by re-summing every atom from scratch each time: a node always
# differs from its generating parent by exactly one atom (see CompactSubTree.get_successors), and that
# parent is always among the node's get_predecessors() (get_successors/get_predecessors are inverses),
# so we can recover the parent's already-cached score and just add the one new atom's weight to it.
def path_weight_score_fn(base_tree: TreeShapedConjunction, weights: dict[tuple[int, int], float]):
    cache = {base_tree.initial_compact: 0.0}  # the empty rule body includes no atoms, so it scores 0

    def score(node):
        if node not in cache:
            scored_predecessor = next((p for p in node.get_predecessors(base_tree) if p in cache), None)
            assert scored_predecessor is not None, "score() must be called in search() traversal order"
            cache[node] = cache[scored_predecessor] + _new_atom_weight(base_tree, weights, scored_predecessor, node)
        return cache[node]

    return score


# The weight of the single atom present in `node` but not in `predecessor`.
def _new_atom_weight(base_tree, weights, predecessor, node):
    if len(node.var_ids) != len(predecessor.var_ids):
        return 0.0  # a freshly added child always starts out with an empty mask -- nothing to weigh yet
    for var_id, old_mask, new_mask in zip(node.var_ids, predecessor.masks, node.masks):
        if old_mask != new_mask:
            new_compact_pos = (set(new_mask.elements()) - set(old_mask.elements())).pop()
            real_pos = base_tree.features[var_id].elements()[new_compact_pos]
            return weights.get((var_id, real_pos), 0.0)
    raise AssertionError("node is identical to predecessor")


def apply_optimisation(fe: FactExplainer, rule_body: TreeShapedConjunction, predicate_position: int,
                        var_layer_mask):
    assert fe.model.num_layers == 2 and fe.model.activation(1) is torch.relu, \
        "Optimisation 2's analytic weight-product approximation is only meaningful for a 2-layer ReLU model"

    contributions = compute_path_weights(fe.model, rule_body, var_layer_mask, predicate_position)

    # Sort candidate atoms by increasing analytic weight; pop() from the end to try the most
    # strongly-weighted atoms first.
    contributors_list = sorted((weight, var_id, pos) for (var_id, pos), weight in contributions.items())

    selected_nodes = [0]
    empty_bitset = BitSet.from_subset(fe.model.layer_dimension(0), {})
    selected_features = [empty_bitset]

    while contributors_list:
        weight, var_id, pos = contributors_list.pop()
        idx, is_new = find_index_and_insert(sorted_list=selected_nodes, element=var_id)
        if is_new:
            selected_features = (selected_features[:idx] + [empty_bitset] + selected_features[idx:])
        selected_features[idx] = selected_features[idx].add_element(pos)
        optimised_rule = rule_body.from_subtree(selected_nodes, selected_features)
        output_graph = apply_model(optimised_rule.as_cd_graph, fe.device, fe.model)
        if output_graph.features[0][predicate_position] > fe.threshold:
            return optimised_rule
    raise AssertionError("This part of the code should not be reachable. There's a bug in Optimisation 2")
