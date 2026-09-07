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
# Ported from the old (rule_optimisation_2_old.py) 2-layer/ReLU-only implementation, which computed an
# *analytic* influence value for each candidate atom -- the product of weight-matrix entries along the
# path connecting that atom to the head prediction -- instead of grounding and running the model per
# atom (as Optimisation 1 does). Atoms are then greedily added in decreasing order of this weight until
# the threshold is met, exactly like Optimisation 1's second phase.
#
# The old code assumed every variable is tied to a single scalar feature position at a time, so "multiply
# two matrix entries" was unambiguous. The new BasicExplanation tree (built by get_basic_explanation) is
# more general: a variable's relevant positions at a layer are a BitSet that can hold several bits at
# once (var_layer_mask), since backpropagate_relevance aggregates "any positive row" across every
# relevant position of its parent rather than tracking a single position-to-position link. So instead of
# a flat product of two scalars, we propagate a full analytic WEIGHT VECTOR down through the tree: each
# hop distributes the upstream weight across every matrix entry from a relevant row, and multiple
# contributing rows are summed (weighted by their own upstream magnitude, not treated as equally-weighted
# 1s). This reduces to the exact old formula whenever a hop has a single relevant position (the common
# case, since a variable's own relevant position at its creation layer is always a single bit), and is
# the direct generalisation of what the old code already did for its own multi-path case (its own
# comment: "where col2 is -1 there might be multiple influence paths -- we sum them here").


def _compute_path_weights(model, rule_body: TreeShapedConjunction, var_layer_mask, predicate_position: int):
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


def apply_optimisation(fe: FactExplainer, rule_body: TreeShapedConjunction, predicate_position: int,
                        var_layer_mask):
    assert fe.model.num_layers == 2 and fe.model.activation(1) is torch.relu, \
        "Optimisation 2's analytic weight-product approximation is only meaningful for a 2-layer ReLU model"

    contributions = _compute_path_weights(fe.model, rule_body, var_layer_mask, predicate_position)

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
