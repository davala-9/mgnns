from __future__ import annotations
from typing import TYPE_CHECKING

from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction
from src.utils.bitset import BitSet
from src.model.cd_graph import TraceCollector
from src.model.gnn_transformation import apply_model
from src.utils.utils import find_index_and_insert

if TYPE_CHECKING:  # avoid a circular import: fact_explanation.py imports this module back
    from src.rule_extraction.fact_explanation import FactExplainer


# OPTIMISATION 1
# Make a dictionary of all unary atoms of gamma_i and compute an ``influence'' value of each
# by creating the minimal sub-rule of BasicExplanation that connects this atom with the head variable.
# Ground such rule and apply the GNN to it, then check the value of the node and position corresponding to the head
# fact. If zero, start travelling backwards through the relevant part of the computation graph; at each node of the
# computation tree, sum the values of the relevant positions of the activations. Keep doing this until not zero.


def apply_optimisation(fe: FactExplainer, rule_body: TreeShapedConjunction, predicate_position: int,
                       var_layer_mask: dict[tuple[int, int], BitSet]):

    # Build a dictionary mapping a pair (var_id, pos), representing a unary atom in the rule, to a "contribution":
    # a pair (layer, value) representing the first layer (from the end) where some activation has "value" non-zero.
    contributors_to_influence_dict = {}
    for var_id in range(len(rule_body)):
        if rule_body.features[var_id].is_empty():
            continue
        temp_tree, temp_id_to_var_id = rule_body.get_subtree_for(var_id)
        temp_id = temp_id_to_var_id.inverse[var_id]
        dim = rule_body.features[var_id].dimension
        empty_bitset = BitSet.from_subset(dim, set())
        base_tree = temp_tree
        for new_id in range(len(temp_tree)): # Label every other node on the path as empty, isolating this atom
            base_tree = base_tree.with_feature(new_id, empty_bitset)
        relevant_positions = var_layer_mask[(var_id,0)].elements()
        for pos in relevant_positions:
            labelled_tree = base_tree.with_feature(temp_id, BitSet.from_subset(dim, {pos})) # mask holds only relevant pos
            temp_cd_graph = labelled_tree.as_cd_graph # recall that this ensures that node id matches the tree's index
            trace = TraceCollector()
            apply_model(temp_cd_graph, fe.device, fe.model,trace)

            # Compute the influence value for this contributor, iterating over the computation tree
            contribution_score = 0
            current_variable_id = 0  # start from root variable
            next_var_id = next(iter(temp_tree.children[current_variable_id].values()), None) # Just get only child
            for l in range(fe.model.num_layers, -1, -1):
                # Identify the correct variable for this level: stay in current variable unless next variable matches l
                if next_var_id is not None and l == temp_tree.levels[next_var_id]:
                    current_variable_id = next_var_id
                    next_var_id = next(iter(temp_tree.children[current_variable_id].values()), None)
                rule_var_id = temp_id_to_var_id[current_variable_id]
                for k in var_layer_mask[rule_var_id, l].elements():
                    contribution_score += trace.activations[l][rule_var_id][k]
                if contribution_score > 0:
                    contributors_to_influence_dict[(var_id, pos)] = (l, contribution_score)
                    break
                contributors_to_influence_dict[(var_id, pos)] = (0, 0)

    # Next we sort contributors by contribution.
    contributors_list = []
    for (y, j), (s,l) in contributors_to_influence_dict.items():
        contributors_list.append((s, l, y, j))
    contributors_list = sorted(contributors_list)

    # Start trying rules from the empty rule, adding atoms in increasing order of contribution
    # We represent in two lists the nodes that we are including and their features that we are including in the optim
    selected_nodes = [0]
    empty_bitset = BitSet.from_subset(fe.model.layer_dimension(0),{}) # Remember this is immutable
    selected_features = [empty_bitset]

    while contributors_list:
        # Add the node and feature to the lists selected by the optimisation
        influence_score, layer, var_id, pos = contributors_list.pop()
        idx, is_new = find_index_and_insert(sorted_list=selected_nodes, element=var_id)
        if is_new:
            selected_features = (selected_features[:idx] + [empty_bitset] + selected_features[idx:])
        selected_features[idx] = selected_features[idx].add_element(pos)
        # Test the new version of the rule
        optimised_rule = rule_body.from_subtree(selected_nodes,selected_features)
        temp_cd_graph = optimised_rule.as_cd_graph
        output_graph = apply_model(temp_cd_graph, fe.device, fe.model)
        if output_graph.features[0][predicate_position] > fe.threshold:
            return optimised_rule
    raise AssertionError("This part of the code should not be reachable. There's a bug in Optimisation 1")

    # TODO ITP: maybe a clean-up step like in approximation 2 can help make improvements