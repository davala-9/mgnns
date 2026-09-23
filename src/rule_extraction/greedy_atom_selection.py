from src.model.gnn_transformation import apply_model
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction
from src.utils.bitset import BitSet
from src.utils.utils import find_index_and_insert


# The final stage shared by Optimisations 1 and 2: starting from the empty rule body (root only, no atoms),
# add the atoms (var_id, pos) of rule_body one at a time in the given order, and return the first resulting
# sub-rule on which the model derives predicate_position at the root. Returns None if no prefix is sound.
def add_atoms_until_sound(rule_body: TreeShapedConjunction, atoms_in_order, device, model, threshold,
                          predicate_position: int):
    # selected_nodes is kept sorted; selected_features[i] is the feature BitSet of selected_nodes[i].
    selected_nodes = [0]
    empty_bitset = BitSet.from_subset(model.layer_dimension(0), set())
    selected_features = [empty_bitset]

    for var_id, pos in atoms_in_order:
        idx, is_new = find_index_and_insert(sorted_list=selected_nodes, element=var_id)
        if is_new:
            selected_features.insert(idx, empty_bitset)
        selected_features[idx] = selected_features[idx].add_element(pos)
        candidate_rule = rule_body.from_subtree(selected_nodes, selected_features)
        output_graph = apply_model(candidate_rule.as_cd_graph, device, model)
        if output_graph.features[0][predicate_position] > threshold:
            return candidate_rule
    return None
