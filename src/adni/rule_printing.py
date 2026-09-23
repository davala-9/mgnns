from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction


# Converts a tree built by matrix_to_tree into a printable Datalog rule "<head>[?X0] :- body atoms .".
# TODO: see if we can merge with the rule printer in TreeShapedConjunction
def tree_to_rule(tree: TreeShapedConjunction, internal_encoder, head_predicate: str) -> str:
    datavar = [f"X{var_id}" for var_id in range(len(tree))]
    body_atoms = []
    for var_id in range(len(tree)):
        for feat in tree.features[var_id].elements():
            predicate = internal_encoder.unary_pred_position_dict.inverse[feat]
            body_atoms.append(f"<{predicate}>[?{datavar[var_id]}]")
        for (_, colour, _), child_id in tree.children[var_id].items():
            predicate = internal_encoder.binary_pred_colour_dict.inverse[colour]
            body_atoms.append(f"<{predicate}>[?{datavar[child_id]},?{datavar[var_id]}]")
    head = f"<{head_predicate}>[?{datavar[0]}]"
    return head + " :- " + ", ".join(body_atoms) + " ."
