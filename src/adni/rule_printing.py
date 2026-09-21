from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction


# Converts a tree built by matrix_to_tree into a printable Datalog rule "<head>[?X0] :- body atoms .",
# in the same "<predicate>[?vars]" style as full_program.unfold_and_print_rule. Written from scratch
# rather than reusing identity.IdentityEncoderDecoder.unfold_all: that function assigns each var a fresh
# data variable the first time it's discovered as a *child*, which assumes every var has exactly one
# parent -- true for the general (compute_tree_for) case, but not for matrix_to_tree's trees, where a
# matrix entry (i, j, value) wires node_j as an extra child of node_i on top of its part_of edge from
# root, so it would get renamed partway through and split into two disconnected variables. Assigning
# each var_id its own data variable upfront, once, sidesteps that entirely.
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
