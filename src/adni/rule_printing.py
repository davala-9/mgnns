from src.adni.matrix_to_tree import matrix_to_tree
from src.adni.signature import AdniSignature
from src.adni.sparse_triangular_matrix import SparseTriangularMatrix
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


def matrix_to_rule(matrix: SparseTriangularMatrix, signature: AdniSignature, head_predicate: str) -> str:
    touched = {k for (i, j) in matrix.to_dict() for k in (i, j)}
    tree = matrix_to_tree(matrix, signature, included_nodes=sorted(touched))
    return tree_to_rule(tree, signature.internal_encoder, head_predicate)
