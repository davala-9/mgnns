from src.adni.matrix_to_tree import NODE_PREDICATE, PART_OF_PREDICATE, colour_predicate_for, node_predicate_for
from src.adni.sparse_triangular_matrix import SparseTriangularMatrix

POSITIVE_PREDICATE = "positive"


# A cheap, model-only estimate of how much a matrix cell (i, j) = value contributes to the "positive"
# prediction: at each hidden position p that actually helps "positive", sum node_j's own-feature contribution
# to p via the layer-1 colour-`value` (matrices B) and node_i's own-feature contribution to p via its layer-1
# self-update (matrix_A), then weight each p by its part_of weight into "positive" and sum.
def heuristic_for_element(i: int, j: int, value: int, internal_encoder, model) -> float:
    node_pos = internal_encoder.unary_pred_position_dict[NODE_PREDICATE]
    nodei_pos = internal_encoder.unary_pred_position_dict[node_predicate_for(i)]
    nodej_pos = internal_encoder.unary_pred_position_dict[node_predicate_for(j)]
    positive_pos = internal_encoder.unary_pred_position_dict[POSITIVE_PREDICATE]
    part_of_colour = internal_encoder.binary_pred_colour_dict[PART_OF_PREDICATE]
    value_colour = internal_encoder.binary_pred_colour_dict[colour_predicate_for(value)]

    part_of_row = model.matrix_B(2, part_of_colour)[positive_pos, :]
    value_matrix = model.matrix_B(1, value_colour)
    self_matrix = model.matrix_A(1)
    contribution = (
        value_matrix[:, node_pos] + value_matrix[:, nodej_pos]
        + self_matrix[:, node_pos] + self_matrix[:, nodei_pos]
    )
    relevant = part_of_row > 0
    return (part_of_row[relevant] * contribution[relevant]).sum().item()


# heuristic_for_element for every nonzero cell of `matrix`, keyed by its (i, j) position.
def heuristic_for_matrix(matrix: SparseTriangularMatrix, internal_encoder, model) -> dict:
    return {
        (i, j): heuristic_for_element(i, j, value, internal_encoder, model)
        for (i, j), value in matrix.to_dict().items()
    }
