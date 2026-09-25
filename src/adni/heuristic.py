from src.adni.signature import AdniSignature
from src.adni.sparse_triangular_matrix import SparseTriangularMatrix


# A cheap, model-only estimate of how much a matrix cell (i, j) = value contributes to the "positive"
# prediction: at each hidden position p that actually helps "positive", sum node_j's own-feature contribution
# to p via the layer-1 colour-`value` (matrices B) and node_i's own-feature contribution to p via its layer-1
# self-update (matrix_A), then weight each p by its part_of weight into "positive" and sum.
def heuristic_for_element(i: int, j: int, value: int, signature: AdniSignature, model) -> float:
    node_pos = signature.node_pos
    nodei_pos = signature.node_k_pos(i)
    nodej_pos = signature.node_k_pos(j)

    part_of_row = model.matrix_B(2, signature.part_of_colour)[signature.positive_pos, :]
    value_matrix = model.matrix_B(1, signature.value_colour(value))
    self_matrix = model.matrix_A(1)
    contribution = (
        value_matrix[:, node_pos] + value_matrix[:, nodej_pos]
        + self_matrix[:, node_pos] + self_matrix[:, nodei_pos]
    )
    relevant = part_of_row > 0
    return (part_of_row[relevant] * contribution[relevant]).sum().item()


# heuristic_for_element for every nonzero cell of `matrix`, keyed by its (i, j) position.
def heuristic_for_matrix(matrix: SparseTriangularMatrix, signature: AdniSignature, model) -> dict:
    return {
        (i, j): heuristic_for_element(i, j, value, signature, model)
        for (i, j), value in matrix.to_dict().items()
    }
