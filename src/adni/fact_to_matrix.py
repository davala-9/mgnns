import numpy as np

from src.adni.matrix_search import minimise
from src.adni.matrix_to_tree import is_sound, prune_isolated_nodes
from src.adni.signature import AdniSignature
from src.adni.sparse_triangular_matrix import SparseTriangularMatrix
from src.rule_extraction.fact_explanation import FactContext
from src.utils.bitset import BitSet
from src.utils.utils import backpropagate_relevance

"""This module is responsible for implementing an adni-specific version of fact explanations.
It works analogously to fact_explanation, except that instead of building a TreeShapedConjunction, 
it builds a SparseTriangularMatrix. See more information in SparseTriangularMatrix"""


# Writes `value` into matrix cell (i, j), raising exception if that cell already holds a *different* value.
# matrix_to_tree's (i, j) with i <= j always means "node_j sends to node_i" but adni datasets are symmetric so it's
# always written at (min, max). Note also ADNI has at most one edge between two brain regions, so no issue.
def _set_matrix_entry(matrix: SparseTriangularMatrix, k1: int, k2: int, value: int) -> SparseTriangularMatrix:
    i, j = (k1, k2) if k1 <= k2 else (k2, k1)
    existing = matrix.get(i, j)
    if existing != 0 and existing != value:
        raise ValueError(f"conflicting values for matrix cell ({i}, {j}): already {existing}, now {value}")
    return matrix.with_value(i, j, value)


# For each position in `positions`, get the single neighbour whose entry in `neighbour_vectors` is maximal at that
# position. Ties are broken by choosing a neighbour already in `chosen`. Returns {position: neighbour_real_idx}
def _pick_best_neighbours(positions, neighbours: list, neighbour_vectors: np.ndarray, chosen: set) -> dict:
    picks = {}
    for position in positions:
        column = neighbour_vectors[:, position]
        best_value = column.max()
        if best_value <= 0:
            continue
        tied = [idx for idx, value in zip(neighbours, column) if value == best_value]
        reused = [idx for idx in tied if idx in chosen]
        pick = min(reused) if reused else min(tied)
        chosen.add(pick)
        picks[position] = pick
    return picks


# Given a real "positive" prediction on one constant, this produces a SparseTriangularMatrix representing an explanatory
# rule. It works as follows:
#  1. From the predicted constant (root), find its part_of neighbours that actually drove "positive" at layer 2
#  2. For each such neighbour, find *its* neighbours via each colour that drove its own layer-1 activation.
def derive_matrix_from_fact(fact: tuple[str, str, str], trace, external_encoder, signature: AdniSignature, model,
                             threshold: float) -> SparseTriangularMatrix:
    cd_graph = trace.cd_graph
    activations = trace.activations
    constant_to_index = {name: i for i, name in enumerate(cd_graph.node_names)}
    fact_context = FactContext(fact, external_encoder, signature.internal_encoder, constant_to_index)
    assert activations[model.num_layers][fact_context.cd_fact_const_index][fact_context.cd_fact_pred_pos] > threshold, \
        "Error: the fact to be explained is not derived by the model on this dataset."

    L = model.num_layers
    root_idx = fact_context.cd_fact_const_index
    part_of_colour = signature.part_of_colour

    matrix = SparseTriangularMatrix.empty(signature.d, signature.max_value)

    # Step 1: root's part_of neighbours that drove "positive".
    mask_L = BitSet.from_subset(model.layer_dimension(L), {fact_context.cd_fact_pred_pos})
    positions_1 = backpropagate_relevance(mask_L, model.matrix_B(L, part_of_colour), previous_activations=None) \
        .elements()
    part_of_edges = cd_graph.edges[:, cd_graph.edge_colours == part_of_colour]
    root_neighbours = part_of_edges[:, part_of_edges[1] == root_idx][0].tolist()
    if not root_neighbours or not positions_1:
        return matrix

    neighbour_vectors_1 = np.array([activations[L - 1][n] for n in root_neighbours])
    chosen_children = set()
    child_picks = _pick_best_neighbours(positions_1, root_neighbours, neighbour_vectors_1, chosen_children)

    # A child reused for more than one position gets the union of those positions as its own layer-1 relevance mask.
    positions_by_child = {}
    for position, child_real_idx in child_picks.items():
        positions_by_child.setdefault(child_real_idx, set()).add(position)

    # Step 2: for each selected child, its own neighbours (via each colour value) that drove its layer-1
    # activation from their layer-0 feature vectors.
    for child_real_idx, own_positions in positions_by_child.items():
        child_k = signature.node_index_of(activations[0][child_real_idx])
        child_mask = BitSet.from_subset(model.layer_dimension(1), own_positions)
        chosen_grandchildren = set()
        for value in range(1, signature.max_value + 1):
            colour = signature.value_colour(value)
            positions_0 = backpropagate_relevance(child_mask, model.matrix_B(1, colour), previous_activations=None) \
                .elements()
            if not positions_0:
                continue
            colour_edges = cd_graph.edges[:, cd_graph.edge_colours == colour]
            neighbours = colour_edges[:, colour_edges[1] == child_real_idx][0].tolist()
            if not neighbours:
                continue
            neighbour_vectors_0 = np.array([activations[0][n] for n in neighbours])
            picks = _pick_best_neighbours(positions_0, neighbours, neighbour_vectors_0, chosen_grandchildren)
            for neighbour_real_idx in set(picks.values()):
                neighbour_k = signature.node_index_of(activations[0][neighbour_real_idx])
                matrix = _set_matrix_entry(matrix, child_k, neighbour_k, value)

    return matrix


# Shrinks derive_matrix_from_fact's raw matrix down to an actually-minimal sound rule
def derive_minimal_matrix_from_fact(fact: tuple[str, str, str], trace, external_encoder, signature: AdniSignature,
                                     model, threshold: float, device) -> tuple[SparseTriangularMatrix, set]:
    matrix = derive_matrix_from_fact(fact, trace, external_encoder, signature, model, threshold)

    constant_to_index = {name: i for i, name in enumerate(trace.cd_graph.node_names)}
    fact_context = FactContext(fact, external_encoder, signature.internal_encoder, constant_to_index)
    position = fact_context.cd_fact_pred_pos

    def check_soundness_with_nodes(candidate: SparseTriangularMatrix, included_nodes) -> bool:
        return is_sound(candidate, signature, model, device, position, threshold, included_nodes)

    def check_soundness(candidate: SparseTriangularMatrix) -> bool:
        return is_sound(candidate, signature, model, device, position, threshold)

    matrix = minimise(matrix, signature, model, check_soundness)
    included_nodes = prune_isolated_nodes(matrix, check_soundness_with_nodes)
    return matrix, included_nodes
