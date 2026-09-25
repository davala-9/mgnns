from typing import Callable

from src.adni.signature import AdniSignature
from src.adni.sparse_triangular_matrix import SparseTriangularMatrix
from src.model.gnn_transformation import apply_model
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, TreeShapedConjunctionBuilder
from src.utils.bitset import BitSet

"""See the explanation at SparseTriangularMatrix to understand what each such matrix represents and why"""
def matrix_to_tree(matrix: SparseTriangularMatrix, signature: AdniSignature, included_nodes=None) -> TreeShapedConjunction:
    if (matrix.d, matrix.max_value) != (signature.d, signature.max_value):
        raise ValueError(f"matrix is {matrix.d}x{matrix.d} with values up to {matrix.max_value}, but the signature "
                         f"has d={signature.d} and max_value={signature.max_value}")
    if included_nodes is None:
        included_nodes = range(matrix.d)
    n_unary = signature.internal_encoder.get_n_unary_predicates()
    n_colours = signature.internal_encoder.get_n_binary_predicates()

    builder = TreeShapedConjunctionBuilder(n_colours)
    # The root is not an instance of any unary predicate -- it's the constant the matrix describes.
    root_id = builder.add(features=BitSet.from_subset(n_unary, set()), level=1, parent=-1)

    # Every included node_k is a child of the root via part_of, and is itself both "node" and "node_k".
    node_var_ids = {}
    for k in included_nodes:
        features = BitSet.from_subset(n_unary, {signature.node_pos, signature.node_k_pos(k)})
        var_id = builder.add(features=features, level=0, parent=root_id, edge=(1, signature.part_of_colour, k))
        node_var_ids[k] = var_id

    # Each sparse-matrix entry (i, j, value) becomes the fact (node_j, colour_predicate(value), node_i):
    # node_j and node_i both already exist (added above), so this wires an extra edge directly into the
    # builder's children dict rather than through add() (which only ever spawns a brand-new node).
    for (i, j), value in matrix.to_dict().items():
        if i not in node_var_ids or j not in node_var_ids:
            continue
        builder.children[node_var_ids[i]][(0, signature.value_colour(value), j)] = node_var_ids[j]

    return builder.build()


# Whether the model, applied to matrix_to_tree(matrix, signature, included_nodes), derives the unary
# predicate at `position` for the root with a score above `threshold`.
def is_sound(matrix: SparseTriangularMatrix, signature: AdniSignature, model, device, position: int, threshold: float,
             included_nodes=None) -> bool:
    tree = matrix_to_tree(matrix, signature, included_nodes=included_nodes)
    output_graph = apply_model(tree.as_cd_graph, device, model)
    return output_graph.features[0][position].item() > threshold


# this removes nodes that do not participate in the connections captured in the matrix from the tree that we build.
# It's an optimisation that greatly reduces the size of the tree.
# Returns the set of node indices that should stay in the final tree.
def prune_isolated_nodes(matrix: SparseTriangularMatrix,
                          check_soundness_with_nodes: Callable[[SparseTriangularMatrix, set], bool],
                          verbose: bool = False) -> set:
    entries = matrix.to_dict()
    touched = {k for (i, j) in entries for k in (i, j)}
    isolated = sorted(set(range(matrix.d)) - touched)
    included = set(range(matrix.d))
    for k in isolated:
        candidate = included - {k}
        if check_soundness_with_nodes(matrix, candidate):
            included = candidate
            if verbose:
                print(f"Dropped isolated node {k}: still sound without it.")
    return included
