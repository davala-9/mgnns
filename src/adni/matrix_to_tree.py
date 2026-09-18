from src.adni.sparse_triangular_matrix import SparseTriangularMatrix
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, TreeShapedConjunctionBuilder
from src.utils.bitset import BitSet

NODE_PREDICATE = "node"
PART_OF_PREDICATE = "part_of"


def node_predicate_for(k: int) -> str:
    return f"node_{k}"


def colour_predicate_for(value: int) -> str:
    return f"{value}.0"


# A TreeShapedConjunction is a tree: children[parent][edge] = child produces the real-world fact
# (child, colour_predicate, parent) -- messages flow child -> parent, matching how compute_tree_for
# treats "child" as relevance flowing backward from what a node receives (see as_cd_graph and
# CanonicalEncoderDecoder.encode_dataset, whose (subject, colour, object) edges land the same way).
def matrix_to_tree(matrix: SparseTriangularMatrix, internal_encoder) -> TreeShapedConjunction:
    unary_positions = internal_encoder.unary_pred_position_dict
    binary_colours = internal_encoder.binary_pred_colour_dict
    n_unary = internal_encoder.get_n_unary_predicates()
    n_colours = internal_encoder.get_n_binary_predicates()

    node_predicates = [node_predicate_for(k) for k in range(matrix.d)]
    colour_predicates = [colour_predicate_for(v) for v in range(1, matrix.max_value + 1)]

    missing_unary = {NODE_PREDICATE, *node_predicates} - set(unary_positions)
    if missing_unary:
        raise ValueError(f"internal_encoder is missing unary predicates: {sorted(missing_unary)}")
    missing_binary = {PART_OF_PREDICATE, *colour_predicates} - set(binary_colours)
    if missing_binary:
        raise ValueError(f"internal_encoder is missing binary predicates: {sorted(missing_binary)}")

    builder = TreeShapedConjunctionBuilder(n_colours)
    # The root is not an instance of any unary predicate -- it's the constant the matrix describes.
    root_id = builder.add(features=BitSet.from_subset(n_unary, set()), level=1, parent=-1)

    # Every node_k is a child of the root via part_of, and is itself both "node" and "node_k".
    part_of_colour = binary_colours[PART_OF_PREDICATE]
    node_var_ids = []
    for k in range(matrix.d):
        features = BitSet.from_subset(n_unary, {unary_positions[NODE_PREDICATE], unary_positions[node_predicates[k]]})
        var_id = builder.add(features=features, level=0, parent=root_id, edge=(1, part_of_colour, k))
        node_var_ids.append(var_id)

    # Each sparse-matrix entry (i, j, value) becomes the fact (node_j, colour_predicate(value), node_i):
    # node_j and node_i both already exist (added above), so this wires an extra edge directly into the
    # builder's children dict rather than through add() (which only ever spawns a brand-new node).
    for (i, j), value in matrix.to_dict().items():
        colour = binary_colours[colour_predicate_for(value)]
        builder.children[node_var_ids[i]][(0, colour, j)] = node_var_ids[j]

    return builder.build()
