from src.adni.sparse_triangular_matrix import SparseTriangularMatrix
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, TreeShapedConjunctionBuilder
from src.utils.bitset import BitSet

NODE_PREDICATE = "node"
PART_OF_PREDICATE = "part_of"


def node_predicate_for(k: int) -> str:
    return f"node_{k}"


def colour_predicate_for(value: int) -> str:
    return f"{value}.0"


# Recovers (d, max_value) from an internal_encoder that already declares an ADNI signature: d is how
# many node_k unary predicates it has (0, 1, 2, ... until one is missing), max_value is how many
# colour_predicate_for(v) binary predicates it has (1, 2, ... until one is missing).
def infer_dimensions(internal_encoder) -> tuple[int, int]:
    d = 0
    while node_predicate_for(d) in internal_encoder.unary_pred_position_dict:
        d += 1
    max_value = 0
    while colour_predicate_for(max_value + 1) in internal_encoder.binary_pred_colour_dict:
        max_value += 1
    return d, max_value


# A TreeShapedConjunction is a tree: children[parent][edge] = child produces the real-world fact
# (child, colour_predicate, parent) -- messages flow child -> parent, matching how compute_tree_for
# treats "child" as relevance flowing backward from what a node receives (see as_cd_graph and
# CanonicalEncoderDecoder.encode_dataset, whose (subject, colour, object) edges land the same way).
#
# included_nodes restricts which of the matrix's 0..d-1 nodes actually become part_of children of the
# root (default: all of them). Meant for dropping nodes that turned out to have no incident matrix
# entry at all once a search has minimised the matrix down: such a node contributes nothing but its own
# (unconditional) part_of self-loop, so leaving it out entirely can shrink the tree further than
# minimising the matrix's cells alone ever could. Any matrix entry touching an excluded node is skipped;
# in the intended (isolated-node) use this never actually happens, since an included node's own matrix
# entries are exactly what would have kept it out of `included_nodes` in the first place.
def matrix_to_tree(matrix: SparseTriangularMatrix, internal_encoder, included_nodes=None) -> TreeShapedConjunction:
    if included_nodes is None:
        included_nodes = range(matrix.d)
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

    # Every included node_k is a child of the root via part_of, and is itself both "node" and "node_k".
    part_of_colour = binary_colours[PART_OF_PREDICATE]
    node_var_ids = {}
    for k in included_nodes:
        features = BitSet.from_subset(n_unary, {unary_positions[NODE_PREDICATE], unary_positions[node_predicates[k]]})
        var_id = builder.add(features=features, level=0, parent=root_id, edge=(1, part_of_colour, k))
        node_var_ids[k] = var_id

    # Each sparse-matrix entry (i, j, value) becomes the fact (node_j, colour_predicate(value), node_i):
    # node_j and node_i both already exist (added above), so this wires an extra edge directly into the
    # builder's children dict rather than through add() (which only ever spawns a brand-new node).
    for (i, j), value in matrix.to_dict().items():
        if i not in node_var_ids or j not in node_var_ids:
            continue
        colour = binary_colours[colour_predicate_for(value)]
        builder.children[node_var_ids[i]][(0, colour, j)] = node_var_ids[j]

    return builder.build()
