from src.rule_extraction.typed_constraint import TypedConstraint

# A variable is either the ROOT of the conjunction, or any other (NONROOT) variable.
ROOT, NONROOT = "ROOT", "NONROOT"

# NONROOT's feature "node" is always one, "positive" is always zero, and exactly one of these must be one.
NONROOT_REGION_PREDICATES = [f"node_{i}" for i in range(90)]

# NONROOT-NONROOT edges: at most one of these colours.
NONROOT_NONROOT_COLOUR_PREDICATES = [f"{i}.0" for i in range(1, 11)]


# Builds the ADNI TypedConstraint. unary_pred_to_position and binary_pred_to_colour translate this
# dataset's unary (resp. binary) predicate names into the positions (resp. colours) used elsewhere.
def adni_constraint(unary_pred_to_position: dict, binary_pred_to_colour: dict) -> TypedConstraint:
    tc = TypedConstraint({ROOT, NONROOT})

    tc.set_features_always_zero(ROOT, *unary_pred_to_position.values())

    tc.set_features_always_one(NONROOT, unary_pred_to_position["node"])
    tc.set_features_always_zero(NONROOT, unary_pred_to_position["positive"])
    tc.set_features_one_of(NONROOT, *(unary_pred_to_position[p] for p in NONROOT_REGION_PREDICATES))

    for colour in binary_pred_to_colour.values():
        tc.set_edge_never_exists(ROOT, NONROOT, colour)

    tc.set_edge_always_one(NONROOT, ROOT, binary_pred_to_colour["part_of"])

    tc.set_edges_atmost_one_of(
        NONROOT, NONROOT, *(binary_pred_to_colour[p] for p in NONROOT_NONROOT_COLOUR_PREDICATES)
    )

    # A child is always NONROOT, regardless of its parent's type or the edge's colour.
    for colour in binary_pred_to_colour.values():
        tc.set_child_type(ROOT, colour, NONROOT)
        tc.set_child_type(NONROOT, colour, NONROOT)

    return tc
