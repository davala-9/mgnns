from dataclasses import dataclass, field


# For a given type (resp. pair of types) this dataclass just wraps together all its feature (resp. edge) constraints.
# When this is used for feature (resp. edge) constraints, it lists predicates (resp. colours).
@dataclass
class GroupedConstraints:
    atleast_one: set = field(default_factory=set)
    atmost_one: set = field(default_factory=set)
    always_one: set = field(default_factory=set)
    always_zero: set = field(default_factory=set)
    one_of: list = field(default_factory=list)  # list[set]
    atmost_one_of: list = field(default_factory=list)  # list[set]

# A typed constraint assigns a type to each node. Then it provides the following information:
# -- Which feature constraints affect each node type
# -- Which edge constraints affect each pair of node types
# Note that instances of this class are mutable!
class TypedConstraint:

    def __init__(self, types: set[str]):
        self.types = set(types)
        self._feature_constraints = {t: GroupedConstraints() for t in self.types}
        self._edge_constraints: dict[tuple[str, str], GroupedConstraints] = {}

    def _check_type(self, t):
        if t not in self.types:
            raise ValueError(f"{t!r} is not one of this constraint's declared types {self.types}")

    ### Feature constraints: each node type has a GroupedConstraint object representing its feature constraints

    def get_feature_constraints(self, node_type) -> GroupedConstraints:
        self._check_type(node_type)
        return self._feature_constraints[node_type]

    # These features are always 1
    def set_features_always_one(self, node_type, *predicates) -> "TypedConstraint":
        self.get_feature_constraints(node_type).always_one.update(predicates)
        return self

    # These features are always 0
    def set_features_always_zero(self, node_type, *predicates) -> "TypedConstraint":
        self.get_feature_constraints(node_type).always_zero.update(predicates)
        return self

    # Exactly one of these features is one
    def set_features_one_of(self, node_type, *predicates) -> "TypedConstraint":
        self.get_feature_constraints(node_type).one_of.append(set(predicates))
        return self

    ### Edge constraints: each pair of node types as a GroupedConstraint representing edges allowed between

    def get_edge_constraints(self, from_type, to_type) -> GroupedConstraints:
        self._check_type(from_type)
        self._check_type(to_type)
        return self._edge_constraints.setdefault((from_type, to_type), GroupedConstraints())

    # Node of from_type is connected to at least one node of to_type via edge of colour colour
    def set_edge_exists_atleast_one(self, from_type, to_type, colour) -> "TypedConstraint":
        self.get_edge_constraints(from_type, to_type).atleast_one.add(colour)
        return self

    # Node of from_type is connected to at most one node of to_type via edge of colour colour
    def set_edge_exists_atmost_one(self, from_type, to_type, colour) -> "TypedConstraint":
        self.get_edge_constraints(from_type, to_type).atmost_one.add(colour)
        return self

    # Each node of from_type is connected to each node of to_type via edge of colour colour
    def set_edge_always_one(self, from_type, to_type, colour) -> "TypedConstraint":
        self.get_edge_constraints(from_type, to_type).always_one.add(colour)
        return self

    # No node of from_type is connected to a node of to_type via edge of colour colour
    def set_edge_never_exists(self, from_type, to_type, colour) -> "TypedConstraint":
        self.get_edge_constraints(from_type, to_type).always_zero.add(colour)
        return self

    # Each node of from_type is connected to each node of to_type via exactly one edge among these colours
    def set_edges_one_of(self, from_type, to_type, *colours) -> "TypedConstraint":
        self.get_edge_constraints(from_type, to_type).one_of.append(set(colours))
        return self

    # A node of from_type is connected to a node of to_type via no more than one edge among these colours
    def set_edges_atmost_one_of(self, from_type, to_type, *colours) -> "TypedConstraint":
        self.get_edge_constraints(from_type, to_type).atmost_one_of.append(set(colours))
        return self

