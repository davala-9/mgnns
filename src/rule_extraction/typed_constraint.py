from dataclasses import dataclass, field


# For a given type (resp. pair of types) this dataclass just wraps together all its feature (resp. edge) constraints.
@dataclass
class GroupedConstraints:
    always_one: set = field(default_factory=set)
    always_zero: set = field(default_factory=set)
    one_of: list = field(default_factory=list)  # list[set]


# A typed constraint assigns a type to each node. Then it provides the following information:
# -- Which feature constaints affect each node type
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

    # Feature constraints: each node type has a GroupedConstraint object representing its feature constraints

    def get_feature_constraints(self, node_type) -> GroupedConstraints:
        self._check_type(node_type)
        return self._feature_constraints[node_type]

    def set_features_always_one(self, node_type, *predicates) -> "TypedConstraint":
        self.get_feature_constraints(node_type).always_one.update(predicates)
        return self

    def set_features_always_zero(self, node_type, *predicates) -> "TypedConstraint":
        self.get_feature_constraints(node_type).always_zero.update(predicates)
        return self

    def set_features_one_of(self, node_type, *predicates) -> "TypedConstraint":
        self.get_feature_constraints(node_type).one_of.append(set(predicates))
        return self

    # Edge constraints: each pair of node types as a GroupedConstraint representing edges allowed between

    def set_edge_always_exists(self, from_type, to_type, colour) -> "TypedConstraint":
        self.get_edge_constraints(from_type, to_type).always_one.add(colour)
        return self

    def set_edge_never_exists(self, from_type, to_type, colour) -> "TypedConstraint":
        self.get_edge_constraints(from_type, to_type).always_zero.add(colour)
        return self

    def set_edges_one_of(self, from_type, to_type, *colours) -> "TypedConstraint":
        self.get_edge_constraints(from_type, to_type).one_of.append(set(colours))
        return self

    def get_edge_constraints(self, from_type, to_type) -> GroupedConstraints:
        self._check_type(from_type)
        self._check_type(to_type)
        return self._edge_constraints.setdefault((from_type, to_type), GroupedConstraints())
