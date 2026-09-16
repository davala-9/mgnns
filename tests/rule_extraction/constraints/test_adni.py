from src.rule_extraction.constraints.adni import (
    ROOT, NONROOT, NONROOT_REGION_PREDICATES, NONROOT_NONROOT_COLOUR_PREDICATES, adni_constraint,
)
from src.rule_extraction.typed_constraint import GroupedConstraints


def unary_translation():
    predicates = ["node", "positive"] + NONROOT_REGION_PREDICATES
    return {predicate: position for position, predicate in enumerate(predicates)}


def binary_translation():
    predicates = ["part_of", "0.0"] + NONROOT_NONROOT_COLOUR_PREDICATES
    return {predicate: colour for colour, predicate in enumerate(predicates)}


def test_types():
    tc = adni_constraint(unary_translation(), binary_translation())
    assert tc.types == {ROOT, NONROOT}


def test_root_has_every_feature_always_zero():
    unary = unary_translation()
    tc = adni_constraint(unary, binary_translation())
    assert tc.get_feature_constraints(ROOT).always_zero == set(unary.values())


def test_nonroot_feature_constraints():
    unary = unary_translation()
    tc = adni_constraint(unary, binary_translation())
    constraints = tc.get_feature_constraints(NONROOT)
    assert constraints.always_one == {unary["node"]}
    assert constraints.always_zero == {unary["positive"]}
    assert constraints.one_of == [{unary[p] for p in NONROOT_REGION_PREDICATES}]


def test_root_nonroot_has_no_edges_allowed():
    binary = binary_translation()
    tc = adni_constraint(unary_translation(), binary)
    assert tc.get_edge_constraints(ROOT, NONROOT).always_zero == set(binary.values())


def test_nonroot_root_has_exactly_one_part_of_edge():
    binary = binary_translation()
    tc = adni_constraint(unary_translation(), binary)
    assert tc.get_edge_constraints(NONROOT, ROOT).always_one == {binary["part_of"]}


def test_nonroot_nonroot_has_atmost_one_of_the_numeric_colours():
    binary = binary_translation()
    tc = adni_constraint(unary_translation(), binary)
    constraints = tc.get_edge_constraints(NONROOT, NONROOT)
    assert constraints.atmost_one_of == [{binary[p] for p in NONROOT_NONROOT_COLOUR_PREDICATES}]
    # "0.0" is deliberately excluded from the at-most-one-of set.
    assert binary["0.0"] not in constraints.atmost_one_of[0]


def test_unmentioned_type_pairs_have_no_constraints():
    tc = adni_constraint(unary_translation(), binary_translation())
    assert tc.get_edge_constraints(ROOT, ROOT) == GroupedConstraints()
    assert tc.get_edge_constraints(NONROOT, NONROOT).always_one == set()
    assert tc.get_edge_constraints(NONROOT, NONROOT).always_zero == set()
