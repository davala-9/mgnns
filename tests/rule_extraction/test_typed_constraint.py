import pytest

from src.rule_extraction.typed_constraint import GroupedConstraints, TypedConstraint

ROOT, NONROOT = "ROOT", "NONROOT"


class TestGroupedConstraints:
    @pytest.mark.parametrize("field_name", ["atleast_one", "atmost_one", "always_one", "always_zero"])
    def test_fresh_instances_do_not_share_mutable_set_defaults(self, field_name):
        a = GroupedConstraints()
        b = GroupedConstraints()
        getattr(a, field_name).add("D")
        assert getattr(b, field_name) == set()

    @pytest.mark.parametrize("field_name", ["one_of", "atmost_one_of"])
    def test_fresh_instances_do_not_share_mutable_list_defaults(self, field_name):
        a = GroupedConstraints()
        b = GroupedConstraints()
        getattr(a, field_name).append({"D"})
        assert getattr(b, field_name) == []


class TestFeatureConstraints:
    def test_every_declared_type_starts_with_empty_constraints(self):
        tc = TypedConstraint({ROOT, NONROOT})
        assert tc.get_feature_constraints(ROOT) == GroupedConstraints()
        assert tc.get_feature_constraints(NONROOT) == GroupedConstraints()

    def test_setters_only_affect_their_own_type(self):
        tc = TypedConstraint({ROOT, NONROOT})
        tc.set_features_always_zero(ROOT, "C")
        assert tc.get_feature_constraints(ROOT).always_zero == {"C"}
        assert tc.get_feature_constraints(NONROOT).always_zero == set()

    def test_always_one_and_one_of_accumulate(self):
        tc = TypedConstraint({NONROOT})
        tc.set_features_always_one(NONROOT, "D")
        tc.set_features_one_of(NONROOT, "A", "B")
        tc.set_features_one_of(NONROOT, "E", "F")
        constraints = tc.get_feature_constraints(NONROOT)
        assert constraints.always_one == {"D"}
        assert constraints.one_of == [{"A", "B"}, {"E", "F"}]

    def test_setters_return_self_for_chaining(self):
        tc = TypedConstraint({ROOT})
        result = tc.set_features_always_zero(ROOT, "C").set_features_always_one(ROOT, "D")
        assert result is tc

    def test_unknown_type_raises(self):
        tc = TypedConstraint({ROOT})
        with pytest.raises(ValueError):
            tc.get_feature_constraints(NONROOT)


class TestEdgeConstraints:
    def test_unset_pair_starts_with_empty_constraints(self):
        tc = TypedConstraint({ROOT, NONROOT})
        assert tc.get_edge_constraints(ROOT, NONROOT) == GroupedConstraints()

    def test_atleast_one_and_atmost_one_setters_only_affect_their_own_type_pair(self):
        tc = TypedConstraint({ROOT, NONROOT})
        tc.set_edge_exists_atleast_one(ROOT, NONROOT, "part_of")
        tc.set_edge_exists_atmost_one(ROOT, NONROOT, "part_of")
        assert tc.get_edge_constraints(ROOT, NONROOT).atleast_one == {"part_of"}
        assert tc.get_edge_constraints(ROOT, NONROOT).atmost_one == {"part_of"}
        assert tc.get_edge_constraints(NONROOT, ROOT).atleast_one == set()
        assert tc.get_edge_constraints(NONROOT, ROOT).atmost_one == set()
        assert tc.get_edge_constraints(NONROOT, NONROOT).atleast_one == set()
        assert tc.get_edge_constraints(NONROOT, NONROOT).atmost_one == set()

    def test_always_one_setter_only_affects_its_own_type_pair(self):
        tc = TypedConstraint({ROOT, NONROOT})
        tc.set_edge_always_one(ROOT, NONROOT, "part_of")
        assert tc.get_edge_constraints(ROOT, NONROOT).always_one == {"part_of"}
        assert tc.get_edge_constraints(NONROOT, ROOT).always_one == set()
        assert tc.get_edge_constraints(NONROOT, NONROOT).always_one == set()

    def test_never_exists_and_one_of(self):
        tc = TypedConstraint({NONROOT})
        tc.set_edge_never_exists(NONROOT, NONROOT, "part_of")
        tc.set_edges_one_of(NONROOT, NONROOT, "linked_to", "other_pred")
        constraints = tc.get_edge_constraints(NONROOT, NONROOT)
        assert constraints.always_zero == {"part_of"}
        assert constraints.one_of == [{"linked_to", "other_pred"}]

    def test_atmost_one_of_accumulates(self):
        tc = TypedConstraint({NONROOT})
        tc.set_edges_atmost_one_of(NONROOT, NONROOT, "linked_to", "other_pred")
        tc.set_edges_atmost_one_of(NONROOT, NONROOT, "third_pred")
        constraints = tc.get_edge_constraints(NONROOT, NONROOT)
        assert constraints.atmost_one_of == [{"linked_to", "other_pred"}, {"third_pred"}]

    def test_repeated_lookup_returns_the_same_object(self):
        tc = TypedConstraint({ROOT, NONROOT})
        first = tc.get_edge_constraints(ROOT, NONROOT)
        first.always_one.add("part_of")
        second = tc.get_edge_constraints(ROOT, NONROOT)
        assert second is first
        assert second.always_one == {"part_of"}

    def test_unknown_type_raises(self):
        tc = TypedConstraint({ROOT})
        with pytest.raises(ValueError):
            tc.get_edge_constraints(ROOT, NONROOT)
