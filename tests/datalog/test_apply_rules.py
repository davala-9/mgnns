from src.datalog.apply_rules import (
    is_var, match_term, match_atom, split_atoms, parse_atom,
    parse_rule, ground_term, apply_rule,
)
from src.utils.utils import TYPE_PRED


# --- is_var ---

def test_is_var():
    assert is_var("?X0")
    assert not is_var("dog")


# --- match_term ---

def test_match_term_binds_new_var():
    assert match_term("?X", "dog", {}) == {"?X": "dog"}

def test_match_term_consistent_existing_var():
    assert match_term("?X", "dog", {"?X": "dog"}) == {"?X": "dog"}

def test_match_term_conflicting_existing_var():
    assert match_term("?X", "cat", {"?X": "dog"}) is None

def test_match_term_matching_constant():
    assert match_term("dog", "dog", {}) == {}

def test_match_term_mismatching_constant():
    assert match_term("dog", "cat", {}) is None

def test_match_term_does_not_mutate_input():
    bindings = {"?X": "dog"}
    match_term("?Y", "cat", bindings)
    assert bindings == {"?X": "dog"}


# --- match_atom ---

def test_match_atom_unary_match():
    atom = ("_dog", ["?X"])
    fact = ("rex", TYPE_PRED, "_dog")
    assert match_atom(atom, fact, {}) == {"?X": "rex"}

def test_match_atom_unary_predicate_mismatch():
    atom = ("_cat", ["?X"])
    fact = ("rex", TYPE_PRED, "_dog")
    assert match_atom(atom, fact, {}) is None

def test_match_atom_binary_match():
    atom = ("_likes", ["?X", "?Y"])
    fact = ("rex", "_likes", "fido")
    assert match_atom(atom, fact, {}) == {"?X": "rex", "?Y": "fido"}

def test_match_atom_binary_predicate_mismatch():
    atom = ("_hates", ["?X", "?Y"])
    fact = ("rex", "_likes", "fido")
    assert match_atom(atom, fact, {}) is None

def test_match_atom_repeated_variable_self_join_matches():
    # <_similar>[?X0,?X0] should only match facts where subject == object
    atom = ("_similar", ["?X0", "?X0"])
    fact = ("dog", "_similar", "dog")
    assert match_atom(atom, fact, {}) == {"?X0": "dog"}

def test_match_atom_repeated_variable_self_join_rejects_mismatch():
    # Regression test for the binding-threading bug: dog != cat must fail
    atom = ("_similar", ["?X0", "?X0"])
    fact = ("dog", "_similar", "cat")
    assert match_atom(atom, fact, {}) is None


# --- split_atoms ---

def test_split_atoms_simple():
    body = "<_a>[?X], <_b>[?X,?Y]"
    assert split_atoms(body) == ["<_a>[?X]", "<_b>[?X,?Y]"]

def test_split_atoms_ignores_commas_inside_brackets():
    body = "<_a>[?X0,?X1], <_b>[?X2]"
    assert split_atoms(body) == ["<_a>[?X0,?X1]", "<_b>[?X2]"]


# --- parse_atom / parse_rule ---

def test_parse_atom_binary():
    assert parse_atom("<_likes>[?X0,?X1]") == ("_likes", ["?X0", "?X1"])

def test_parse_atom_unary():
    assert parse_atom("<_dog>[?X0]") == ("_dog", ["?X0"])

def test_parse_rule():
    rule = "<_a>[?X0,?X1] :- <_b>[?X0,?X2], <_c>[?X2,?X1] ."
    head, body = parse_rule(rule)
    assert head == ("_a", ["?X0", "?X1"])
    assert body == [("_b", ["?X0", "?X2"]), ("_c", ["?X2", "?X1"])]


# --- ground_term ---

def test_ground_term_variable():
    assert ground_term("?X", {"?X": "dog"}) == "dog"

def test_ground_term_constant():
    assert ground_term("dog", {}) == "dog"

def test_ground_term_unbound_variable_raises():
    try:
        ground_term("?X", {})
        assert False, "expected KeyError"
    except KeyError:
        pass


# --- apply_rule ---
# apply_rule now takes the rule as an unparsed string and parses it internally.

def test_apply_rule_simple_join_binary_head():
    rule = "<_grandparent>[?X,?Z] :- <_parent>[?X,?Y], <_parent>[?Y,?Z] ."
    facts = {("a", "_parent", "b"), ("b", "_parent", "c"), ("c", "_parent", "d")}
    result = apply_rule(rule, facts)
    assert result == {("a", "_grandparent", "c"), ("b", "_grandparent", "d")}

def test_apply_rule_unary_head():
    rule = "<_person>[?X] :- <_parent>[?X,?Y] ."
    facts = {("a", "_parent", "b"), ("b", "_parent", "c")}
    result = apply_rule(rule, facts)
    assert result == {("a", TYPE_PRED, "_person"), ("b", TYPE_PRED, "_person")}

def test_apply_rule_free_head_variable_grounds_with_every_constant():
    # ?Y is unsafe/free (doesn't appear in the body), so it ranges over all constants
    rule = "<_thing>[?X,?Y] :- <_dog>[?X] ."
    facts = {("rex", TYPE_PRED, "_dog"), ("rex", "_likes", "fido")}
    result = apply_rule(rule, facts)
    # constants present: rex, fido
    assert result == {("rex", "_thing", "rex"), ("rex", "_thing", "fido")}

def test_apply_rule_no_matching_facts_yields_empty():
    rule = "<_x>[?X,?Y] :- <_missing>[?X,?Y] ."
    facts = {("a", "_parent", "b")}
    assert apply_rule(rule, facts) == set()
