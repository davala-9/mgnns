import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from src.utils.utils import TYPE_PRED

Fact = Tuple[str, str, str]
Atom = Tuple[str, List[str]]  # (predicate, arguments)
Binding = Dict[str, str]


class UnsafeRuleWarning(UserWarning):
    """A rule's head contains a variable that isn't bound by its body.

    This engine intentionally allows such rules (no Datalog safety condition):
    the free head variable is grounded over every constant in the dataset. This
    warning just flags when that's happening, since it can be expensive and is
    sometimes a sign of a rule-authoring mistake.
    """


def is_var(x: str) -> bool:
    return x.startswith("?")

# Gets an atom term, a constant, and a partial binding, and returns a binding(extension) trying to match them
def match_term(atom_term: str, fact_term: str, bindings: Binding) -> Optional[Binding]:
    if is_var(atom_term):
        existing = bindings.get(atom_term)
        if existing is not None:
            return bindings if existing == fact_term else None # None: if binding exists but clashes with fact term
        # no binding has been defined for the variable: extend it
        bindings_copy = dict(bindings)
        bindings_copy[atom_term] = fact_term
        return bindings_copy
    else:  # the term is a constant
        return bindings if atom_term == fact_term else None

# An atom is input as a pair, consisting of a predicate and a list with one or two arguments (of the atom).
# A fact is a triple of the form (subject, relation, object); relation might be TYPE_PRED if representing a unary atom
def match_atom(atom: Atom, fact: Fact, bindings: Binding) -> Optional[Binding]:
    atom_predicate, atom_argument_list = atom
    if len(atom_argument_list) not in (1, 2):
        raise ValueError(f"atom must have 1 or 2 arguments, got {len(atom_argument_list)}: {atom!r}")
    fact_subject, fact_relation, fact_object = fact
    if fact_relation == TYPE_PRED: # UNARY FACT, so fact_object is the predicate
        if fact_object != atom_predicate: # no predicate match
            return None
        return match_term(atom_argument_list[0], fact_subject, bindings)
    else: # BINARY FACT, so fact_relation is the predicate
        if fact_relation != atom_predicate: # no predicate match
            return None
        binding_extension_1 = match_term(atom_argument_list[0], fact_subject, bindings)
        if binding_extension_1 is None:
            return None
        return match_term(atom_argument_list[1], fact_object, binding_extension_1)

# Splits the body of the rule into atoms (as strings). Does no operation other than the splitting.
def split_atoms(body: str) -> List[str]:
    atoms = []
    start = 0
    depth = 0
    for i, c in enumerate(body):
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
        elif c == ',' and depth == 0:
            atoms.append(body[start:i].strip())
            start = i + 1
    atoms.append(body[start:].strip())
    return atoms

# Parses an atom as it appears in the rule, and returns a predicate and a list of one or two arguments
def parse_atom(atom: str) -> Atom:
    original = atom
    atom = atom.strip()
    if '<' not in atom or '>' not in atom:
        raise ValueError(f"malformed atom, expected '<predicate>[args]': {original!r}")
    lt = atom.index('<')
    gt = atom.index('>')
    predicate = atom[lt+1:gt]
    args = atom[gt+1:].strip()
    if not args.startswith('[') or not args.endswith(']'):
        raise ValueError(f"malformed atom, expected arguments in brackets: {original!r}")
    args = args[1:-1].strip()
    arguments = [a.strip() for a in args.split(',')] if args else []
    if len(arguments) not in (1, 2):
        raise ValueError(f"atom must have 1 or 2 arguments, got {len(arguments)}: {original!r}")
    return predicate, arguments

# Parses a rule, returning an atom (as string) for the head, and a list of atoms (as string) for the body.
def parse_rule(rule: str) -> Tuple[Atom, List[Atom]]:
    original = rule
    rule = rule.strip()
    if rule.endswith('.'):
        rule = rule[:-1]
    if ":-" not in rule:
        raise ValueError(f"malformed rule, expected a ':-' separator: {original!r}")
    head_text, body_text = rule.split(":-", 1)
    try:
        head = parse_atom(head_text)
    except ValueError as e:
        raise ValueError(f"malformed head in rule {original!r}: {e}") from e
    body_text = body_text.strip()
    body = []
    if body_text:
        for atom_text in split_atoms(body_text):
            try:
                body.append(parse_atom(atom_text))
            except ValueError as e:
                raise ValueError(f"malformed body in rule {original!r}: {e}") from e
    return head, body

# Tries to ground a term (i.e. variable or constant) with a given binding, which we assume covers this term
def ground_term(term: str, binding: Binding) -> str:
    if is_var(term):
        if term not in binding:
            raise KeyError(f"'{term}' not found in binding")
        return binding[term] # term is a variable, so we apply substitution
    else:
        return term  # term is a constant, so it appears as is


# Applies a given rule to a set of facts given as triples
# The body is a list of atoms, where each atom is represented as a pair of a predicate and a list of (1 or 2) arguments
# head is represented in the same way as atoms
def apply_rule(rule: str, facts: Set[Fact]) -> Set[Fact]:
    head, body = parse_rule(rule)
    head_predicate, head_terms = head

    # Warn (but don't refuse) if the head has a variable that's not bound by the body:
    # such rules are allowed here, but they ground that variable over every constant
    # in the dataset, which can be both surprising and expensive.
    body_variables = {term for _, arguments in body for term in arguments if is_var(term)}
    unsafe_variables = [t for t in head_terms if is_var(t) and t not in body_variables]
    if unsafe_variables:
        warnings.warn(
            f"unsafe rule: head variable(s) {unsafe_variables} not bound by the body: {rule!r}",
            UnsafeRuleWarning,
            stacklevel=2,
        )

    bindings = [dict()] # Dict list, each dict a (partial) bindings of variables in the rule to constants in the dataset

    # First, list all constants (we need them for free head variables), and index
    # facts by predicate (unary/binary kept separate) so each atom below only scans
    # the facts it could possibly match, instead of the whole fact set.
    constants = set()
    unary_facts_by_predicate = defaultdict(list)
    binary_facts_by_predicate = defaultdict(list)
    for fact in facts:
        subject, relation, obj = fact
        constants.add(subject)
        if relation == TYPE_PRED:
            unary_facts_by_predicate[obj].append(fact)
        else:
            constants.add(obj)
            binary_facts_by_predicate[relation].append(fact)

    # Then gather all valid body bindings
    for atom in body:
        atom_predicate, atom_arguments = atom
        facts_by_predicate = unary_facts_by_predicate if len(atom_arguments) == 1 else binary_facts_by_predicate
        candidate_facts = facts_by_predicate.get(atom_predicate, [])
        new_bindings = []
        for b in bindings:
            for fact in candidate_facts:
                b2 = match_atom(atom, fact, b)
                if b2 is not None:
                    new_bindings.append(b2)
        bindings = new_bindings

    # Extend the bindings with any free head variables (remember, we DON'T have the usual Datalog safety condition here)
    for head_term in head_terms:
        if is_var(head_term):
            new_bindings = []
            for binding in bindings:
                if head_term not in binding: # Free variable, so we bind it in all possible ways
                   for constant in constants:
                        new_binding = dict(binding)
                        new_binding[head_term] = constant
                        new_bindings.append(new_binding)
                else: # We already have the variable binded
                    new_bindings.append(binding)
            bindings = new_bindings

    # Finally, return a fact for each possible distinct head binding
    if len(head_terms) not in (1, 2):
        raise ValueError(f"head atom must have 1 or 2 arguments, got {len(head_terms)}: {head_predicate!r}")
    results = set()
    if len(head_terms) == 1:
        for binding in bindings:
            results.add((ground_term(head_terms[0], binding), TYPE_PRED, head_predicate))
    else:
        for binding in bindings:
            results.add((ground_term(head_terms[0], binding),
                         head_predicate,
                         ground_term(head_terms[1], binding)))
    return results
