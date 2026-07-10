from typing import Tuple

from src.utils.utils import TYPE_PRED

Fact = Tuple[str, str, str]

def is_var(x):
    return x.startswith("?")

# Gets an atom term, a constant, and a partial binding, and returns a binding(extension) trying to match them
def match_term(atom_term, fact_term, bindings):
    bindings_copy = dict(bindings)
    if is_var(atom_term):
        if atom_term in bindings_copy:
            if bindings_copy[atom_term] != fact_term:
                return None
            # else: the binding already matches; no need to add anything.
        else:  # no binding has been defined for the variable
            bindings_copy[atom_term] = fact_term
    else:  # the term is a constant
        if atom_term != fact_term:
            return None
        # else: constants match; no extension to the binding.
    return bindings_copy

# An atom is input as a pair, consisting of a predicate and a list with one or two arguments (of the atom).
# A fact is a triple of the form (subject, relation, object); relation might be TYPE_PRED if representing a unary atom
def match_atom(atom, fact, bindings):
    atom_predicate, atom_argument_list = atom
    fact_subject, fact_relation, fact_object = fact
    if fact_relation == TYPE_PRED: # UNARY FACT, so fact_object is the predicate
        if fact_object != atom_predicate: # no predicate match
            return None
        return match_term(atom_argument_list[0],fact_subject,bindings)
    else: # BINARY FACT, so fact_relation is the predicate
        if fact_relation != atom_predicate: # no predicate match
            return None
        binding_extension_1 = match_term(atom_argument_list[0],fact_subject,bindings)
        if binding_extension_1 is None:
            return None
        return match_term(atom_argument_list[1],fact_object,binding_extension_1)

# Splits the body of the rule into atoms (as strings). Does no operation other than the splitting.
def split_atoms(body):
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
def parse_atom(atom):
    atom = atom.strip()
    lt = atom.index('<')
    gt = atom.index('>')
    predicate = atom[lt+1:gt]
    args = atom[gt+1:].strip()
    assert args[0] == '['
    assert args[-1] == ']'
    args = args[1:-1]
    arguments = [a.strip() for a in args.split(',')]
    return predicate, arguments

# Parses a rule, returning an atom (as string) for the head, and a list of atoms (as string) for the body.
def parse_rule(rule):
    rule = rule.strip()
    if rule.endswith('.'):
        rule = rule[:-1]
    head_text, body_text = rule.split(":-", 1)
    head = parse_atom(head_text)
    body = [parse_atom(a) for a in split_atoms(body_text)]
    return head, body

# Tries to ground a term (i.e. variable or constant) with a given binding, which we assume covers this term
def ground_term(term, binding):
    if is_var(term):
        if term not in binding:
            raise KeyError(f"'{term}' not found in binding")
        return binding[term] # term is a variable, so we apply substitution
    else:
        return term  # term is a constant, so it appears as is


# Applies a given rule to a set of facts given as triples
# The body is a list of atoms, where each atom is represented as a pair of a predicate and a list of (1 or 2) arguments
# head is represented in the same way as atoms
def apply_rule(rule, facts):
    head, body = parse_rule(rule)

    bindings = [dict()] # Dict list, each dict a (partial) bindings of variables in the rule to constants in the dataset

    # First, list all constants (we need them for free head variables)
    constants = set()
    for fact in facts:
        a,b,c = fact
        constants.add(a)
        if b != TYPE_PRED:
            constants.add(c)

    # Then gather all valid body bindings
    for atom in body:
        new_bindings = []
        for b in bindings:
            for fact in facts:
                b2 = match_atom(atom, fact, b)
                if b2 is not None:
                    new_bindings.append(b2)
        bindings = new_bindings

    # Extend the bindings with any free head variables (remember, we DON'T have the usual Datalog safety condition here)
    head_predicate, head_terms = head
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
    results = set()
    if len(head_terms) == 1:
        for binding in bindings:
            results.add((ground_term(head_terms[0],binding), TYPE_PRED, head_predicate))
    else:
        for binding in bindings:
            results.add((ground_term(head_terms[0],binding),
                         head_predicate,
                         ground_term(head_terms[1],binding)))
    return results