import re

"""This module is the interface between the internal encoding of an ADNI dataset and the rest of the adni code.
An internal encoder declares an ADNI signature if it has:
---unary predicates "node", "positive" and node_0, ..., node_{d-1} (one per brain region), for some d >= 1;
---binary predicates "part_of" and 1.0, ..., {max_value}.0 (one per edge colour value), for some max_value >= 1.
Specific assumptions on how ADNI datasets look like are described in SparseTriangularMatrix.
Throughout this module, it helps to consider that k always ranges across the number of brain regions (0 to d-1)"""


NODE_PREDICATE = "node"
POSITIVE_PREDICATE = "positive"
PART_OF_PREDICATE = "part_of"

_NODE_K_PATTERN = re.compile(r"node_(\d+)") # unary predicates of the form node_[number]
_COLOUR_PATTERN = re.compile(r"(\d+)\.0") # binary predicates of the form [number].0


def node_predicate_for(k: int) -> str:
    return f"node_{k}"


def colour_predicate_for(value: int) -> str:
    return f"{value}.0"


# The numbers n for which some predicate in `names` fully matches `pattern` with n as its group.
def _numbers_matching(pattern: re.Pattern, names) -> set:
    return {int(match.group(1)) for match in map(pattern.fullmatch, names) if match}


class AdniSignature:
    # Checks that `internal_encoder` declares an ADNI signature and recovers d and max_value from it.
    def __init__(self, internal_encoder):
        self.internal_encoder = internal_encoder

        node_ks = _numbers_matching(_NODE_K_PATTERN, internal_encoder.unary_pred_position_dict)
        values = _numbers_matching(_COLOUR_PATTERN, internal_encoder.binary_pred_colour_dict)
        self.d = len(node_ks)
        self.max_value = len(values)

        problems = []
        missing_unary = {NODE_PREDICATE, POSITIVE_PREDICATE} - set(internal_encoder.unary_pred_position_dict)
        if missing_unary:
            problems.append(f"missing unary predicates {sorted(missing_unary)}")
        if PART_OF_PREDICATE not in internal_encoder.binary_pred_colour_dict:
            problems.append(f"missing binary predicate '{PART_OF_PREDICATE}'")
        if self.d == 0:
            problems.append(f"no {node_predicate_for('k')} unary predicates")
        elif node_ks != set(range(self.d)):
            problems.append(f"node_k predicates are not exactly node_0..node_{self.d - 1}: got k in {sorted(node_ks)}")
        if self.max_value == 0:
            problems.append(f"no {colour_predicate_for('v')} binary predicates")
        elif values != set(range(1, self.max_value + 1)):
            problems.append(f"colour predicates are not exactly 1.0..{self.max_value}.0: got v in {sorted(values)}")
        if problems:
            raise ValueError("internal_encoder does not declare an ADNI signature: " + "; ".join(problems))

        self.node_pos = internal_encoder.unary_pred_position_dict[NODE_PREDICATE]
        self.positive_pos = internal_encoder.unary_pred_position_dict[POSITIVE_PREDICATE]
        self.part_of_colour = internal_encoder.binary_pred_colour_dict[PART_OF_PREDICATE]

    def node_k_pos(self, k: int) -> int:
        return self.internal_encoder.unary_pred_position_dict[node_predicate_for(k)]

    def value_colour(self, value: int) -> int:
        return self.internal_encoder.binary_pred_colour_dict[colour_predicate_for(value)]

    # Given a feature vector, identifies the k (brain region value) where the vector has a 1
    # Note that if there are more than one, it identifies only the first, but there should not be others.
    # Checking this is omitted for efficiency.
    def node_index_of(self, features) -> int:
        for k in range(self.d):
            if features[self.node_k_pos(k)] > 0:
                return k
        raise ValueError("feature vector has no node_k feature set")
