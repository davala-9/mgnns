from typing import Callable

from src.datalog.apply_rules import format_rule
from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.iclr22 import ICLREncoderDecoder
from src.encodings.noncanonical.noncanonical import NonCanonicalEncoder
from src.rule_extraction.lattice_search import (
    Frontier, BFSFrontier, SinglePathFrontier, GreedyBestSuccessorFrontier, AllMinimalPolicy, FirstResultPolicy,
    search,
)
from src.rule_extraction.rule_optimisation_2 import compute_path_weights, path_weight_score_fn
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, TreeShapedConjunctionBuilder
from src.rule_extraction.typed_constraint import TypedConstraint
from src.utils.bitset import BitSet
from src.utils.utils import backpropagate_relevance
import time

# Filters var_id's own-feature mask through every given typed_constraint's always_zero constraint for
# var_id's type under that constraint. var_types maps var_id to a list of types, one per typed_constraint
# (same order); a var_id missing from var_types (e.g. because no typed_constraints were given) is left untouched.
def filter_own_features_by_typed_constraints(mask, var_id, typed_constraints, var_types):
    for tc, node_type in zip(typed_constraints, var_types.get(var_id, [])):
        mask = tc.filter_out_always_zero_features(node_type, mask)
    return mask

class EquivalentProgramExtractor:

    def __init__(self, device, model, threshold, external_encoder: NonCanonicalEncoder,
                 internal_encoder: CanonicalEncoderDecoder):
        self.device = device
        self.model = model
        self.threshold = threshold
        self.external_encoder = external_encoder
        self.internal_encoder = internal_encoder
        self.base_tree: dict[int, TreeShapedConjunction] = {}
        self.var_layer_mask: dict[int, dict] = {}  # needed to compute atom_weights() for the Hail Mary heuristic
        self._atom_weight_cache: dict[int, dict] = {}

    # Build initial upper bound/search space for a given position/predicate.
    def compute_tree_for(self, predicate_position, *typed_constraints_with_root_type: tuple[TypedConstraint, str]):
        for tc, root_type in typed_constraints_with_root_type:
            if root_type not in tc.types:
                raise ValueError(f"{root_type!r} is not one of this constraint's declared types {tc.types}")
        typed_constraints = [tc for tc, _ in typed_constraints_with_root_type]

        explanation_builder = TreeShapedConjunctionBuilder(self.internal_encoder.get_n_binary_predicates())
        L = self.model.num_layers
        explanation_builder.add(features=None,level=L,parent=-1)
        # Maps a (var id, layer) to the relevant Feature Mask: this is the paper's \mu.
        initial_mask = BitSet.from_subset(self.internal_encoder.get_n_unary_predicates(),{predicate_position})
        var_layer_mask = {(0, L): initial_mask}
        # Maps each var_id to a list of types, one per constraint in typed_constraints (same order).
        var_types: dict[int, list] = {0: [root_type for _, root_type in typed_constraints_with_root_type]}
        # Paper's algorithm for constructing the conjunction
        for l in range(L, 0, -1):  # Iterate backwards over all layers from L to 1 (both inclusive).
            num_vars = explanation_builder.num_vars()
            for var_id in range(num_vars):
                # First update self mask for the next layer.
                var_layer_mask[(var_id, l - 1)] =\
                    backpropagate_relevance(var_layer_mask[(var_id, l)], self.model.matrix_A(l))
                # Then introduce all relevant children and define their relevant positions
                for colour in self.internal_encoder.get_colours():
                    # Optimisation: use TypedConstraints to prune if no neighbour by this colour is possible.
                    if any(not tc.may_have_neighbour_of_colour(node_type, colour)
                           for tc, node_type in zip(typed_constraints, var_types.get(var_id, []))):
                        continue
                    # Compute types under TypedConstraints for any child by colour. Currently unique and deterministic.
                    child_types = [tc.get_child_type(node_type, colour)
                                   for tc, node_type in zip(typed_constraints, var_types.get(var_id, []))]
                    candidates = backpropagate_relevance(var_layer_mask[(var_id, l)], self.model.matrix_B(l, colour))
                    # If any typed_constraint says a node of var_id's type has at most one neighbour of
                    # child_type via this colour, that single (possible) child covers every candidate position
                    # at once, rather than spawning one variable per candidate.
                    at_most_one = any(
                        colour in tc.get_edge_constraints(node_type, child_type).atmost_one
                        for tc, node_type, child_type
                        in zip(typed_constraints, var_types.get(var_id, []), child_types))
                    if at_most_one:
                        if not candidates.is_empty():
                            new_var_id = explanation_builder.add(
                                features=None, level=l - 1, parent=var_id,
                                edge=(l, colour, frozenset(candidates.elements())))
                            var_layer_mask[(new_var_id, l - 1)] = candidates
                            var_types[new_var_id] = child_types
                    else:
                        for j in candidates.elements():
                            new_var_id = (
                                explanation_builder.add(
                                    features=None, level=l - 1, parent=var_id, edge=(l, colour, frozenset({j}))))
                            var_layer_mask[(new_var_id, l - 1)] = \
                                BitSet.from_subset(self.model.layer_dimension(l - 1), {j})
                            var_types[new_var_id] = child_types

        # TODO: consider the non-deterministic version of having multiple trees, to turn combinatorial explosion
        #  into linear

        for var_id in range(explanation_builder.num_vars()):  # Add the atoms for the feature vectors in layer 0
            # Needs to be done separately, otherwise this is not done to the new variables added!
            mask = filter_own_features_by_typed_constraints(
                var_layer_mask[(var_id, 0)], var_id, typed_constraints, var_types)
            var_layer_mask[(var_id, 0)] = mask
            explanation_builder.features[var_id] = mask
        return explanation_builder.build(tuple(typed_constraints), var_types), var_layer_mask

    # Maps a data-signature predicate name back to its canonical unary predicate position.
    def resolve_predicate_position(self, data_predicate: str) -> int:
        for i in range(self.internal_encoder.get_n_unary_predicates()):
            can_predicate = self.internal_encoder.get_unary_predicate_for_position(i)
            if self.external_encoder.unary_can_predicate_to_data_predicate(can_predicate) == data_predicate:
                return i
        raise ValueError(f"Unknown predicate: {data_predicate!r}")

    # Build all search spaces. If predicate_positions is None, builds all of them.
    def compute_all_upper_bounds(self, predicate_positions=None):
        if predicate_positions is None:
            predicate_positions = range(self.internal_encoder.get_n_unary_predicates())
        is_iclr = isinstance(self.external_encoder, ICLREncoderDecoder)
        # The TypedConstraint depends only on the encoding, not on the predicate, so one instance is built and
        # shared by every predicate's tree (nothing downstream mutates it).
        typed_constraint = None
        if is_iclr:
            typed_constraint = self.external_encoder.typed_constraint()
        for i in predicate_positions:
            if is_iclr:
                typed_constraints_with_root_type = ((typed_constraint, self.external_encoder.root_role(i)),)
            else:
                typed_constraints_with_root_type = ()
            self.base_tree[i], self.var_layer_mask[i] = (
                self.compute_tree_for(i, *typed_constraints_with_root_type))

    # Optimisation 2's analytic per-atom weights (see rule_optimisation_2.compute_path_weights), computed
    # lazily and cached: they're only needed to steer the Hail Mary fallback (see hail_mary_frontier below)
    # towards promising atoms first, so there's no point computing them for predicates that never time out.
    def atom_weights(self, predicate_position):
        if predicate_position not in self._atom_weight_cache:
            self._atom_weight_cache[predicate_position] = compute_path_weights(
                self.model, self.base_tree[predicate_position],
                self.var_layer_mask[predicate_position], predicate_position)
        return self._atom_weight_cache[predicate_position]

    # A frontier_factory for hail_mary(): a greedy climb (see GreedyBestSuccessorFrontier) guided by the
    # weights above, instead of following an arbitrary path.
    def hail_mary_frontier(self, predicate_position):
        weights = self.atom_weights(predicate_position)
        return GreedyBestSuccessorFrontier(path_weight_score_fn(self.base_tree[predicate_position], weights))

    # Explores the lattice of subtrees of the base_tree, yielding every inclusion-minimal sound one (a
    # generator). The exploration order is controlled by `frontier_factory` (default: level-by-level BFS).
    def extract_smallest_rules_for(self, predicate_position, deadline,
                                   frontier_factory: Callable[[], Frontier] = BFSFrontier):
        pred = self.internal_encoder.unary_pred_position_dict.inverse[predicate_position]
        print("Computing rules for predicate " + pred)
        base_tree = self.base_tree[predicate_position]

        def check_soundness(node):
            return node.check_soundness(base_tree, self.device, self.model, self.threshold, predicate_position)

        rule_counter = 0
        for node in search(base_tree, check_soundness, frontier_factory(), AllMinimalPolicy(), deadline):
            rule_counter += 1
            print("Rules extracted so far: " + str(rule_counter))
            yield node

    def hail_mary(self, predicate_position, frontier_factory: Callable[[], Frontier] = SinglePathFrontier):
        base_tree = self.base_tree[predicate_position]

        def check_soundness(node):
            return node.check_soundness(base_tree, self.device, self.model, self.threshold, predicate_position)

        yield from search(base_tree, check_soundness, frontier_factory(), FirstResultPolicy())

    def unfold_and_print_rule(self, pred_pos, compressed_rule_body, rules_for_this_predicate, output):
        rule_body = self.base_tree[pred_pos].extract_from_compact(compressed_rule_body)
        head_can_predicate = self.internal_encoder.get_unary_predicate_for_position(pred_pos)
        head_pred = self.external_encoder.unary_can_predicate_to_data_predicate(head_can_predicate)
        rule_bodies, head = self.external_encoder.unfold_all(
            can_conj=rule_body, internal_encoder=self.internal_encoder, head_predicate=head_pred)
        for rule_body in rule_bodies:
            if frozenset(rule_body) not in rules_for_this_predicate:
                rules_for_this_predicate.add(frozenset(rule_body))
                output.write(format_rule(head, set(rule_body)) + '\n\n')


    def get_all_rules(self, program_file, time_budget, predicate_positions=None):
        if predicate_positions is None:
            predicate_positions = range(self.internal_encoder.get_n_unary_predicates())
        with open(program_file, 'w') as output:
            for pred_pos in predicate_positions:
                deadline = time.monotonic() + time_budget if time_budget is not None else None
                rules_for_this_predicate = set()
                rule_counter = 0
                for compressed_rule_body in self.extract_smallest_rules_for(pred_pos, deadline):
                    rule_counter += 1
                    self.unfold_and_print_rule(pred_pos, compressed_rule_body, rules_for_this_predicate, output)
                if rule_counter == 0:
                    print("Time's up, going for a Hail Mary...")
                    frontier_factory = lambda: self.hail_mary_frontier(pred_pos)
                    compressed_rule_body = next(self.hail_mary(pred_pos, frontier_factory), None)
                    if compressed_rule_body is not None:
                        self.unfold_and_print_rule(pred_pos, compressed_rule_body, rules_for_this_predicate, output)


