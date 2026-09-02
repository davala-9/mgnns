from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.noncanonical import NonCanonicalEncoder
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, TreeShapedConjunctionBuilder
from src.utils.bitset import BitSet
from src.utils.utils import backpropagate_relevance, TYPE_PRED
import time

class EquivalentProgramExtractor:

    def __init__(self, device, model, threshold, external_encoder: NonCanonicalEncoder,
                 internal_encoder: CanonicalEncoderDecoder):
        self.device = device
        self.model = model
        self.threshold = threshold
        self.external_encoder = external_encoder
        self.internal_encoder = internal_encoder
        self.base_tree: dict[int, TreeShapedConjunction] = {}

    # Build initial upper bound/search space for a given position/predicate
    def compute_tree_for(self, predicate_position):
        explanation_builder = TreeShapedConjunctionBuilder(self.internal_encoder.get_n_binary_predicates())
        L = self.model.num_layers
        explanation_builder.add(features=None,level=L,parent=None)
        # Companion to basic_explanation. Maps a (var id, layer) to the relevant Feature Mask. This is the paper's \mu.
        initial_mask = BitSet.from_subset(self.internal_encoder.get_n_unary_predicates(),{predicate_position})
        var_layer_mask = {(0, L): initial_mask}
        # Paper's algorithm for constructing the conjunction
        for l in range(L, 0, -1):  # Iterate backwards over all layers from L to 1 (both inclusive).
            num_vars = explanation_builder.num_vars()
            for var_id in range(num_vars):
                var_layer_mask[(var_id, l - 1)] =\
                    backpropagate_relevance(var_layer_mask[(var_id, l)], self.model.matrix_A(l))
                # Introduce children new variables for var and define their relevant positions
                for colour in self.internal_encoder.get_colours():
                    for j in backpropagate_relevance(var_layer_mask[(var_id, l)],
                                                     self.model.matrix_B(l, colour)).elements():
                        new_var_id = (
                            explanation_builder.add(features=None, level=l - 1, parent=var_id, edge=(l, colour, j)))
                        var_layer_mask[(new_var_id, l - 1)] = \
                            BitSet.from_subset(self.model.layer_dimension(l - 1), {j})

        for var_id in range(explanation_builder.num_vars()):  # Add the atoms for the feature vectors in layer 0
            explanation_builder.features[var_id] = var_layer_mask[(var_id, 0)]
            # Needs to be done separately, otherwise this is not done to the new variables added!
        return explanation_builder.build()

    # Build all search spaces
    def compute_all_upper_bounds(self):
        for i in range(self.internal_encoder.get_n_unary_predicates()):
            self.base_tree[i] = self.compute_tree_for(i)

    # We construct and explore a lattice out of the subtrees of the base_tree
    # We explore layer by layer, to minimise memory usage
    # At each point, we have a 'current_layer' and a 'next_layer'
    # Works as a generator function
    def extract_smallest_rules_for(self, predicate_position, deadline):
        pred = self.internal_encoder.unary_pred_position_dict.inverse[predicate_position]
        print("Computing rules for predicate " + pred)
        # Initialisation
        bottom_node = self.base_tree[predicate_position].initial_compact
        current_layer = [bottom_node]
        subsumed = {} # Dictionary of nodes that are subsumed already by some smaller sound node
        if bottom_node.check_soundness(self.device, self.model, self.threshold, predicate_position):
            yield bottom_node
            subsumed[bottom_node] = True
        else:
            subsumed[bottom_node] = False
        layer_counter = 0
        rule_counter = 0

        # Main loop. Invariant: 'subsumed' maps each node of the current layer to either true or false
        while current_layer:
            print("Current layers explored: " + str(layer_counter))
            print("Rules extracted so far: " + str(rule_counter))
            # Expand the next layer
            next_layer = set() # Seen nodes in the next layer
            for x in current_layer:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                if not subsumed[x]:
                    next_layer.update(x.get_successors())
            # Process nodes in the next layer.
            new_subsumed = {} # Make a new dictionary to save memory - we don't need 'subsumed' in the next iter
            for x in next_layer:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                preds = x.get_predecessors()
                if any(subsumed.get(p, False) for p in preds):
                    new_subsumed[x] = True
                else:
                    if x.check_soundness(self.device, self.model, self.threshold, predicate_position):
                        new_subsumed[x] = True
                        yield x
                        rule_counter += 1
                    else:
                        new_subsumed[x] = False
            subsumed = new_subsumed
            current_layer = next_layer
            layer_counter += 1

    def hail_mary(self, predicate_position):
        node = self.base_tree[predicate_position].initial_compact
        while True:
            if node.check_soundness(
                    self.device,
                    self.model,
                    self.threshold,
                    predicate_position
            ):
                yield node
                return
            node = next(node.get_successors(), None)
            if node is None:
                return

    def unfold_and_print_rule(self, pred_pos, compressed_rule_body, rules_for_this_predicate, output):
        rule_body = self.base_tree[pred_pos].extract_from_compact(compressed_rule_body)
        head_can_predicate = self.internal_encoder.get_unary_predicate_for_position(pred_pos)
        head_pred = self.external_encoder.unary_can_predicate_to_data_predicate(head_can_predicate)
        head_pred_arity = self.external_encoder.unary_can_predicate_to_data_predicate_arity(
            head_can_predicate)
        rule_bodies, head = self.external_encoder.unfold_all(
            can_conj=rule_body, internal_encoder=self.internal_encoder, head_predicate=head_pred)
        for rule_body in rule_bodies:
            if frozenset(rule_body) not in rules_for_this_predicate:
                rules_for_this_predicate.add(frozenset(rule_body))
                # Write the rule
                body_atoms = []
                rule_body = set(rule_body)  # Remove duplicates
                for (s, p, o) in rule_body:
                    if p == TYPE_PRED:
                        body_atoms.append("<{}>[?{}]".format(o, s))
                    else:
                        body_atoms.append("<{}>[?{},?{}]".format(p, s, o))
                if head_pred_arity == 2:
                    written_head = "<{}>[?{},?{}]".format(head[1], head[0], head[2])
                else:
                    written_head = "<{}>[?{}]".format(head[2], head[0])
                rule = written_head + " :- " + ", ".join(body_atoms) + " .\n"
                output.write(rule + '\n')


    def get_all_rules(self, program_file, time_budget):
        with (open(program_file, 'w') as output):
            for pred_pos in range(self.internal_encoder.get_n_unary_predicates()):
                # TODO undo this if. This is a quick and dirty thing for a specific application.
                if self.internal_encoder.unary_pred_position_dict.inverse[pred_pos] == "positive":
                    deadline = time.monotonic() + time_budget if time_budget is not None else None
                    rules_for_this_predicate = set()
                    rule_counter = 0
                    for compressed_rule_body in self.extract_smallest_rules_for(pred_pos, deadline):
                        rule_counter += 1
                        self.unfold_and_print_rule(pred_pos, compressed_rule_body, rules_for_this_predicate, output)
                    if rule_counter == 0:
                        print("Time's up, going for a Hail Mary...")
                        compressed_rule_body = next(self.hail_mary(pred_pos), None)
                        if compressed_rule_body is not None:
                            self.unfold_and_print_rule(pred_pos, compressed_rule_body, rules_for_this_predicate, output)

        output.close()

