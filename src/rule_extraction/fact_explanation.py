import numpy as np
import torch

from src.model.cd_graph import TraceCollector, CDGraph
from src.model.gnn_transformation import apply_gnn_transformation, apply_model, apply_c_decoder, apply_nc_decoder
from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.noncanonical import NonCanonicalEncoder, GroundContext
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunctionBuilder
from src.utils.utils import TYPE_PRED, backpropagate_relevance
from src.utils.bitset import BitSet
from src.rule_extraction.rule_optimisation_3 import RuleOptimisation3
from src.datalog.apply_rules import apply_rule

# We bundle a bunch of auxiliary info about a fact we want to explain
class FactContext:
    def __init__(self, fact: tuple[str, str, str], external_encoder: NonCanonicalEncoder,
                 internal_encoder: CanonicalEncoderDecoder, node_to_index_dict):
        self.fact = fact
        self.ent1, self.ent2, self.ent3 = fact
        self.cd_ent1, self.cd_ent2, self.cd_ent3 = external_encoder.get_canonical_equivalent(fact)
        self.cd_fact_const_index = node_to_index_dict[self.cd_ent1]
        self.cd_fact_pred_pos = internal_encoder.unary_pred_position_dict[self.cd_ent3]

# This class manages the explanation of a fact derived by the GNN
class FactExplainer:

    def __init__(self, device, model, threshold, trace: TraceCollector, external_encoder: NonCanonicalEncoder,
                 internal_encoder: CanonicalEncoderDecoder, input_dataset=None):
        self.device = device
        self.model = model
        self.threshold = threshold
        self.external_encoder = external_encoder
        self.internal_encoder = internal_encoder
        self.activations = trace.activations
        self.cd_graph = trace.cd_graph
        self.node_to_index = {node: i for i, node in enumerate(self.cd_graph.node_names)}  # Helpful dictionary
        self.input_dataset = input_dataset

    # This is the Gamma_i in the papers. It computes a most general explanation but prunes exploiting matrix sparsity
    # Takes a Fact as input, but wrapped with some auxiliary values as a FactContext
    def get_basic_explanation(self, fact_context: FactContext):
        explanation_builder = TreeShapedConjunctionBuilder(self.internal_encoder.get_n_binary_predicates())
        L = self.model.num_layers
        explanation_builder.add(features=None,level=L,parent=-1) # Root node
        # Companion to basic_explanation. Maps a variable id to the id of the constant that grounds it. \nu in the paper
        varid_2_constid = {0: fact_context.cd_fact_const_index}
        # Companion to basic_explanation. Maps a (var id, layer) to the relevant Feature Mask. This is the paper's \mu.
        initial_mask = BitSet.from_subset(self.model.layer_dimension(L),{fact_context.cd_fact_pred_pos})
        var_layer_mask = {(0, L): initial_mask}
        # Paper's algorithm for constructing the conjunction
        for l in range(L, 0, -1):  # Iterate backwards over all layers from L to 1 (both inclusive).
            num_vars = explanation_builder.num_vars()
            for var_id in range(num_vars):
                var_layer_mask[(var_id, l - 1)] = backpropagate_relevance(var_layer_mask[(var_id, l)],
                                                                       self.model.matrix_A(l),
                                                                       self.activations[l - 1][varid_2_constid[var_id]])
                # Introduce children new variables for var and define their relevant positions
                for colour in self.internal_encoder.get_colours():
                    edge_mask = self.cd_graph.edge_colours == colour
                    colour_edges = self.cd_graph.edges[:, edge_mask]
                    neighbours = colour_edges[:, colour_edges[1] == varid_2_constid[var_id]][0].tolist()
                    if not neighbours:
                        continue
                    neighbour_vectors = np.array([self.activations[l - 1][neighbour] for neighbour in neighbours])
                    for j in backpropagate_relevance(var_layer_mask[(var_id,l)],
                                                     self.model.matrix_B(l, colour),
                                                     previous_activations=None).elements():
                        # Find the neighbour that contributes maximum to aggregation
                        best_idx = np.argmax(neighbour_vectors[:, j])
                        if neighbour_vectors[best_idx, j] > 0:
                            new_var_id = (
                                explanation_builder.add(features=None,level=l-1,parent=var_id,edge=(l, colour, j)))
                            varid_2_constid[new_var_id] = neighbours[best_idx]
                            var_layer_mask[(new_var_id, l - 1)] = (
                                BitSet.from_subset(self.model.layer_dimension(l - 1), {j}))

        for var_id in range(explanation_builder.num_vars()):  # Add the atoms for the feature vectors in layer 0
            explanation_builder.features[var_id] = var_layer_mask[(var_id, 0)]
            # Needs to be done separately, otherwise this is not done to the new variables added!

        return explanation_builder.build(), varid_2_constid, var_layer_mask

    def explain_fact(self, fact: tuple[str,str,str]):

        fact_context = FactContext(fact, self.external_encoder, self.internal_encoder, self.node_to_index)
        assert self.activations[2][fact_context.cd_fact_const_index][fact_context.cd_fact_pred_pos] > self.threshold, \
            "Error: the fact to be explained is not derived by the model on this dataset."

        print("Computing Gamma_i")
        rule_body, var_const_idx, var_layer_mask = self.get_basic_explanation(fact_context)
        print("Length Gamma_i: {}".format(len(rule_body)))

        # TODO: Refactor all 3 optimisations
        # print("Attempting approximation 1...")
        # optimised_body_1 = run_optimisation_1(self)
        # print("Length rule 1: {}".format(len(optimised_body_1)))

        # print("Attempting approximation 2...")
        # optimised_body_2 = run_optimisation_2(self)
        # print("Length rule 2: {}".format(len(optimised_body_2)))

        #if len(optimised_body_2) < len(optimised_body_1):
        #    print("Approximation 2 wins")
        #    rule_body = optimised_body_2
        # else:
        #    print("Approximation 1 wins")

        optimiser3 =  RuleOptimisation3(device=self.device,
                                        model=self.model,
                                        threshold=self.threshold,
                                        pred_position=fact_context.cd_fact_pred_pos,
                                        base_tree=rule_body)
        minimised = optimiser3.minimise_rule()
        if minimised is not None:  # If no simplification was found in time, keep the original rule_body
            rule_body = rule_body.extract_from_compact(minimised)
        # TODO: this should be a call to the external encoder
        if fact_context.ent2 == TYPE_PRED:
            head_predicate = fact_context.ent3
        else:
            head_predicate = fact_context.ent2

        # Unfold into body via external encoder/decoder
        # This converts a TreeShapedConjunction into a simple list of triples, plus a list of head variables
        rule_body, head = self.external_encoder.unfold_match_ground(
            can_conj=rule_body,
            internal_encoder=self.internal_encoder,
            head_predicate=head_predicate,
            grounding_context=GroundContext(fact, self.cd_graph, var_const_idx))

        # Write the rule
        body_atoms = []
        rule_body = set(rule_body)  # Remove duplicates
        for (s, p, o) in rule_body:
            if p == TYPE_PRED:
                body_atoms.append("<{}>[?{}]".format(o, s))
            else:
                body_atoms.append("<{}>[?{},?{}]".format(p, s, o))
        if head[1] is not TYPE_PRED:
            written_head =  "<{}>[?{},?{}]".format(head[1], head[0], head[2])
        else:
            written_head = "<{}>[?{}]".format(head[2], head[0])
        rule = written_head + " :- " + ", ".join(body_atoms) + " .\n"

        # Verify that the rule is sound:
        if not rule_body: # Soundness check algorithm for rules with empty body TODO: generalise this like in the paper
            cd_graph = CDGraph(self.internal_encoder.get_n_binary_predicates(),
                               self.internal_encoder.get_n_unary_predicates(),
                               features=torch.zeros((1,self.internal_encoder.get_n_unary_predicates()),dtype=torch.float),
                               edges=torch.empty((2,0), dtype=torch.long),
                               edge_colours=torch.tensor([], dtype=torch.long),
                               node_names=["X0"])
            output_cd_graph = apply_model(cd_graph, self.device, self.model)
            cd_dataset_facts_scores_dict = apply_c_decoder(output_cd_graph, self.threshold, self.internal_encoder)
            predictions_dict = apply_nc_decoder(cd_dataset_facts_scores_dict, self.external_encoder)
        else:
            predictions_dict = apply_gnn_transformation(dataset=rule_body,
                                                external_encoder=self.external_encoder,
                                                internal_encoder=self.internal_encoder,
                                                model=self.model,
                                                threshold=self.threshold,
                                                device=self.device)
        assert predictions_dict[head] > self.threshold

        # Verify that the rule is sufficient
        # TODO: The None option should not be allowed, but so far we leave it to not break the tests.
        if self.input_dataset is not None:
            assert fact in apply_rule(rule,self.input_dataset)

        # Return rule
        return rule

