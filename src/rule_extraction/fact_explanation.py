import numpy as np

from src.model.cd_graph import TraceCollector
from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.noncanonical import NonCanonicalEncoder
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, Variable
from src.utils.utils import TYPE_PRED, backpropagate_relevance
from src.utils.bitset import BitSet


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
                 internal_encoder: CanonicalEncoderDecoder):

        self.device = device
        self.model = model
        self.threshold = threshold
        self.external_encoder = external_encoder
        self.internal_encoder = internal_encoder
        self.activations = [trace.fl0,trace.fl1,trace.fl2] # Index matches layer
        self.cd_graph = trace.cd_graph
        self.node_to_index = {node: i for i, node in enumerate(self.cd_graph.node_names)}  # Helpful dictionary


    # This is the Gamma_i in the papers. It computes a most general explanation but prunes exploiting matrix sparsity
    # Takes a Fact as input, but wrapped with some auxiliary values as a FactContext
    def get_basic_explanation(self, fact_context: FactContext):

        # We initialise the conjunction as an empty tree-shaped conjunction
        L = self.model.num_layers
        initial_mask = BitSet.from_subset(self.internal_encoder.get_n_unary_predicates(),
                                          {fact_context.cd_fact_pred_pos})
        root_variable = Variable(level=L)
        conjunction = TreeShapedConjunction(root_variable, self.internal_encoder.get_n_binary_predicates)
        # Maps a Variable to the (index of the) constant that grounds it. This is the \nu mapping in the paper
        var_const_idx = {root_variable: fact_context.cd_fact_const_index}
        # Maps a (Variable, Layer) to the relevant Feature Mask. This is the paper's \mu.
        var_layer_mask = {(root_variable, L): initial_mask}

        # Paper's algorithm for constructing the conjunction
        for l in range(L, 0, -1):  # Iterate backwards over all layers from L to 1 (both inclusive).
            for var in conjunction.walk():
                var_layer_mask[(var, l - 1)] = backpropagate_relevance(var_layer_mask[(var, l)],
                                                                       self.model.matrix_A(l),
                                                                       self.activations[l - 1][var_const_idx[var]])
                # Introduce children new variables for var and define their relevant positions
                for colour in self.internal_encoder.get_colours():
                    edge_mask = self.cd_graph.edge_colours == colour
                    colour_edges = self.cd_graph.edges[:, edge_mask]
                    neighbours = colour_edges[:, colour_edges[1] == var_const_idx[var]][0].tolist()
                    if not neighbours:
                        continue
                    neighbour_vectors = np.array([self.activations[l - 1][neighbour] for neighbour in neighbours])
                    for j in backpropagate_relevance(var_layer_mask[(var,l)],
                                                     self.model.matrix_B(l, colour),
                                                     previous_activations=None).elements():
                        # Find the neighbour that contributes maximum to aggregation
                        best_idx = np.argmax(neighbour_vectors[:, j])
                        if neighbour_vectors[best_idx, j] > 0:
                            new_variable = Variable(level=l - 1)
                            var.children[(l, colour, j)] = new_variable
                            var_const_idx[new_variable] = neighbours[best_idx]
                            var_layer_mask[(new_variable, l - 1)] = (
                                BitSet.from_subset(self.model.layer_dimension(l - 1), {j}))

        for var in conjunction.walk():  # Add the atoms for the feature vectors in layer 0
            var.features = var_layer_mask[(var, 0)]
            # Needs to be done separately, otherwise this is not done to the new variables added!

        # TODO: Redo this
        # (gr_features, node_to_gr_row_dict, gr_edge_list, gr_colour_list) = self.can_encoder_decoder.encode_dataset(gamma_i)
        # gnn_output_gr, _  = self.model(Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(self.device))
        # assert (gnn_output_gr[node_to_gr_row_dict[nodes.const_node_dict[x1]]][cd_fact_pred_pos] >=
        #         self.cfg.derivation_threshold), "ERROR: Gamma_i is not sound. This should not happen; there's a bug."
        # TODO: also check that the variable levels match the \mu and the trees.

        return conjunction

    def explain_fact(self, fact: tuple[str,str,str]):

        fact_context = FactContext(fact, self.external_encoder, self.internal_encoder, self.node_to_index)
        assert self.activations[2][fact_context.cd_fact_const_index][fact_context.cd_fact_pred_pos] >= self.threshold, \
            "Error: the fact to be explained is not derived by the model on this dataset."

        print("Computing Gamma_i")
        rule_body = self.get_basic_explanation(fact_context)
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

        # Optimisation 3 used to go here and was applied to the best of 1 or 2

        # Unfold into body via external encoder/decoder
        # This converts a TreeShapedConjunction into a simple list of triples, plus a list of head variables
        rule_body, head_variables = self.external_encoder.unfold(can_conj=rule_body,
                                                                 internal_encoder=self.internal_encoder,
                                                                 explainer=self)

        # Write the rule
        body_atoms = []
        rule_body = set(rule_body)  # Remove duplicates
        for (s, p, o) in rule_body:
            if p == TYPE_PRED:
                body_atoms.append("<{}>[?{}]".format(o, s))
            else:
                body_atoms.append("<{}>[?{},?{}]".format(p, s, o))
        if fact_context.ent2 is not TYPE_PRED:
            head =  "<{}>[?{},?{}]".format(fact_context.ent2,head_variables[0],head_variables[1])
        else:
            head = "<{}>[?{}]".format(fact_context.ent3,head_variables[0])
        return head + " :- " + ", ".join(body_atoms) + " .\n"

