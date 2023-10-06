import time

import nodes
import torch
import numpy as np

type_pred = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'

class CanonicalEncoderDecoder:

    def __init__(self, load_from_document=None, unary_predicates=None, binary_predicates=None):

        self.unary_pred_position_dict = {}
        self.position_unary_pred_dict = {}
        self.binary_pred_colour_dict = {}
        self.colour_binary_pred_dict = {}

        if load_from_document is not None:
            for line in open(load_from_document, 'r').readlines():
                arity, num, pred = line.split()
                if arity == "UNARY":
                    self.unary_pred_position_dict[pred] = int(num)
                    self.position_unary_pred_dict[int(num)] = pred
                elif arity == "BINARY":
                    self.binary_pred_colour_dict[pred] = int(num)
                    self.colour_binary_pred_dict[int(num)] = pred
                else:
                    print("ERROR: line not recognised, it will be skipped: {}".format(line))
        elif unary_predicates is not None and binary_predicates is not None:
            for i, pred in enumerate(unary_predicates):
                self.unary_pred_position_dict[pred] = i
                self.position_unary_pred_dict[i] = pred
            for i, pred in enumerate(binary_predicates):
                self.binary_pred_colour_dict[pred] = i
                self.colour_binary_pred_dict[i] = pred
        else:
            print("ERROR: no predicates found. Please provide lists of predicates or load encoder/decoder from file.")

        self.colours = self.colour_binary_pred_dict.keys()
        self.feature_dimension = len(self.position_unary_pred_dict)

    def save_to_file(self, target_file):
        output = open(target_file, 'w')
        for i in self.position_unary_pred_dict:
            output.write("{}\t{}\t{}\n".format("UNARY", i, self.position_unary_pred_dict[i]))
        for i in self.colour_binary_pred_dict:
            output.write("{}\t{}\t{}\n".format("BINARY", i, self.colour_binary_pred_dict[i]))
        output.close()

    def encode_dataset(self, dataset, use_dummy_constants=False):

        # Maps each node v_a in the graph to set of positions 1 <= i <= dim such that U_i(a) is in the dataset
        # (value can be empty for some nodes!)
        node_positions_dict = {}

        # Maps each colour 'c' to the set of pairs of nodes v_a,v_b, such that Ec(a,b) is in the dataset
        # (value never empty for any colour)
        colour_nodepairs_dict = {}

        for RDF_triple in dataset:
            if RDF_triple[1] == type_pred:
                pred = RDF_triple[2]
                position = self.unary_pred_position_dict[pred]
                constant = RDF_triple[0]
                if constant not in nodes.const_node_dict:
                    nodes.add_node_for_constant(constant)
                node = nodes.const_node_dict[constant]
                if node not in node_positions_dict:
                    node_positions_dict[node] = {position}
                else:
                    node_positions_dict[node].add(position)
            else:
                pred = RDF_triple[1]
                colour = self.binary_pred_colour_dict[pred]
                origin_constant = RDF_triple[0]
                if origin_constant not in nodes.const_node_dict:
                    nodes.add_node_for_constant(origin_constant)
                origin_node = nodes.const_node_dict[origin_constant]
                if origin_node not in node_positions_dict:
                    node_positions_dict[origin_node] = set()
                destination_constant = RDF_triple[2]
                if destination_constant not in nodes.const_node_dict:
                    nodes.add_node_for_constant(destination_constant)
                destination_node = nodes.const_node_dict[destination_constant]
                if destination_node not in node_positions_dict:
                    node_positions_dict[destination_node] = set()
                if colour not in colour_nodepairs_dict:
                    colour_nodepairs_dict[colour] = {(origin_node, destination_node)}
                else:
                    colour_nodepairs_dict[colour].add((origin_node, destination_node))

        # Hugh's trick to penalise the model just learning to increase the biases for key facts.
        if use_dummy_constants:
            for c2 in self.colours:
                special_constant_2 = '#{}'.format(c2)
                node_for_sc2 = nodes.get_node_for_constant(special_constant_2)
                node_positions_dict[node_for_sc2] = set()
                for node in node_positions_dict:
                    if nodes.node_const_dict[node].startswith('#'):
                        if c2 not in colour_nodepairs_dict[c2]:
                            colour_nodepairs_dict[c2] = {(node_for_sc2, node)}
                        else:
                            colour_nodepairs_dict[c2].add((node_for_sc2, node))
                for c1 in self.colours:
                    special_constant_1 = '#{}#{}'.format(c2, c1)
                    node_for_sc1 = nodes.get_node_for_constant(special_constant_1)
                    node_positions_dict[node_for_sc1] = set()
                    if c1 not in colour_nodepairs_dict[c1]:
                        colour_nodepairs_dict[c1] = {(node_for_sc1, node_for_sc2)}
                    else:
                        colour_nodepairs_dict[c1].add((node_for_sc1, node_for_sc2))


        # For optimisation reasons, we store separately edges and colours, in edge_list and edge_colour_list, resp.
        edge_list = []
        edge_colour_list = []
        for colour in self.colours:
            if colour in colour_nodepairs_dict:
                edge_list += list(colour_nodepairs_dict[colour])
                edge_colour_list += [colour for _ in colour_nodepairs_dict[colour]]

        x = np.zeros((len(node_positions_dict), self.feature_dimension))
        # NOTE: a node is NOT necessarily the same as the corresponding row in matrix x.
        # This is important because the returned edges refer to rows of x and not to nodes.
        return_nodes = {}
        for i, node in enumerate(node_positions_dict.keys()):
            return_nodes[node] = i
            for position in node_positions_dict[node]:
                x[i][position] = 1
        x = torch.FloatTensor(x)
        return_edge_list = []
        for pair in edge_list:
            i = return_nodes[pair[0]]
            j = return_nodes[pair[1]]
            return_edge_list += [[i, j]]
        return_edge_list = torch.LongTensor(return_edge_list).t().contiguous()
        return_edge_colour_list = torch.LongTensor(edge_colour_list)
        if len(return_edge_list) == 0:
            return_edge_list = torch.LongTensor([[], []])

        return x, return_nodes, return_edge_list, return_edge_colour_list

    def decode_graph(self, node_row_dict, feature_vectors, threshold):

        threshold_indices = torch.nonzero(feature_vectors > threshold)

        row_node_dict = {}
        for node in node_row_dict:
            row_node_dict[node_row_dict[node]] = node

        facts_scores_dict = {}

        for index in threshold_indices:
            index = index.tolist()
            node = row_node_dict[index[0]]
            position = index[1]
            const = nodes.node_const_dict[node]
            predicate = self.position_unary_pred_dict[position]
            facts_scores_dict[(const, type_pred, predicate)] = feature_vectors[index[0]][index[1]].item()

        return facts_scores_dict


# Encoder described in Section 3.1 of the KR23 paper. To represent functional terms more succintly, the decoder
# supports only decoding of (col,d)-datasets whose terms have been produced by this encoder.
class ICLREncoderDecoder:

    def __init__(self, load_from_document=None, unary_predicates=None, binary_predicates=None):

        self.binary_canonical = {1: "binary-pred-1", 2: "binary-pred-2", 3: "binary-pred-3", 4: "binary-pred-4"}
        self.input_predicate_to_unary_canonical_dict = {}
        self.unary_canonical_to_input_predicate_dict = {}
        self.data_predicate_to_arity = {}

        if load_from_document is not None:
            for line in open(load_from_document, 'r').readlines():
                data_pred, canonical_pred, arity = line.split()
                self.input_predicate_to_unary_canonical_dict[data_pred] = canonical_pred
                self.unary_canonical_to_input_predicate_dict[canonical_pred] = data_pred
                self.data_predicate_to_arity[data_pred] = int(arity)
        elif unary_predicates is not None and binary_predicates is not None:
            self.unary_canonical_counter = 0
            for pred in unary_predicates + binary_predicates:
                if pred not in self.input_predicate_to_unary_canonical_dict:
                    self.unary_canonical_counter += 1
                    new_predicate = "unary-pred-{}".format(self.unary_canonical_counter)
                    self.input_predicate_to_unary_canonical_dict[pred] = new_predicate
                    self.unary_canonical_to_input_predicate_dict[new_predicate] = pred
                    if pred in unary_predicates:
                        self.data_predicate_to_arity[pred] = 1
                    else:
                        self.data_predicate_to_arity[pred] = 2
        else:
            print("ERROR: No predicates found. Please give lists of predicates or load encoder/decoder from a file.")

        self.tuple_term_dict = {}
        self.term_tuple_dict = {}
        self.term_counter = 0

    def save_to_file(self, target_file):
        output = open(target_file, 'w')
        for data_pred in self.input_predicate_to_unary_canonical_dict:
            output.write("{}\t{}\t{}\n".format(data_pred,
                                               self.input_predicate_to_unary_canonical_dict[data_pred],
                                               self.data_predicate_to_arity[data_pred]))
        output.close()

    def term_for_tuple(self, tup):
        if tup not in self.tuple_term_dict:
            self.term_counter += 1
            new_term = "term-{}".format(self.term_counter)
            self.tuple_term_dict[tup] = new_term
            self.term_tuple_dict[new_term] = tup
        return self.tuple_term_dict[tup]

    def exists_term_for(self, tup):
        return tup in self.tuple_term_dict

    def canonical_unary_predicates(self):
        return self.unary_canonical_to_input_predicate_dict.keys()

    def canonical_binary_predicates(self):
        return self.binary_canonical.values()

    def encode_dataset(self, dataset):

        encoded_dataset = []

        for (s, p, o) in dataset:
            if p == type_pred:
                pred = o
                constant = s
                encoded_dataset.append((self.term_for_tuple((constant,)),
                                        type_pred,
                                        self.input_predicate_to_unary_canonical_dict[pred]))
            else:
                pred = p
                origin_constant = s
                destination_constant = o
                a = self.term_for_tuple((origin_constant,))
                b = self.term_for_tuple((destination_constant,))
                ab = self.term_for_tuple((origin_constant, destination_constant))
                ba = self.term_for_tuple((destination_constant, origin_constant))
                encoded_dataset.append((ab, type_pred, self.input_predicate_to_unary_canonical_dict[pred]))
                encoded_dataset.append((a, self.binary_canonical[1], ab))
                encoded_dataset.append((ab, self.binary_canonical[1], a))
                encoded_dataset.append((b, self.binary_canonical[1], ba))
                encoded_dataset.append((ba, self.binary_canonical[1], b))
                encoded_dataset.append((b, self.binary_canonical[2], ab))
                encoded_dataset.append((ab, self.binary_canonical[2], b))
                encoded_dataset.append((a, self.binary_canonical[2], ba))
                encoded_dataset.append((ba, self.binary_canonical[2], a))
                encoded_dataset.append((ab, self.binary_canonical[3], ba))
                encoded_dataset.append((ba, self.binary_canonical[3], ab))
                encoded_dataset.append((a, self.binary_canonical[4], b))
                encoded_dataset.append((b, self.binary_canonical[4], a))
        return encoded_dataset

    def get_canonical_equivalent(self, fact):
        (s, p, o) = fact
        if p == type_pred:
            return self.term_for_tuple((s,)), type_pred, self.input_predicate_to_unary_canonical_dict[o]
        else:
            ab = self.term_for_tuple((s, o))
            return ab, type_pred, self.input_predicate_to_unary_canonical_dict[p]

    def decode_fact(self, s, p, o):
        assert(p == type_pred)
        tup = self.term_tuple_dict[s]
        if len(tup) == 1:
            a = tup[0]
            return a, type_pred, self.unary_canonical_to_input_predicate_dict[o]
        else:
            a = tup[0]
            b = tup[1]
            return a, self.unary_canonical_to_input_predicate_dict[o], b

    def get_data_predicate(self, canonical_predicate):
        return self.unary_canonical_to_input_predicate_dict[canonical_predicate]

    # COME HERE!! Problem: if introducing new variables for pair node, one of them should match the one of the single
    # node. Fix that.
    def unfold(self, rule_body):
        new_body = []
        nontop_bin_atoms = set()
        top_bin_atoms = set()
        variable_unification = {}
        new_variables_counter = 0
        facts_for_c1_c2 = []
        second_round = []
        pair_var_to_first_var = {}
        pair_var_to_second_var = {}
        for (s, p, o) in rule_body:
            if p == type_pred:
                if self.data_predicate_to_arity[self.unary_canonical_to_input_predicate_dict[o]] == 1:
                    if s in variable_unification:
                        y1 = variable_unification[s]
                    else:
                        new_variables_counter += 1
                        y1 = "Y{}".format(new_variables_counter)
                        variable_unification[s] = y1
                    new_body.append((y1, type_pred, self.unary_canonical_to_input_predicate_dict[o]))
                else:
                    second_round.append((s, p, o))
            else:
                # We need not generate anything from facts of colours 1-3,
                # since the necessary fact is covered already by decoding c4.
                # But we add them to a separate list since they will help us later.
                if p == self. binary_canonical[1]:
                    facts_for_c1_c2.append((s, p, o))
                elif p == self.binary_canonical[2]:
                    facts_for_c1_c2.append((s, p, o))
                elif p == self.binary_canonical[4]:
                    if s in variable_unification:
                        y1 = variable_unification[s]
                    else:
                        new_variables_counter += 1
                        y1 = "Y{}".format(new_variables_counter)
                        variable_unification[s] = y1
                    if o in variable_unification:
                        y2 = variable_unification[o]
                    else:
                        new_variables_counter += 1
                        y2 = "Y{}".format(new_variables_counter)
                        variable_unification[o] = y2
                    top_bin_atoms.add((y1, y2))
        # The second round takes care of canonical facts involving terms for pairs of constants
        for (s, p, o) in facts_for_c1_c2:
            if p == self.binary_canonical[1]:
               # Exactly one of the two variables (the one for a single term) should be in variable_unification
                if s in variable_unification:
                    pair_var_to_first_var[o] = variable_unification[s]
                else:
                    pair_var_to_first_var[s] = variable_unification[o]
            if p == self.binary_canonical[2]:
                if s in variable_unification:
                    pair_var_to_second_var[o] = variable_unification[s]
                else:
                    pair_var_to_second_var[s] = variable_unification[o]
        for (s, p, o) in second_round:
            assert s in pair_var_to_first_var or s in pair_var_to_second_var
            if s not in pair_var_to_first_var:
                new_variables_counter += 1
                y1 = "Y{}".format(new_variables_counter)
                pair_var_to_first_var[s] = y1
            if s not in pair_var_to_second_var:
                new_variables_counter += 1
                y2 = "Y{}".format(new_variables_counter)
                pair_var_to_second_var[s] = y2
            y1 = pair_var_to_first_var[s]
            y2 = pair_var_to_second_var[s]
            new_body.append((y1, self.unary_canonical_to_input_predicate_dict[o], y2))
            nontop_bin_atoms.add((y1, y2))
        # Small optimisation to avoid adding TOP(x,y) if we already have R(x,y) for some R
        for (s, o) in top_bin_atoms:
            if (s, o) not in nontop_bin_atoms and (o, s) not in nontop_bin_atoms:
                new_body.append((s, "TOP", o))
        return new_body, variable_unification
