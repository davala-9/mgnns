import torch
from utils import type_pred
from bidict import bidict
from cd_graph import CDGraph
from collections import defaultdict
import sys
from abc import ABC, abstractmethod

ineq_pred = "owl:differentFrom"

class CanonicalEncoderDecoder:

    # If the signature lacks unary or binary predicates, we add dummies to the signature (but not use them in any facts)
    DUMMY_PRED = "DUMMY_PRED"
    DUMMY_COL = "DUMMY_COL"

    def __init__(self, load_from_document=None, unary_predicates=None, binary_predicates=None):

        self.unary_pred_position_dict: bidict = bidict()
        self.binary_pred_colour_dict: bidict = bidict()

        if load_from_document is not None:
            # We trust the document!
            for line in open(load_from_document, 'r').readlines():
                arity, position, predicate = line.split()
                if arity == "UNARY":
                    self.unary_pred_position_dict[predicate] = int(position)
                elif arity == "BINARY":
                    self.binary_pred_colour_dict[predicate] = int(position)
                else:
                    sys.exit("ERROR: line not recognised: {}".format(line))
        else:
            for i, predicate in enumerate(unary_predicates):
                self.unary_pred_position_dict[predicate] = i
            if not self.unary_pred_position_dict:
                self.unary_pred_position_dict[self.DUMMY_PRED] = 0 # Add a single dummy unary predicate
            for i, predicate in enumerate(binary_predicates):
                self.binary_pred_colour_dict[predicate] = i
            if not self.binary_pred_colour_dict:
                self.binary_pred_colour_dict[self.DUMMY_COL] = 0 # Add a single dummy colour

    def save_to_file(self, target_file):
        output = open(target_file, 'w')
        for i in self.unary_pred_position_dict.inverse:
            output.write("{}\t{}\t{}\n".format("UNARY", i, self.unary_pred_position_dict.inverse[i]))
        for i in self.binary_pred_colour_dict.inverse:
            output.write("{}\t{}\t{}\n".format("BINARY", i, self.binary_pred_colour_dict.inverse[i]))
        output.close()

    # Given a (col,d)-dataset, returns its resulting (col,d)-graph
    def encode_dataset(self, dataset):

        nodename_feature_dict = defaultdict(lambda: torch.zeros(delta, dtype=torch.float))
        edges = set()
        delta = len(self.unary_pred_position_dict)
        col_size = len(self.binary_pred_colour_dict)

        for RDF_triple in dataset:
            if RDF_triple[1] == type_pred:  # Fact of form C(a), written (a type C)
                if RDF_triple[2] not in self.unary_pred_position_dict:
                    sys.exit(f"Predicate {RDF_triple[2]} not in the list of unary predicates recognised by this encoder.")
                nodename_feature_dict[RDF_triple[0]][self.unary_pred_position_dict[RDF_triple[2]]] = 1
            else:  # Fact of form R(a,b), written (a R b)
                if RDF_triple[1] not in self.binary_pred_colour_dict:
                    sys.exit(f"Predicate {RDF_triple[1]} not in the list of binary predicates recognised by this encoder.")
                if RDF_triple[0] not in nodename_feature_dict:
                    nodename_feature_dict[RDF_triple[0]] = torch.zeros(delta, dtype=torch.float)
                if RDF_triple[2] not in nodename_feature_dict:
                    nodename_feature_dict[RDF_triple[2]] = torch.zeros(delta, dtype=torch.float)
                edges.add((RDF_triple[0], RDF_triple[2], RDF_triple[1]))

        features = torch.FloatTensor(torch.stack(list(nodename_feature_dict.values())))
        assert features.shape[1] == delta
        node_names = list(nodename_feature_dict.keys()) # Correctness of this relies on dictionaries being ordered.
        edge_list = []
        edge_colour_list = []
        for (oc, dc, pred) in edges:
            edge_list.append([node_names.index(oc),node_names.index(dc)])
            edge_colour_list.append(self.binary_pred_colour_dict[pred])

        return CDGraph(col_size=col_size, delta=len(self.unary_pred_position_dict),
                       features=features, edges= torch.LongTensor(torch.LongTensor(edge_list).t().contiguous()),
                       edge_colours=torch.LongTensor(edge_colour_list), node_names=node_names)

    # Returns a dictionary where the keys are cd_facts and the values are their scores
    def decode_graph(self, cd_graph: CDGraph, threshold):

        facts_scores_dict = {}
        for i, j in torch.nonzero(cd_graph.features > threshold).tolist():
            facts_scores_dict[(cd_graph.node_names[i], type_pred, self.unary_pred_position_dict.inverse[j])] = (
                cd_graph.features[i,j].item())

        return facts_scores_dict

class NonCanonicalEncoder(ABC):

    canonical_unary_predicates = list
    canonical_binary_predicates = list

    @abstractmethod
    def encode_dataset(self, dataset: set[tuple], **kwargs) -> set[tuple]:
        pass

    @abstractmethod
    def decode_dataset(self, dataset: set[tuple]) -> set[tuple]:
        pass

    @abstractmethod
    def decode_fact(self, s: str, p:str, o:str) -> tuple[str, str, str]:
        pass

class IdentityEncoderDecoder(NonCanonicalEncoder):

    def __init__(self, load_from_document=None, unary_predicates=None, binary_predicates=None):
        self.canonical_unary_predicates = []
        self.canonical_binary_predicates = []
        if load_from_document is not None:
            for line in open(load_from_document, 'r').readlines():
                predicate, _, arity = line.split() # predicates are duplicate so we ignore the second
                arity = int(arity)
                if arity == 1:
                    self.canonical_unary_predicates.append(predicate)
                else:
                    self.canonical_binary_predicates.append(predicate)
        else:
            self.canonical_binary_predicates = binary_predicates
            self.canonical_unary_predicates = unary_predicates

    def encode_dataset(self, dataset, **kwargs):
        return dataset

    def decode_dataset(self, dataset):
        return dataset

    def decode_fact(self, s, p, o):
        return s, p, o

    # Save format (predicate \t new_predicate \t arity)
    def save_to_file(self, target_file):
        output = open(target_file, 'w')
        for predicate in self.canonical_unary_predicates:
            output.write("{}\t{}\t{}\n".format(predicate, predicate, 1))
        for predicate in self.canonical_binary_predicates:
            output.write("{}\t{}\t{}\n".format(predicate, predicate, 2))
        output.close()


class ICLREncoderDecoder(NonCanonicalEncoder):

    def __init__(self, load_from_document=None, unary_predicates=None, binary_predicates=None):

        # Fresh predicates that correspond to colours c1, c2, c3, c4 in the paper
        self.canonical_binary_predicates = ["binary-pred-1", "binary-pred-2", "binary-pred-3", "binary-pred-4"]

        self.input_predicate_to_unary_canonical_dict = bidict()
        self.input_predicate_to_arity = {}

        if load_from_document is not None:
            for line in open(load_from_document, 'r').readlines():
                input_predicate, canonical_predicate, arity = line.split()
                self.input_predicate_to_unary_canonical_dict[input_predicate] = canonical_predicate
                self.input_predicate_to_arity[input_predicate] = int(arity)
        else:
            assert set(unary_predicates).isdisjoint(set(binary_predicates)) # Sanity check
            for pred in unary_predicates:
                self.input_predicate_to_unary_canonical_dict[pred] = pred
                self.input_predicate_to_arity[pred] = 1
            for pred in binary_predicates:
                self.input_predicate_to_unary_canonical_dict[pred] = "unary-for-{}".format(pred)
                self.input_predicate_to_arity[pred] = 2

        # Maps pairs of constants to a new single term
        self.pair_term_dict = bidict()
        self.canonical_unary_predicates = list(self.input_predicate_to_unary_canonical_dict.inverse.keys())

    # Save format (predicate \t new_predicate \t arity)
    def save_to_file(self, target_file):
        output = open(target_file, 'w')
        for input_predicate in self.input_predicate_to_unary_canonical_dict:
            output.write("{}\t{}\t{}\n".format(input_predicate,
                                               self.input_predicate_to_unary_canonical_dict[input_predicate],
                                               self.input_predicate_to_arity[input_predicate]))
        output.close()

    def term_for_pair(self, pair):
        if pair not in self.pair_term_dict:
            self.pair_term_dict[pair] = "term-for-{}-{}".format(pair[0], pair[1])
        return self.pair_term_dict[pair]

    # Returns a new dataset over the new signature
    def encode_dataset(self, dataset, use_dummy_constants=False):
        encoded_dataset = []
        col1 = self.canonical_binary_predicates[0] # Abbreviations to make the code match the paper notation
        col2 = self.canonical_binary_predicates[1]
        col3 = self.canonical_binary_predicates[2]
        col4 = self.canonical_binary_predicates[3]
        for s, p, o in dataset:
            if p == type_pred:
                encoded_dataset.append((s, p, self.input_predicate_to_unary_canonical_dict[o]))
            else:
                # The following are simply renames to the paper notation, to make the code easier to read and write
                a = s
                b = o
                ab = self.term_for_pair((a, b))
                ba = self.term_for_pair((b, a))
                encoded_dataset.append((ab, type_pred, self.input_predicate_to_unary_canonical_dict[p]))
                encoded_dataset.extend([(a, col1, ab), (ab, col1, a), (b, col1, ba), (ba, col1, b)])
                encoded_dataset.extend([(b, col2, ab), (ab, col2, b), (a, col2, ba), (ba, col2, a)])
                encoded_dataset.extend([(ab, col3, ba),(ba, col3, ab)])
                encoded_dataset.extend([(a, col4, b), (b, col4, a)])

        if use_dummy_constants:
            # Extract all constants from encoded_dataset. Less efficient than doing it on-the-fly, but it's cleaner code
            constants = set()
            for s, p, o in encoded_dataset:
                constants.add(s)
                if p != type_pred:
                    constants.add(o)
            for a in ['#', '##']:
                for b in constants:
                    ab = self.term_for_pair((a, b))
                    ba = self.term_for_pair((b, a))
                    encoded_dataset.extend([(a, col1, ab), (ab, col1, a), (b, col1, ba), (ba, col1, b)])
                    encoded_dataset.extend([(b, col2, ab), (ab, col2, b), (a, col2, ba), (ba, col2, a)])
                    encoded_dataset.extend([(ab, col3, ba), (ba, col3, ab)])
                    encoded_dataset.extend([(a, col4, b), (b, col4, a)])

        return encoded_dataset

    def get_canonical_equivalent(self, fact):
        (s, p, o) = fact
        if p == type_pred:
            return s, p, self.input_predicate_to_unary_canonical_dict[o]
        else:
            ab = self.term_for_pair((s, o))
            return ab, type_pred, self.input_predicate_to_unary_canonical_dict[p]

    def decode_dataset(self, canonical_dataset):
        return {self.decode_fact(s, p, o) for s, p, o in canonical_dataset}

    def decode_fact(self, s, p, o):
        # All facts in an encoded dataset are unary; binary facts should not be decoded.
        assert(p == type_pred)
        if s in self.pair_term_dict.values():
            a, b = self.pair_term_dict.inverse[s]
            return a, self.input_predicate_to_unary_canonical_dict.inverse[o], b
        else:
            # The node is a for a single constant.
            return s, type_pred, self.input_predicate_to_unary_canonical_dict.inverse[o]

    # Maps a canonical predicate to the arity of the original predicate, either 1 or 2
    def original_arity(self, canonical_predicate):
        return self.input_predicate_to_arity[self.input_predicate_to_unary_canonical_dict.inverse[canonical_predicate]]


    def unfold(self, rule_body, unary_head_predicate):
        # each variable in the canonical rule corresponds to a unary node,
        # in which case it should be matched to a single variable in the unfolded rule,
        # or it represents a binary node,
        # in which case it should be matched to a pair of variables in the unfolded rule.
        # We deduce this inductively, by the connection of this variable to the head variable.
        can_variables_to_unfolded_variables = {} # Maps a variable to a tuple of 1 or 2 variables
        new_body = []

        # first, figure out the arity of the head variable, and assign corresponding variables in the data rule
        if self.original_arity(unary_head_predicate) == 1:
            can_variables_to_unfolded_variables["X1"] = ["X1"]
        else:
            can_variables_to_unfolded_variables["X1"] = ["X1", "X2"]

        # if we encounter a unary predicate U(y) with y a binary variable, and we don't know which variables it is
        # associated to, we just delay processing it until the next round. This won't delay it indefinitely, since in
        # each round we always get to define one additional variable.
        next_round = rule_body
        new_variables_counter = 0
        # Here we store sets of two variables {y,z} such that either R(y,z) or R(z,y) must appear in the body for some R
        top_facts = set()

        while next_round:
            this_round = next_round.copy()
            next_round = []
            for (s, p, o) in this_round:
                if s in can_variables_to_unfolded_variables:
                    if p == type_pred:
                        if self.original_arity(o) == 1 and len(can_variables_to_unfolded_variables[s]) == 1:
                            # Fact of the form A(x) in the data rule
                            new_body.append((can_variables_to_unfolded_variables[s], type_pred, self.get_data_predicate(o)))
                        elif self.original_arity(o) == 2 and len(can_variables_to_unfolded_variables[s]) == 2:
                            # Fact of the form R(x,y) in the data rule
                            new_body.append((can_variables_to_unfolded_variables[s][0], self.get_data_predicate(o), can_variables_to_unfolded_variables[s][1]))
                        else:
                            raise Exception("Error: arity of variable does not match arity of predicate.")
                    # This is the only case we need to cover inequalities. We wait until both variables are matched to unfolded variables
                    elif p == ineq_pred:
                        if o in can_variables_to_unfolded_variables:
                            # Both variables need to be of the same type. Also, if both are binary, the corresponding
                            # two pairs of unfolded variables must have one element in common and in the same position.
                            if len(can_variables_to_unfolded_variables[s]) == 1:
                                assert len(can_variables_to_unfolded_variables[o]) == 1
                                new_body.append((can_variables_to_unfolded_variables[s][0], ineq_pred,
                                                 can_variables_to_unfolded_variables[o][0]))
                            else:
                                assert len(can_variables_to_unfolded_variables[s]) == 2
                                assert len(can_variables_to_unfolded_variables[o]) == 2
                                if can_variables_to_unfolded_variables[s][0] == can_variables_to_unfolded_variables[o][0]:
                                    new_body.append((can_variables_to_unfolded_variables[s][1], ineq_pred, can_variables_to_unfolded_variables[o][1]))
                                else:
                                    assert can_variables_to_unfolded_variables[s][1] == can_variables_to_unfolded_variables[o][1]
                                    new_body.append((can_variables_to_unfolded_variables[s][0], ineq_pred, can_variables_to_unfolded_variables[o][0]))
                        else:
                            next_round.append((s, p, o))
                    else:
                        if p == self.binary_canonical[1]:
                            if len(can_variables_to_unfolded_variables[s]) == 1:
                                # Fact of the form Ec1(f(x),g(x,y)) in the canonical rule
                                if o not in can_variables_to_unfolded_variables:
                                    new_variables_counter += 1
                                    y = "Y{}".format(new_variables_counter)
                                    can_variables_to_unfolded_variables[o] = [can_variables_to_unfolded_variables[s][0], y]
                                    top_facts.add(frozenset((can_variables_to_unfolded_variables[s][0], y)))
                            else:
                                # Fact of the form Ec1((g(x,y),f(x)) in the canonical rule
                                assert(len(can_variables_to_unfolded_variables[s])) == 2
                                if o not in can_variables_to_unfolded_variables:
                                    can_variables_to_unfolded_variables[o] = [can_variables_to_unfolded_variables[s][0]]
                                top_facts.add(frozenset((can_variables_to_unfolded_variables[s][0], can_variables_to_unfolded_variables[s][1])))
                        elif p == self.binary_canonical[2]:
                            if len(can_variables_to_unfolded_variables[s]) == 1:
                                # Fact of the form Ec2(f(x),g(y,x)) in the canonical rule
                                if o not in can_variables_to_unfolded_variables:
                                    new_variables_counter += 1
                                    y = "Y{}".format(new_variables_counter)
                                    can_variables_to_unfolded_variables[o] = [y, can_variables_to_unfolded_variables[s][0]]
                                    top_facts.add(frozenset((can_variables_to_unfolded_variables[s][0], y)))
                            else:
                                # Fact of the form Ec2((g(x,y),f(y)) in the canonical rule
                                if o not in can_variables_to_unfolded_variables:
                                    can_variables_to_unfolded_variables[o] = [can_variables_to_unfolded_variables[s][1]]
                                top_facts.add(frozenset((can_variables_to_unfolded_variables[s][0], can_variables_to_unfolded_variables[s][1])))
                        elif p == self.binary_canonical[3]:
                            # Fact of the form Ec3(g(x,y),g(y,x)) in the canonical rule
                            assert len(can_variables_to_unfolded_variables[s]) == 2
                            if o not in can_variables_to_unfolded_variables:
                                can_variables_to_unfolded_variables[o] = [can_variables_to_unfolded_variables[s][1],
                                                                      can_variables_to_unfolded_variables[s][0]]
                            top_facts.add(frozenset((can_variables_to_unfolded_variables[s][0], can_variables_to_unfolded_variables[s][1])))
                        elif p == self.binary_canonical[4]:
                            # Fact of the form Ec4(f(x),f(y)) in the canonical rule
                            assert len(can_variables_to_unfolded_variables[s]) == 1
                            if o not in can_variables_to_unfolded_variables:
                                new_variables_counter += 1
                                y = "Y{}".format(new_variables_counter)
                                can_variables_to_unfolded_variables[o] = [y]
                                top_facts.add(frozenset((can_variables_to_unfolded_variables[s][0],  y)))
                        else:
                            raise Exception("Error: binary predicate not corresponding to one of the four colours")
                elif o in can_variables_to_unfolded_variables:
                    assert(p != type_pred)
                    if p == self.binary_canonical[1]:
                        if len(can_variables_to_unfolded_variables[o]) == 1:
                            # Fact of the form Ec1(g(x,y),f(x)) in the canonical rule
                            if s not in can_variables_to_unfolded_variables:
                                new_variables_counter += 1
                                y = "Y{}".format(new_variables_counter)
                                can_variables_to_unfolded_variables[s] = [can_variables_to_unfolded_variables[o][0], y]
                                top_facts.add(frozenset((can_variables_to_unfolded_variables[o][0], y)))
                        else:
                            # Fact of the form Ec1(f(x),g(x,y)) in the canonical rule
                            if s not in can_variables_to_unfolded_variables:
                                can_variables_to_unfolded_variables[s] = [can_variables_to_unfolded_variables[o][0]]
                            top_facts.add(frozenset((can_variables_to_unfolded_variables[o][0], can_variables_to_unfolded_variables[o][1])))
                    elif p == self.binary_canonical[2]:
                        if len(can_variables_to_unfolded_variables[o]) == 1:
                            # Fact of the form Ec2((g(x,y),f(y))in the canonical rule
                            if s not in can_variables_to_unfolded_variables:
                                new_variables_counter += 1
                                y = "Y{}".format(new_variables_counter)
                                can_variables_to_unfolded_variables[s] = [y, can_variables_to_unfolded_variables[o][0]]
                                top_facts.add(frozenset((can_variables_to_unfolded_variables[o][0], y)))
                        else:
                            # Fact of the form Ec2(f(x),g(y,x))  in the canonical rule
                            if s not in can_variables_to_unfolded_variables:
                                can_variables_to_unfolded_variables[s] = [can_variables_to_unfolded_variables[o][1]]
                            top_facts.add(frozenset((can_variables_to_unfolded_variables[o][0], can_variables_to_unfolded_variables[o][1])))
                    elif p == self.binary_canonical[3]:
                        # Fact of the form Ec3(g(x,y),g(y,x)) in the canonical rule
                        assert len(can_variables_to_unfolded_variables[o]) == 2
                        if s not in can_variables_to_unfolded_variables:
                            can_variables_to_unfolded_variables[s] = [can_variables_to_unfolded_variables[o][1],
                                                                  can_variables_to_unfolded_variables[o][0]]
                        top_facts.add(frozenset((can_variables_to_unfolded_variables[o][0], can_variables_to_unfolded_variables[o][1])))
                    elif p == self.binary_canonical[4]:
                        # Fact of the form Ec4(f(x),f(y)) in the canonical rule
                        assert len(can_variables_to_unfolded_variables[o]) == 1
                        if s not in can_variables_to_unfolded_variables:
                            new_variables_counter += 1
                            y = "Y{}".format(new_variables_counter)
                            can_variables_to_unfolded_variables[s] = [y]
                            top_facts.add(frozenset((can_variables_to_unfolded_variables[o][0], y)))
                    else:
                        raise Exception("Error: binary predicate not corresponding to one of the four colours")
                else:
                    next_round.append((s, p, o))
        # Little optimisation: remove top facts that are unnecessary because the required variables already apear together
        for (s, p, o) in new_body:
            if frozenset((s, o)) in top_facts:
                top_facts.remove(frozenset((s, o)))
        return new_body, can_variables_to_unfolded_variables, top_facts


    # This is a rather specific function. Given two data variables y1 and y2, this returns a single variable if y1 y2
    # correspond to a single canonical variable y, and two variables if they correspond to a canonical variable each.
    def find_canonical_variable(self, can_variables_to_data_variables, y1, y2):
        binary = None
        unary_y1 = None
        unary_y2 = None
        for cvar in can_variables_to_data_variables:
            if len(can_variables_to_data_variables[cvar]) == 2 and (
                    (can_variables_to_data_variables[cvar][0] == y1 and can_variables_to_data_variables[cvar][1] == y2) or
                    (can_variables_to_data_variables[cvar][0] == y2 and can_variables_to_data_variables[cvar][1] == y1)):
                binary = cvar
            elif len(can_variables_to_data_variables[cvar]) == 1:
                if can_variables_to_data_variables[cvar][0] == y1:
                    unary_y1 = cvar
                elif can_variables_to_data_variables[cvar][0] == y2:
                    unary_y2 = cvar
        if binary:
            return [binary]
        elif unary_y1 and unary_y2:
            return [unary_y1, unary_y2]
        else:
            raise Exception("Error: data variables {} and {} do not seem to match any canonical variable. Bug.".format(y1,y2))
