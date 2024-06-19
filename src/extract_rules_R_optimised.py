#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch_geometric.data import Data
import argparse
from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder
import numpy as np
import nodes
import datetime
import random
from random import sample
from utils import load_predicates

parser = argparse.ArgumentParser(description="Extract full equivalent program from the GNN")

parser.add_argument('--load-model-name',
                    help='Name of the file with the trained model')
parser.add_argument('--threshold',
                    type=float,
                    default=0.5,
                    help='Classification threshold of the model')
parser.add_argument('--predicates',
                    help='Name of the file with the predicates of the signature ')
parser.add_argument('--output',
                    help='Name of the file  where the extracted rules will be stored.',
                    default=None)
parser.add_argument('--canonical-encoder-file',
                    help='File with the canonical encoder/decoder used to train the model.')
parser.add_argument('--iclr22-encoder-file',
                    default=None,
                    help='File with the iclr22 encoder/decoder used to train the model, if it was used.')
parser.add_argument('--encoding-scheme',
                    default='canonical',
                    nargs='?',
                    choices=['iclr22', 'canonical'],
                    help='Choose the encoder-decoder that will be applied to the data (canonical by default).')

args = parser.parse_args()
type_pred = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'


def relevant(input_positions, mat):
    output_positions = []
    for j in range(mat.size()[1]):
        rel = False
        for ii in range(mat.size()[0]):
            if ii in input_positions and mat[ii][j] != 0:
                rel = True
        if rel:
            output_positions.append(j)
    return output_positions


if __name__ == "__main__":

    print(datetime.datetime.now())

    if args.encoding_scheme == 'iclr22':
        iclr_encoder_decoder = ICLREncoderDecoder(load_from_document=args.iclr22_encoder_file)
    can_encoder_decoder = CanonicalEncoderDecoder(load_from_document=args.canonical_encoder_file)

    # gd stands for "graph dataset"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.load_model_name).to(device)

    canonical_program = []
    variable_counter = 1  # X1 is always a variable
    next_rep = {}

    if args.output is not None:
        output_file = open(args.output, 'w')

    def merge_sort(list1,list2):
        return sorted(list(set(list1+list2)))

    for i in range(model.dimensions[-1]):
        print("Constructing Gamma^i_all for position {}".format(i))
        # Construct \Gamma_i^all and the function \mu mapping stuff to masks
        # mask: a representation of mu as a nested dictionary: layer -> variable -> relevant positions
        c1 = can_encoder_decoder.binary_pred_colour_dict[iclr_encoder_decoder.binary_canonical[1]]
        c2 = can_encoder_decoder.binary_pred_colour_dict[iclr_encoder_decoder.binary_canonical[2]]
        c3 = can_encoder_decoder.binary_pred_colour_dict[iclr_encoder_decoder.binary_canonical[3]]
        # x is the head variable, y1, y2, y3 are the unique variables connected to it by colours c1, c2, c3, resp.
        # z11, z12, z21, z22 represent the results of aggregating for variables y1 and y2 in layer 1
        x, y1, y2, y3, z11, z12, z21, z22 = "X1", "Y1", "Y2", "Y3", "Z11", "Z12", "Z21", "Z22"
        variables = [x, y1, y2, y3, z11, z12, z21, z22]
        variable_to_index = {vari: indi for indi, vari in enumerate(variables)}
        # This graph is always present whenever we deduce a fact of the form R(x,y) in the ICLR encoding.
        base_graph = [(x, iclr_encoder_decoder.binary_canonical[1], y1),
                      (y1, iclr_encoder_decoder.binary_canonical[1], x),
                      (x, iclr_encoder_decoder.binary_canonical[2], y2),
                      (y2, iclr_encoder_decoder.binary_canonical[2], x),
                      (x, iclr_encoder_decoder.binary_canonical[3], y3),
                      (y3, iclr_encoder_decoder.binary_canonical[3], x),
                      (y1, iclr_encoder_decoder.binary_canonical[2], y3),
                      (y3, iclr_encoder_decoder.binary_canonical[2], y1),
                      (y2, iclr_encoder_decoder.binary_canonical[1], y3),
                      (y3, iclr_encoder_decoder.binary_canonical[1], y2),
                      (y1, iclr_encoder_decoder.binary_canonical[4], y2),
                      (y2, iclr_encoder_decoder.binary_canonical[4], y1),
                      # Second level
                      (y1, iclr_encoder_decoder.binary_canonical[1], z11),
                      (z11, iclr_encoder_decoder.binary_canonical[1], y1),
                      (y1, iclr_encoder_decoder.binary_canonical[2], z12),
                      (z12, iclr_encoder_decoder.binary_canonical[2], y1),
                      (y2, iclr_encoder_decoder.binary_canonical[1], z21),
                      (z21, iclr_encoder_decoder.binary_canonical[1], y2),
                      (y2, iclr_encoder_decoder.binary_canonical[2], z22),
                      (z22, iclr_encoder_decoder.binary_canonical[2], y2)]
        relevant_positions = {2: {x: [i]},
                              1: {x: [], y1: [], y2: [], y3: []},
                              0: {x: [], y1: [], y2: [], y3: [], z11: [], z12: [], z21: [], z22: []}}
        #  LAYER 1
        relevant_positions[1][x] = relevant(relevant_positions[2][x], model.matrix_A(2))
        relevant_positions[1][y1] = relevant(relevant_positions[2][x], model.matrix_B(2, c1))
        relevant_positions[1][y2] = relevant(relevant_positions[2][x], model.matrix_B(2, c2))
        relevant_positions[1][y3] = relevant(relevant_positions[2][x], model.matrix_B(2, c3))
        #  LAYER 0
        # x
        relevant_positions[0][x] = relevant(relevant_positions[1][x], model.matrix_A(1))
        # ignore colours c1 and c2 because there are no unary facts
        relevant_positions[0][y3] = relevant(relevant_positions[1][x], model.matrix_B(1, c3))
        # y1 has no relevant positions, since there are no unaries
        # y2 has no relevant positions, since there are no unaries
        relevant_positions[0][z11] = relevant(relevant_positions[1][y1], model.matrix_B(1, c1))
        relevant_positions[0][z12] = relevant(relevant_positions[1][y1], model.matrix_B(1, c2))
        relevant_positions[0][z21] = relevant(relevant_positions[1][y2], model.matrix_B(1, c1))
        relevant_positions[0][z22] = relevant(relevant_positions[1][y2], model.matrix_B(1, c2))
        # y3
        relevant_positions[0][y3] = merge_sort(relevant_positions[0][y3], relevant(relevant_positions[1][y3], model.matrix_A(1)))
        # ignore colours c1 and c2 because there are no unary facts
        relevant_positions[0][x] = merge_sort(relevant_positions[0][x], relevant(relevant_positions[1][y3], model.matrix_B(1, c3)))

        relevant_variables = [variab for variab in variables if len(relevant_positions[0][variab]) > 0]
        N = len(relevant_variables)
        # In the lattice exploration, bodies of candidate rules are represented as an N-tuple of numbers.
        # Each element in the tuple is a number corresponding to a variable var.
        # The binary translation of the number (read right-to-left) is a mask for the predicates in the positions of
        # relevant_positions[0][var]
        # For example, if relevant[0][y] = (1,3,6), conjunction U1(y) ^ U6(y) is 5 (101).
        # Binary atoms are not represented since they are always present

        def number_to_mask(num, pad_to=0):
            if num == -1:
                return ""
            return bin(num)[2:].zfill(pad_to)[::-1]  # Encode indices in a binary string.


        def mask_to_number(ind_string):
            return int(ind_string[::-1], base=2)


        def evaluate_rule(head_predicate_position, candidate_body):
            # Given a rule, evaluate whether it is captured by the model.
            # Optimised model application where we only apply all relevant operations.
            # This should contain the vectors for all the variables
            vectors_layer_0 = {}
            for var_index in range(8):
                vectors_layer_0[var_index] = torch.zeros(model.dimensions[0])
            for var_index in range(N):
                y_var = relevant_variables[var_index]
                if candidate_body[var_index] > 0:
                    mask = number_to_mask(candidate_body[var_index], pad_to=len(relevant_positions[0][y_var]))
                    for ii, jj in enumerate(relevant_positions[0][y_var]):
                        if mask[ii] == '1':
                            vectors_layer_0[variable_to_index[y_var]][jj] = 1
            vectors_layer_1 = {}
            vectors_layer_1[0] = torch.relu(torch.add(
                torch.add(torch.matmul(model.matrix_A(1),vectors_layer_0[0]),torch.matmul(model.matrix_B(1,c3),vectors_layer_0[3])),
                model.bias(1)))
            vectors_layer_1[1] = torch.relu(torch.add(
                torch.add(torch.matmul(model.matrix_B(1,c1), vectors_layer_0[4]), torch.matmul(model.matrix_B(1, c2), vectors_layer_0[5])),
                model.bias(1)))
            vectors_layer_1[2] = torch.relu(torch.add(
                torch.add(torch.matmul(model.matrix_B(1, c1), vectors_layer_0[6]), torch.matmul(model.matrix_B(1, c2), vectors_layer_0[7])),
                model.bias(1)))
            vectors_layer_1[3] = torch.relu(torch.add(
                torch.add(torch.matmul(model.matrix_A(1), vectors_layer_0[3]), torch.matmul(model.matrix_B(1, c3),vectors_layer_0[0])),
                model.bias(1)))
            return torch.nn.Sigmoid()(torch.add(torch.add(torch.add(torch.add(
                torch.matmul(model.matrix_A(2)[head_predicate_position],vectors_layer_1[0]), torch.matmul(model.matrix_B(2,c1)[head_predicate_position],vectors_layer_1[1])),
                torch.matmul(model.matrix_B(2,c2)[head_predicate_position],vectors_layer_1[2])), torch.matmul(model.matrix_B(2,c3)[head_predicate_position], vectors_layer_1[3])),
                model.bias(2)[head_predicate_position]).subtract(10)).detach() >= args.threshold


        def get_successors(candidate_body):
            successors = []
            # To avoid repetition, we change zeros to ones only from the first zero after the last 1 in (the binary
            # representation of) the node.
            var_index = N-1
            var_with_one_found = False
            while (not var_with_one_found) and var_index >= 0:
                y_var = relevant_variables[var_index]
                mask = number_to_mask(candidate_body[var_index], pad_to=len(relevant_positions[0][y_var]))
                mask_index = len(mask) - 1
                one_found = False
                while (not one_found) and mask_index >= 0:
                    if mask[mask_index] == '0':
                        new_mask = mask[:mask_index] + '1' + mask[mask_index+1:]
                        new_body = candidate_body[:var_index] + (mask_to_number(new_mask),) + candidate_body[var_index+1:]
                        successors.append(new_body)
                    else:
                        one_found = True
                        var_with_one_found = True
                    mask_index = mask_index - 1
                var_index = var_index - 1
            return successors


        def get_predeccessors(candidate_body):
            predecessors = []
            # To avoid repetition, we change ones to zeros only from the first one after the last zero in (the binary
            # representation of) the node.
            var_index = N - 1
            var_with_zero_found = False
            while (not var_with_zero_found) and var_index >= 0:
                y_var = relevant_variables[var_index]
                mask = number_to_mask(candidate_body[var_index], pad_to=len(relevant_positions[0][y_var]))
                mask_index = len(mask) - 1
                zero_found = False
                while (not zero_found) and mask_index >= 0:
                    if mask[mask_index] == '1':
                        new_mask = mask[:mask_index] + '0' + mask[mask_index + 1:]
                        new_body = candidate_body[:var_index] + (mask_to_number(new_mask),) + candidate_body[var_index + 1:]
                        predecessors.append(new_body)
                    else:
                        zero_found = True
                        var_with_zero_found = True
                    mask_index = mask_index - 1
                var_index = var_index - 1
            return predecessors

        def is_smaller_or_equal(candidate_1, candidate_2):
            for val1, val2 in zip(candidate_1, candidate_2):
                if val1 >= 0:
                    if val2 < 0:
                        return False
                    else:
                        if (val1 | val2) != val2:
                            return False
            return True

        print("Exploring lattice for {}".format(i))
        # Explore the lattice
        total_levels = 0
        for var in relevant_positions[0]:
            for pos in relevant_positions[0][var]:
                total_levels += 1

        # Start with a top-down short exploration, to optimise

        min_pos = []
        next_frontier = []  # pair: first element is number 1 (variable X1) and second is a tuple that just contains 0
        empty_body = []
        full_body = []
        for v_index in range(N):
            empty_body.append(0)
            full_body.append(mask_to_number('1' * len(relevant_positions[0][relevant_variables[v_index]])))
        full_body = tuple(full_body)
        empty_body = tuple(empty_body)
        if evaluate_rule(i, full_body):
            if evaluate_rule(i, empty_body):
                min_pos = [empty_body]
            else:
                next_frontier = [empty_body]
        else:
            print("No rule captured for head predicate position {}".format(i))
            min_pos = []
        counter = 0
        while next_frontier:
            counter += 1
            print("Frontier iterations: {}/{}".format(counter, total_levels))
            print("Frontier size: {}".format(len(next_frontier)))
            print("MinPos size: {}".format(len(min_pos)))
            frontier = next_frontier.copy()
            next_frontier = []
            while frontier:
                body_candidate = frontier.pop()
                for succ in get_successors(body_candidate):
                    if evaluate_rule(i, succ):
                        subsumed = False
                        min_pos_copy = min_pos.copy()
                        while not subsumed and min_pos_copy:
                            pos_conj = min_pos_copy.pop()
                            if is_smaller_or_equal(pos_conj, succ):
                                subsumed = True
                        if not subsumed:
                            min_pos.append(succ)
                    else:
                        next_frontier.append(succ)

        # Write the rule
        cd_fact_predicate = can_encoder_decoder.position_unary_pred_dict[i]
        for body_candidate in min_pos:
            body = set(base_graph)
            for v_index in range(N):
                var_y = relevant_variables[v_index]
                unaries_mask = number_to_mask(body_candidate[v_index])
                for index_unary_mask, unary_mask_values in enumerate(unaries_mask):
                    if unary_mask_values == '1':
                        body.add((var_y, type_pred, can_encoder_decoder.position_unary_pred_dict[relevant_positions[0][var_y][index_unary_mask]]))
            if args.encoding_scheme == "iclr22":
                body, can_variable_to_data_variable, top_facts = iclr_encoder_decoder.unfold(body, cd_fact_predicate)
            # Process top facts
            for pair in top_facts:
                [y1, y2] = list(pair)
                body.append((y1, "TOP", y2))
            # Add body atoms
            body_atoms = []
            for (s, p, o) in body:
                if p == type_pred:
                    body_atoms.append("<{}>[?{}]".format(o, s))
                else:
                    body_atoms.append("<{}>[?{},?{}]".format(p, s, o))
            # Add head atoms
            if args.encoding_scheme == 'canonical':
                head = "<{}>[?X1]".format(can_encoder_decoder.position_unary_pred_dict[i])
            else:
                if iclr_encoder_decoder.data_predicate_to_arity[
                    iclr_encoder_decoder.unary_canonical_to_input_predicate_dict[cd_fact_predicate]] == 1:
                    head = "<{}>[?X1]".format(
                        iclr_encoder_decoder.unary_canonical_to_input_predicate_dict[cd_fact_predicate])
                else:
                    head = "<{}>[?X1,?X2]".format(
                        iclr_encoder_decoder.unary_canonical_to_input_predicate_dict[cd_fact_predicate])
            rule = head + " :- " + ", ".join(body_atoms) + " .\n"
            if args.output is not None:
                output_file.write(rule + '\n')

    output_file.close()
    print(datetime.datetime.now())
