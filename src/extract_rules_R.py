#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch_geometric.data import Data
import argparse
from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder
import numpy as np
import nodes
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
        # Construct \Gamma_i^all and the mask \mu
        # For convenience during lattice exploration, \Gamma^i_all and mu are represented via three structures:
        # --mask: a representation of mu as a nested dictionary: layer -> variable -> relevant positions
        # --path_to_root: a dictionary that maps a variable to the list of atoms connecting it to the root.
        # --successor_variables: a dictionary mapping each variable to a list of its successors in the tree.

        variables = ["X1"]
        path_to_root = {"X1": []}
        successor_variables = {"X1": []}
        relevant_positions = {model.num_layers: {"X1": [i]}}
        for layer in range(model.num_layers - 1, -1, -1):  # iterate over layer L-1 to 0, both inclusive
            relevant_positions[layer] = {}
            new_variables = []
            for y in variables:
                relevant_positions[layer][y] = relevant(relevant_positions[layer+1][y], model.matrix_A(layer+1))
                for c in can_encoder_decoder.colours:
                    new_relevant_positions = relevant(relevant_positions[layer+1][y], model.matrix_B(layer+1,c))
                    for j in new_relevant_positions:
                        variable_counter += 1
                        z = "X{}".format(variable_counter)
                        new_variables.append(z)
                        relevant_positions[layer][z] = [j]
                        path_to_root[z] = path_to_root[y] + [(y, can_encoder_decoder.colour_binary_pred_dict[c], z)]
                        successor_variables[z] = []
                        successor_variables[y].append(z)
            variables = variables + new_variables
        variable_to_index = {vari: indi for indi, vari in enumerate(variables)}

    # In the lattice exploration, bodies of candidate rules are represented as a pair. The first element of the
    # pair is a number encoding (in binary) which variables are present, as a Boolean mask on the 'variables' list.
    # The second element of the pair is a tuple of numbers of length equal to the number of '1' in the first element
    # of the pair. The i-th number in this tuple corresponds to the variable y associated to the i-th '1' in the
    # first element of the pair. This number can be translated to a mask on relevant[0][y] that represents the
    # unary predicates for y in the body of the rule.
    # For example, if relevant[0][y] = (1,3,6), conjunction U1(y) ^ U6(y) is 5 (101).
    # Binary atoms are not represented since they can be computed from the list of variables and path_to_root
    # A unary mask number of -1 represents that there are no relevant unaries (distinct from 0, meaning at least
    # one relevant unary that is not present).

        def number_to_mask(num, pad_to=0):
            if num == -1:
                return ""
            return bin(num)[2:].zfill(pad_to)[::-1]  # Encode indices in a binary string.


        def mask_to_number(ind_string):
            return int(ind_string[::-1], base=2)


        def evaluate_rule(head_predicate_position, candidate_body):
            # Given a rule, evaluate whether it is captured by the model.
            rule_dataset = set()
            index_into_unaries_tuple_of_last_seen_variable = -1
            for index_in_var_mask, var_mask_value in enumerate(number_to_mask(candidate_body[0])):
                y_var = variables[index_in_var_mask]
                if var_mask_value == '1':
                    index_into_unaries_tuple_of_last_seen_variable += 1
                    rule_dataset = rule_dataset.union(set(path_to_root[y_var]))  # Add all binary facts
                    # print("candidate_body")
                    # print(number_to_mask(candidate_body[0]))
                    # print(candidate_body[1])
                    unary_mask = number_to_mask(candidate_body[1][index_into_unaries_tuple_of_last_seen_variable])
                    # print("unary mask {}".format(unary_mask))
                    for index_in_unary_mask, unary_mask_value in enumerate(unary_mask):
                        if unary_mask_value == '1':
                            # print(relevant_positions[0][y_var])
                            # print(index_in_unary_mask)
                            rule_dataset.add(
                                (y_var, type_pred, can_encoder_decoder.position_unary_pred_dict[
                                    relevant_positions[0][y_var][index_in_unary_mask]]))
            if rule_dataset:
                (features, node_to_row, edge_list, edge_colour_list) = can_encoder_decoder.encode_dataset(rule_dataset)
                rule_data = Data(x=features, edge_index=edge_list, edge_type=edge_colour_list).to(device)
                gnn_output_rule = model(rule_data)
                output_node = nodes.const_node_dict["X1"]  # This variable always appears in non-empty candidates.
                return gnn_output_rule[node_to_row[output_node]][head_predicate_position] >= args.threshold
            else:
                x = torch.FloatTensor(np.zeros((1, model.dimensions[0])))
                rule_data = Data(x=x, edge_index=torch.full((2, 0), 0, dtype=torch.long),
                                 edge_type=torch.LongTensor([])).to(device)
                gnn_output_rule = model(rule_data)
                return gnn_output_rule[0][head_predicate_position] >= args.threshold


        def get_successors(candidate_body):
            # verify(candidate_body)
            successors = set()  # Output
            indices_successors_to_introduce = []  # Indices of succs that we might need to add (may contain duplicates!)
            index_in_unaries_tuple_of_last_seen_variable = -1
            variables_mask = number_to_mask(candidate_body[0])
            for index_in_var_mask, var_mask_value in enumerate(variables_mask):
                y_var = variables[index_in_var_mask]
                if var_mask_value == '1':
                    index_in_unaries_tuple_of_last_seen_variable += 1
                    while indices_successors_to_introduce and index_in_var_mask == indices_successors_to_introduce[0]:  # successor was already present
                        indices_successors_to_introduce.pop(0)  # Use a while loop because of potential duplicates
                    # Create successors obtained by adding a unary fact for an existing variable
                    unary_mask = number_to_mask(candidate_body[1][index_in_unaries_tuple_of_last_seen_variable], pad_to=len(relevant_positions[0][y_var]))  # Pad because we need to see all possible zeroes.
                    for index_in_unary_mask, unary_mask_value in enumerate(unary_mask):
                        if unary_mask_value == '0':
                            new_unary_mask = unary_mask[:index_in_unary_mask] + '1' + unary_mask[index_in_unary_mask + 1:]
                            new_successor = (candidate_body[0], candidate_body[1][:index_in_unaries_tuple_of_last_seen_variable] + (mask_to_number(new_unary_mask),) + candidate_body[1][index_in_unaries_tuple_of_last_seen_variable+1:]) # Substitution
                            # verify(new_successor)
                            successors.add(new_successor)

                    # schedule successors for addition
                    new_successors = [variable_to_index[vari] for vari in successor_variables[y_var]]
                    indices_successors_to_introduce.extend(new_successors)
                    indices_successors_to_introduce = sorted(indices_successors_to_introduce)
                else:
                    assert var_mask_value == '0'
                    if indices_successors_to_introduce and index_in_var_mask == indices_successors_to_introduce[0]:
                        # add a successor here
                        while indices_successors_to_introduce and index_in_var_mask == indices_successors_to_introduce[0]:
                            indices_successors_to_introduce.pop(0)  # To remove duplicates.
                        new_variables_mask = variables_mask[:index_in_var_mask] + '1' + variables_mask[index_in_var_mask+1:] # Substitution
                        if relevant_positions[0][y_var]:
                            new_unaries_mask_tuple = candidate_body[1][:index_in_unaries_tuple_of_last_seen_variable+1] + (0,) + candidate_body[1][index_in_unaries_tuple_of_last_seen_variable+1:] # Insertion
                        else:
                            new_unaries_mask_tuple = candidate_body[1][ :index_in_unaries_tuple_of_last_seen_variable + 1] + (-1,) + candidate_body[1][ index_in_unaries_tuple_of_last_seen_variable + 1:]  # Insertion
                        new_successor = (mask_to_number(new_variables_mask), new_unaries_mask_tuple)
                        # verify(new_successor)
                        successors.add(new_successor)
            # Finally, take care of successors going beyond those in candidate_bod[0]
            current_index = len(variables_mask) - 1
            while indices_successors_to_introduce:
                current_index += 1
                if indices_successors_to_introduce and current_index == indices_successors_to_introduce[0]:
                    while indices_successors_to_introduce and current_index == indices_successors_to_introduce[0]:
                        indices_successors_to_introduce.pop(0)
                    new_variables_mask = variables_mask + '1'
                    y_var = variables[current_index]
                    if relevant_positions[0][y_var]:
                        new_unaries_mask_tuple = candidate_body[1] + (0,)
                    else:
                        new_unaries_mask_tuple = candidate_body[1] + (-1,)
                    new_successor = (mask_to_number(new_variables_mask), new_unaries_mask_tuple)
                    # verify(new_successor)
                    successors.add(new_successor)
                variables_mask = variables_mask + '0'
            return list(successors)

        def verify(candidate):
            var_num = candidate[0]
            unaries_tup = candidate[1]
            var_mask = number_to_mask(var_num)
            index_last_seen_variable = -1
            for var_mask_index, var_mask_value in enumerate(var_mask):
                if var_mask_value == '1':
                    index_last_seen_variable += 1
                    var_y = variables[var_mask_index]
                    unaries_mask = number_to_mask(unaries_tup[index_last_seen_variable], pad_to=len(relevant_positions[0][var_y]))
                    #print("test")
                    #print(unaries_mask)
                    #print(relevant_positions[0][var_y])
                    assert len(unaries_mask) == len(relevant_positions[0][var_y])


        def is_smaller_or_equal(candidate_1, candidate_2):
            # First, check if the first candidate has a variable not in the second
            if (candidate_1[0] | candidate_2[0]) != candidate_2[0]:
                return False
            # Then, compare the unary predicates for the variables, one by one.
            mask_1 = number_to_mask(candidate_1[0])
            mask_2 = number_to_mask(candidate_2[0])
            index_in_unaries_tuple_of_last_seen_variable_1 = -1
            index_in_unaries_tuple_of_last_seen_variable_2 = -1
            for var_mask_value_1, var_mask_value_2 in zip(mask_1, mask_2):
                if var_mask_value_1 == '1':
                    index_in_unaries_tuple_of_last_seen_variable_1 += 1
                    assert var_mask_value_2 == '1'
                    index_in_unaries_tuple_of_last_seen_variable_2 += 1
                    unary_mask_number_1 = candidate_1[1][index_in_unaries_tuple_of_last_seen_variable_1]
                    unary_mask_number_2 = candidate_2[1][index_in_unaries_tuple_of_last_seen_variable_2]
                    if (unary_mask_number_1 | unary_mask_number_2) != unary_mask_number_2:
                        return False
                elif var_mask_value_2 == '1':
                    index_in_unaries_tuple_of_last_seen_variable_2 += 1
            return True


        print("Exploring lattice for {}".format(i))
        # Explore the lattice
        total_levels = 0
        for var in relevant_positions[0]:
            total_levels += 1
            for pos in relevant_positions[0][var]:
                total_levels += 1
        min_pos = []
        next_frontier = []  # pair: first element is number 1 (variable X1) and second is a tuple that just contains 0
        if evaluate_rule(i, (1, (0,))):
            min_pos = [(1, (0,))]
        else:
            next_frontier = [(1, (0,))]
        counter = 0
        while next_frontier:
            counter += 1
            print("Frontier iterations: {}/{}".format(counter, total_levels))
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
        for cand in min_pos:
            body_atoms = []
            body = set()
            for [yVar, yIndicesNumber] in cand:
                yIndices = bin(yIndicesNumber)[2:][::-1]
                body = body.union(set(path_to_root[yVar]))
                for indexK in range(len(yIndices)):
                    if yIndices[indexK] == '1':
                        body.add((y, type_pred, can_encoder_decoder.position_unary_pred_dict[relevant_positions[0][y][indexK]]))
            if args.encoding_scheme == "iclr22":
                body, can_variable_to_data_variable, top_facts = iclr_encoder_decoder.unfold(body, cd_fact_predicate)
            # Process top facts
            for pair in top_facts:
                [y1, y2] = list(pair)
                body.append(y1, "TOP", y2)
            for (s, p, o) in body:
                if p == type_pred:
                    body_atoms.append("<{}>[?{}]".format(o, s))
                else:
                    body_atoms.append("<{}>[?{},?{}]".format(p, s, o))
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
