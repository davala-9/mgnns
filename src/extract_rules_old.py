#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import model_tools
import argparse
from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder

import nodes
from model_tools import get_element, get_dimension, relevant_colours
from random import sample
from utils import load_predicates

parser = argparse.ArgumentParser(description="Extract rules from trained GNNs")

parser.add_argument('--load-model-name',
                    help='Name of the file with the trained model')
parser.add_argument('--threshold',
                    type=float,
                    default=0.5,
                    help='Classification threshold')
parser.add_argument('--predicates',
                    help='Name of the file with the predicates of the signature ')
parser.add_argument('--output',
                    help='Name of the file  where the extracted rules will be stored.',
                    default=None)

args = parser.parse_args()

global variable_counter
variable_counter = 1  # X1 is always a variable


def get_new_variable():
    variable_counter += 1
    return "X{}".format(variable_counter)

def relevant(vec, mat):
    assert vec.size()[0] == mat.size()[0]
    ret = torch.zeros(mat.size()[1])
    for j in range(mat.size()[1]):
        for ii in range(mat.size()[0]):
            if vec[ii] != 0 and mat[ii][j] != 0:
                ret[j] = 1
    return ret


def compute_successor(body):



if __name__ == "__main__":

    model = torch.load(args.load_model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading predicates from {}".format(args.predicates))
    data_binary_predicates, data_unary_predicates = load_predicates(args.predicates)
    print("{} unary predicates and {} binary predicates in the signature.".format(
        len(data_unary_predicates), len(data_binary_predicates)))
    if args.encoding_scheme == 'canonical':
        cd_unary_predicates = data_unary_predicates
        cd_binary_predicates = data_binary_predicates
    else:
        iclr_encoder_decoder = ICLREncoderDecoder(data_unary_predicates, data_binary_predicates)
        cd_unary_predicates = iclr_encoder_decoder.canonical_unary_predicates()
        cd_binary_predicates = iclr_encoder_decoder.canonical_binary_predicates()
    print("{} unary predicates and {} binary predicates in the (col,delta) signature.".format(
        len(cd_unary_predicates), len(cd_binary_predicates)))

    can_encoder_decoder = CanonicalEncoderDecoder(cd_unary_predicates, cd_binary_predicates)

    def evaluate_rule(head_predicate_position, edges, var_to_pred_pos):

        rule_dataset = []
        for c in model.num_colours:
            for (y,z) in edges[c]:
                rule_dataset.add((y,can_encoder_decoder.colour_binary_pred_dict[c],z))
            for y in var_to_pred_pos:
                for p in var_to_pred_pos[y]:
                    rule_dataset.add((y, type_pred, p))

        (features, node_to_row, edge_list, edge_colour_list) = can_encoder_decoder.encode_dataset(body)
        rule_data = Data(x=features, edge_index=edge_list, edge_type=edge_colour_list).to(device)
        gnn_output_rule = model(rule_data)
        output_node = nodes.const_node_dict["X1"]
        return gnn_output_rule[node_to_row[output_node]][head_predicate_position] >= args.threshold



    for i in range(model.dimensions[-1]):
        v_all = set()
        all_edges = {}
        for c in model.num_colours:
            all_edges[c] = set()
        mask = {}
        for layer in range(model.num_layers + 1):
            mask[layer] = {}
        node_x = nodes.get_node_for_constant("X1")
        v_all.add(node_x)
        initial_mask = torch.zeros(model.dimensions[-1])
        initial_mask[i] = 1
        mask[model.dimension[-1]][node_x] = initial_mask
        for layer in range(model.dimensions[-1],0,-1):
            v_new = set()
            for v in v_all:
                y = nodes.node_const_dict[v]
                mask[layer-1][v] = relevant(mask[layer][v], model.matrix_A(layer))
                for c in range(model.num_colours):
                    relevant_vec = relevant(mask[layer][v], model.matrix_B(layer, c))
                    for j in range(model.dimensions[layer-1]):
                        if relevant_vec[j] == 1:
                            z = get_new_variable()
                            node_z = nodes.get_node_for_constant(z)
                            v_new.add(node_z)
                            all_edges[c].add((z, y))
                            new_mask = torch.zeros(model.dimensions[layer-1])
                            new_mask[j] = 1
                            mask[layer-1][node_z] = new_mask
            v_all = v_all.union(v_new)

        all_variables = set()
        for (y,z) in all_edges:
            all_variables.add(y)
            all_variables.add(z)
        leaves = all_variables.copy()
        for (y,z) in all_edges:
            leaves.remove(z)

        def get_successors(edges, var_to_pred_pos):
            successors = []
            for y in var_to_pred_pos:
                for i in range(model.dimensions[0]):
                    if mask[0][nodes.const_node_dict[y]] == 1 and i not in var_to_pred_pos[y]:
                        new_var_to_pred_pos = var_to_pred_pos.copy()
                        new_var_to_pred_pos[y].add(i)
                        successors.add((edges, new_var_to_pred_pos))
                for c in model.num_colours:
                    for (z, yy) in all_edges:
                        if yy == y and (z,yy) not in edges:
                            if z in leaves:
                                for j in range(model.dimensions[0]):
                                    if mask[0][nodes.const_node_dict[z]] == 1:
                                        new_edges = edges + (z,yy)
                                        new_var_to_pred_pos = var_to_pred_pos.copy()
                                        new_var_to_pred_pos[z] = {i}
                                        successors.append((new_edges, new_var_to_pred_pos))
                            else:
                                new_edges = edges + (z, yy)
                                successors.append((new_edges, var_to_pred_pos))
            return successors

        max_neg = set()
        min_pos = set()
        empty_edges = set()
        empty_var_to_pred_pos = {}
        frontier = [(empty_edges, empty_var_to_pred_pos)]
        while frontier:
            (conj_edges, conj_var_to_pred_pos) = frontier.pop()
            conj_prev = None
            while conj_edges is not None and not evaluate_rule(i,conj):
                successors = get_successors
                if successors:
                    conj_next = successors.pop()

                   #COME HERE!! Finish implementing optimised algorithm
                # Then, modify it so that the number of rules in the body is limited.












    layer_dimensions = {0: get_dimension(model, 0), 1: get_dimension(model, 1), 2: get_dimension(model, 2)}
    binaryPredicates, unaryPredicates = load_predicates(args.predicates)
    predicate_list = unaryPredicates + binaryPredicates
    num_binary = len(binaryPredicates)
    num_unary = len(unaryPredicates)
    writer = open(args.output, 'w')


    def non_zero(matrix, row):
        nonzero_column_indices = []
        for column in range(matrix.size(dim=1)):
            if matrix[row, column] > 0:
                nonzero_column_indices.append(column)
        return frozenset(nonzero_column_indices)


    for predicate in range(model_tools.get_dimension(model, 2)):
        #    for predicate in range(1):

        vector_xy: str = "XY"
        vector_yx = "YX"
        vector_xz = "XZ"
        vector_zx = "ZX"
        vector_yz = "YZ"
        vector_zy = "ZY"
        vector_x = "X"
        vector_y = "Y"
        relevant_feature_vectors = [vector_xy, vector_yx, vector_xz, vector_zx, vector_yz, vector_zy]

        # STEP 1: Extract relevant components of the 6 base vectors by looking at nonzero weights.
        relevant_components_1 = {vector_xy: set(), vector_yx: set(), vector_x: set(), vector_y: set()}
        relevant_components_0 = {vector_xy: set(), vector_yx: set(), vector_xz: set(), vector_zx: set(),
                                 vector_yz: set(), vector_zy: set()}
        relevant_components_1[vector_xy].update(non_zero(get_element(model, 2, model_tools.MATRIX_A), predicate))
        relevant_components_1[vector_x].update(non_zero(get_element(model, 2, model_tools.MATRIX_B1), predicate))
        relevant_components_1[vector_y].update(non_zero(get_element(model, 2, model_tools.MATRIX_B2), predicate))
        relevant_components_1[vector_yx].update(non_zero(get_element(model, 2, model_tools.MATRIX_B3), predicate))
        for k in relevant_components_1[vector_xy]:
            relevant_components_0[vector_xy].update(non_zero(get_element(model, 1, model_tools.MATRIX_A), k))
            relevant_components_0[vector_yx].update(non_zero(get_element(model, 1, model_tools.MATRIX_B3), k))
        #  No need to consider vertex_xy here; we can always assume that the contribution comes from a fresh edge.
        #  Indeed, any contribution by R(x,y) can be mimicked by a contribution by R(x,z) for some z. Recall that we
        #  are trying to obtain the most general rules.
        for k in relevant_components_1[vector_x]:
            relevant_components_0[vector_xz].update(non_zero(get_element(model, 1, model_tools.MATRIX_B1), k))
            relevant_components_0[vector_zx].update(non_zero(get_element(model, 1, model_tools.MATRIX_B2), k))
        for k in relevant_components_1[vector_y]:
            relevant_components_0[vector_yz].update(non_zero(get_element(model, 1, model_tools.MATRIX_B1), k))
            relevant_components_0[vector_zy].update(non_zero(get_element(model, 1, model_tools.MATRIX_B2), k))
        for k in relevant_components_1[vector_yx]:
            relevant_components_0[vector_yx].update(non_zero(get_element(model, 1, model_tools.MATRIX_A), k))
            relevant_components_0[vector_xy].update(non_zero(get_element(model, 1, model_tools.MATRIX_B3), k))
        for vector in relevant_feature_vectors:
            relevant_components_0[vector] = sorted(relevant_components_0[vector])

        # print("Relevant xy: {}".format(relevant_components_0[vector_xy]))
        # print("Relevant yx: {}".format(relevant_components_0[vector_yx]))
        # print("Relevant xz: {}".format(relevant_components_0[vector_xz]))
        # print("Relevant zx: {}".format(relevant_components_0[vector_zx]))
        # print("Relevant yz: {}".format(relevant_components_0[vector_yz]))
        # print("Relevant zy: {}".format(relevant_components_0[vector_zy]))
        node_size = len(relevant_feature_vectors)
        node_max_values = [0] * node_size
        for vector in relevant_feature_vectors:
            print(len(relevant_components_0[vector]))
            i = relevant_feature_vectors.index(vector)
            node_max_values[i] = 2 ** len(relevant_components_0[vector]) - 1


        #  Some auxiliary methods
        # def get_next(node):
        #     the_successors = set()
        #     for p in range(node_size):
        #         if node[p].item() < node_max_values[p]:
        #             subsequent_node = node.clone()
        #             subsequent_node[p] = node[p].item() + 1
        #             the_successors.add(subsequent_node)
        #     return the_successors

        # def vector_to_node_component(node, vectorr, component):
        #     new_node = node.clone()
        #     new_node[component] = int("".join(
        #             [str(int(vectorr[j].item())) for j in relevant_components_0[relevant_feature_vectors[component]]])[
        #             ::-1], base=2)
        #     return new_node

        def get_next_ordered(node):
            # To avoid repetition, we change zeros to ones only from the first zero after the last 1 in (the binary
            # representation of) the node.
            # Find node component `candidat` which contains the first 0 after the last 1.
            candidat = 0
            for p in range(node_size):
                if node[p].item() != 0:
                    if format(node[p].item(), 'b').endswith('0'):
                        candidat = p
                    else:
                        candidat = p + 1

            the_successors = set()
            for p in range(candidat, node_size):
                bin_num = node_component_to_binary(node, p)
                # Find first zero after a 1
                start = 0
                for j in range(len(bin_num)):
                    if bin_num[j] == '1':
                        start = j + 1
                for j in range(start, len(bin_num)):
                    assert bin_num[j] == '0'
                    new_bin_num = bin_num[:j] + '1' + bin_num[j + 1:]
                    new_node = node.clone()
                    new_node[p] = int(new_bin_num, 2)
                    the_successors.add(new_node)
            return the_successors


        def node_component_to_binary(a_node, component):
            current_value = a_node[component].item()
            max_value = node_max_values[component]
            return format(current_value, 'b').zfill(len(format(max_value, 'b')))


        def node_component_to_vector(node, component):
            vec = torch.zeros(model_tools.get_dimension(model, 0))
            current_value = node[component].item()
            binary_current_value = format(current_value, 'b')
            vect = relevant_feature_vectors[component]
            if relevant_components_0[vect]:  # No need to add ones if the list is empty
                for pos in range(len(binary_current_value)):
                    if binary_current_value[-(pos + 1)] == '1':
                        vec[relevant_components_0[vect][pos]] = 1
            return vec


        def vectors_to_node(vxy, vyx, vxz, vzx, vyz, vzy):
            node = [0] * node_size
            input_vectors = [vxy, vyx, vxz, vzx, vyz, vzy]
            for v in relevant_feature_vectors:
                node_index = relevant_feature_vectors.index(v)
                if relevant_components_0[v]:
                    node[node_index] = int("".join(
                        [str(int(input_vectors[node_index][j].item()))
                         for j in relevant_components_0[v]])[::-1], base=2)
            return node


        def evaluate(node):
            vectors_layer_0 = {}
            for vec in relevant_feature_vectors:
                # Transform node into vector list
                vectors_layer_0[vec] = node_component_to_vector(node, relevant_feature_vectors.index(vec))
            # for i in range(10):
            #     if count == 10**i:
            #         for k in range(16):
            #             print("Leaf vector {}".format(vectors_layer_0[k]))
            vectors_layer_1 = {}
            zero_vector = torch.zeros(model_tools.get_dimension(model, 0))
            vectors_layer_1[vector_xy] = model_tools.apply_layer_1_binary(
                model, vectors_layer_0[vector_xy], zero_vector, zero_vector, vectors_layer_0[vector_yx])
            vectors_layer_1[vector_x] = model_tools.apply_layer_1_unary(
                model, zero_vector, torch.max(vectors_layer_0[vector_xy], vectors_layer_0[vector_xz]),
                torch.max(vectors_layer_0[vector_yx], vectors_layer_0[vector_zx]), zero_vector)
            vectors_layer_1[vector_y] = model_tools.apply_layer_1_unary(
                model, zero_vector, torch.max(vectors_layer_0[vector_yx], vectors_layer_0[vector_yz]),
                torch.max(vectors_layer_0[vector_xy], vectors_layer_0[vector_zy]), zero_vector)
            vectors_layer_1[vector_yx] = model_tools.apply_layer_1_binary(
                model, vectors_layer_0[vector_yx], zero_vector, zero_vector, vectors_layer_0[vector_xy])
            # for i in range(10):
            #     if count == 10**i:
            #         for k in range(4):
            #             print("Layer 1 vector {}".format(vectors_layer_1[k]))
            return model_tools.apply_layer_2(
                model, predicate, vectors_layer_1[vector_xy], vectors_layer_1[vector_x], vectors_layer_1[vector_y],
                vectors_layer_1[vector_yx])


        def custom_evaluate(vxy, vyx, vxz, vzx, vyz, vzy):
            vectors_layer_0 = {vector_xy: vxy, vector_yx: vyx, vector_xz: vxz, vector_zx: vzx, vector_yz: vyz,
                               vector_zy: vzy}
            vectors_layer_1 = {}
            zero_vector = torch.zeros(model_tools.get_dimension(model, 0))
            vectors_layer_1[vector_xy] = model_tools.apply_layer_1_binary(
                model, vectors_layer_0[vector_xy], zero_vector, zero_vector, vectors_layer_0[vector_yx])
            vectors_layer_1[vector_x] = model_tools.apply_layer_1_unary(
                model, zero_vector, torch.max(vectors_layer_0[vector_xy], vectors_layer_0[vector_xz]),
                torch.max(vectors_layer_0[vector_yx], vectors_layer_0[vector_zx]), zero_vector)
            vectors_layer_1[vector_y] = model_tools.apply_layer_1_unary(
                model, zero_vector, torch.max(vectors_layer_0[vector_yx], vectors_layer_0[vector_yz]),
                torch.max(vectors_layer_0[vector_xy], vectors_layer_0[vector_zy]), zero_vector)
            vectors_layer_1[vector_yx] = model_tools.apply_layer_1_binary(
                model, vectors_layer_0[vector_yx], zero_vector, zero_vector, vectors_layer_0[vector_xy])
            # for i in range(10):
            #     if count == 10**i:
            #         for k in range(4):
            #             print("Layer 1 vector {}".format(vectors_layer_1[k]))
            return model_tools.apply_layer_2x(
                model, predicate, vectors_layer_1[vector_xy], vectors_layer_1[vector_x], vectors_layer_1[vector_y],
                vectors_layer_1[vector_yx])

        # def random_next(node):
        #     nonmaxxed_components = set()
        #     for x in range(node_size):
        #         if node[x].item() < node_max_values[x]:
        #             nonmaxxed_components.add(x)
        #     x = sample(nonmaxxed_components, 1)[0]
        #     subsequent_node = node.clone()
        #     subsequent_node[x] = node[x].item() + 1
        #     return subsequent_node

        # def remove_subsumed_positives(node, nodes):
        #     candidates = nodes.copy()
        #     for candidate in nodes:
        #         if all(node[x] <= candidate[x] for x in range(node_size)):
        #             candidates.remove(candidate)
        #     return candidates

        def subsumed_by(target, nodes):
            subsumed = False
            for cand in nodes:
                if not subsumed:
                    match = True  # If both are empty, by default it is a match.
                    for x in range(node_size):
                        if match:
                            binary_target = node_component_to_binary(target, x)
                            binary_candidate = node_component_to_binary(cand, x)
                            # binary_target is subsumed if doing an OR with the candidate leaves it unchanged
                            if int(binary_target, 2) != int(binary_target, 2) | int(binary_candidate, 2):
                                match = False
                    subsumed = match
            return subsumed

        # def remove_subsumed_negatives(node, nodes):
        #     candidates = nodes.copy()
        #     for candidate in nodes:
        #         if all(node[x] >= candidate[x] for x in range(node_size)):
        #             candidates.remove(candidate)
        #     return candidates


        # --------------------

        # LATTICE EXPLORATION
        minimal_positives = set()
        frontier = []
        next_level_frontier = []

        top_node = torch.zeros(node_size, dtype=torch.long)
        for i in range(node_size):
            top_node[i] = node_max_values[i]
        bottom_node = torch.zeros(node_size, dtype=torch.long)
        #print("Top node: {}".format(top_node))

        if evaluate(top_node):
            if evaluate(bottom_node):
                minimal_positives = set(bottom_node)
            else:
                next_level_frontier.append(bottom_node)
        else:
            minimal_positives = None

        # targnode = torch.tensor(vectors_to_node(
        #     torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        #     torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0.]),
        #     torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0.]),
        #     torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0.]),
        #     torch.tensor([0., 1., 1., 0., 0., 1., 0., 0., 0.]),
        #     torch.tensor([1., 1., 0., 0., 0., 1., 0., 0., 0.])
        # ))
        #print("targnode: {}".format(targnode))
        # print("EVALUA: ")
        # print(custom_evaluate(
        #     torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        #     torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0.]),
        #     torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        #     torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0.]),
        #     torch.tensor([0., 1., 1., 0., 0., 1., 0., 0., 0.]),
        #     torch.tensor([1., 1., 0., 0., 0., 1., 0., 0., 0.])
        # ))
        # targ_node = torch.tensor([int('00', 2), int('010', 2), int('000', 2), int('1', 2), int('10110', 2),
        #                           int('1011', 2)])
        # #print("targ_node: {}".format(targ_node))

        # We do it ``stratified'' to avoid adding a node to the frontier twice, hopefully frontier in memory won't
        # be too big.
        count = 0
        suma = 0
        for elem in top_node:
            suma += len(format(elem, 'b'))
        print("Total levels: {}".format(suma))
        while next_level_frontier:
            #print("Targ_node in positives: {}".format(subsumed_by(targ_node, minimal_positives)))
            count += 1
            print("Investigating level: {} ".format(count))
            #print("Number of positives: {}".format(len(minimal_positives)))
            # print("Frontier size: {}".format(len(next_level_frontier)))
            # if count % 10 == 0:
            #     print(count)
            #     print("Positives: {}".format(len(minimal_positives)))
            #     print("Negatives: {}".format(len(next_level_frontier)))
            frontier = next_level_frontier.copy()
            next_level_frontier = []
            while frontier:
                current_node = frontier.pop()
                next_nodes = get_next_ordered(current_node)
                for candidate in next_nodes:
                    if evaluate(candidate):
                        if not subsumed_by(candidate, minimal_positives):
                            minimal_positives.add(candidate)
                            # Otherwise, ignore it
                    else:
                        next_level_frontier.append(candidate)

        # print("Node that subsumes targ_node")
        # for node in minimal_positives:
        #     if all(targ_node[x] >= node[x] for x in range(node_size)):
        #         print(node)

        # If the frontier is empty, the next of every max negative is a min positive, so we have the cut
        rules = []


        # Given a binary vector, returns a conjunction of atoms of the form R_j(X,Y), for j the components of the vector
        # that are 1, X the variable 1, and Y the variable 2; if `numerate` option is active, each occurrence of the
        # variable gets a different number.
        def vector_to_conjunction(vect, variable1, numerate1, variable2, numerate2):
            counter = 0
            rul_bod = []
            def append_number(var, flag):
                if flag:
                    return var + str(counter)
                else:
                    return var
            for indx in range(vect.size(dim=0)):
                if vect[indx] == 1:
                    counter += 1
                    rul_bod.append("<{}>[?{},?{}]".format(predicate_list[indx], append_number(variable1, numerate1),
                                                          append_number(variable2, numerate2)))
            return rul_bod


        if minimal_positives:
            for the_node in minimal_positives:
                #  print(the_node)
                rule_body = ["<TOP>[?X,?Y]"]
                vec_xy = node_component_to_vector(the_node, relevant_feature_vectors.index(vector_xy))
                vec_yx = node_component_to_vector(the_node, relevant_feature_vectors.index(vector_yx))
                vec_xz = node_component_to_vector(the_node, relevant_feature_vectors.index(vector_xz))
                vec_zx = node_component_to_vector(the_node, relevant_feature_vectors.index(vector_zx))
                vec_yz = node_component_to_vector(the_node, relevant_feature_vectors.index(vector_yz))
                vec_zy = node_component_to_vector(the_node, relevant_feature_vectors.index(vector_zy))
                # Remove R(x,zi) from a rule body if you already have R(x,y), since zi appears nowhere else
                for comp in range(len(vec_xz)):
                    if vec_xz[comp] == 1 and vec_xy[comp] == 1:
                        vec_xz[comp] = 0
                    if vec_zx[comp] == 1 and vec_yx[comp] == 1:
                        vec_zx[comp] = 0
                    if vec_yz[comp] == 1 and vec_yx[comp] == 1:
                        vec_yz[comp] = 0
                    if vec_zy[comp] == 1 and vec_xy[comp] == 1:
                        vec_zy[comp] = 0
                rule_body += vector_to_conjunction(vec_xy, "X", False, "Y", False)
                rule_body += vector_to_conjunction(vec_yx, "Y", False, "X", False)
                rule_body += vector_to_conjunction(vec_xz, "X", False, "Za", True)
                rule_body += vector_to_conjunction(vec_zx, "Zb", True, "X", False)
                rule_body += vector_to_conjunction(vec_yz, "Y", False, "Zc", True)
                rule_body += vector_to_conjunction(vec_zy, "Zd", True, "Y", False)

                writer.write("<{}>[?X,?Y]".format(predicate_list[predicate]) + " :- " + ", ".join(rule_body) + " .\n")
        writer.write("<TOP>[?X,?Y] :- <{}>[?X,?Y] .\n".format(binaryPredicates[predicate]))
        writer.write("<TOP>[?X,?Y] :- <{}>[?Y,?X] .\n".format(binaryPredicates[predicate]))
