import torch
from torch_geometric.data import Data
import numpy as np
import argparse
import os.path
import data_parser
import nodes
import datetime
import itertools

from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder
from utils import type_pred
from utils import remove_redundant_atoms


from utils import load_predicates
parser = argparse.ArgumentParser(description="Extract a rules captured by the MGNN which derives a given fact on"
                                             "this dataset")

parser.add_argument('--load-model-name',
                    help='Name of the file with the trained model')
parser.add_argument('--threshold',
                    type=float,
                    default=0.5,
                    help='Classification threshold of the model')
parser.add_argument('--dataset',
                    nargs='?',
                    default=None,
                    help='Name of the file with the input dataset.')
parser.add_argument('--predicates',
                    help='Name of the file with the predicates of the signature ')
parser.add_argument('--facts',
                    help='Name of the file with the facts for which we seek an explanation')
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
parser.add_argument('--model-clamping',
                    default=None,
                    help='Clamp to zero all weights with absolute value under this number at the end')
parser.add_argument('--minimal-rule',
                    default=False,
                    action='store_true')

args = parser.parse_args()

type_pred = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'

# Optimised rule extraction algorithm: the hope is that the extracted rules are the shortest/strongest among the
# explanatory rules, and so they are the ones that make most sense.
# IMPORTANT NOTE: Currently supports only max aggregation and 2 layers.

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

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
def threshold_matrix_values(matrix: torch.tensor, threshold: float, negative_only=False):
    below_threshold_mask = matrix <= -threshold
    above_threshold_mask = matrix >= threshold
    if negative_only:
        outside_threshold_mask = torch.logical_or(below_threshold_mask, matrix >= 0)
    else:
        outside_threshold_mask = torch.logical_or(below_threshold_mask, above_threshold_mask)
    inside_threshold_mask = torch.logical_not(outside_threshold_mask)
    matrix[inside_threshold_mask] = 0

if __name__ == "__main__":

    print(datetime.datetime.now())

    # Model loading and processing (including clamping)
    model_name = args.load_model_name[:-3] # Remove extension '.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.load_model_name).to(device)
    if args.model_clamping:
        for layer in range(1, model.num_layers + 1):
            mat = model.matrix_A(layer)
            threshold_matrix_values(model.matrix_A(layer), float(args.model_clamping))
            for colour in range(model.num_colours):
                threshold_matrix_values(model.matrix_B(layer, colour), float(args.model_clamping))

    # Dataset loading and processing
    dataset_path = args.dataset
    assert os.path.exists(dataset_path)
    print("Loading graph data from {}".format(dataset_path))
    dataset = data_parser.parse(dataset_path)
    if args.encoding_scheme == 'canonical':
        cd_dataset = dataset
    else:
        iclr_encoder_decoder = ICLREncoderDecoder(load_from_document=args.iclr22_encoder_file)
        cd_dataset = iclr_encoder_decoder.encode_dataset(dataset)
    can_encoder_decoder = CanonicalEncoderDecoder(load_from_document=args.canonical_encoder_file)
    # gd stands for "graph dataset"
    (gd_features, node_to_gd_row_dict, gd_edge_list, gd_edge_colour_list) = can_encoder_decoder.encode_dataset(
        cd_dataset)
    gd_row_to_node_dict = {node_to_gd_row_dict[node]: node for node in node_to_gd_row_dict}
    gd_data = Data(x=gd_features, edge_index=gd_edge_list, edge_type=gd_edge_colour_list).to(device)
    model.eval()
    gnn_output_gd = model.all_labels(gd_data)

    # Target fact loading and processing (if file contains more than one, only the first one is explained)
    facts_path = args.facts
    assert os.path.exists(facts_path), "No fact found in file {}".format(facts_path)
    print("Loading facts to be explained from {}".format(facts_path))
    lines = open(facts_path, 'r').readlines()
    const_a, pred_R, const_b = lines[0].split()
    fact = (const_a, pred_R, const_b)
    cd_fact = iclr_encoder_decoder.get_canonical_equivalent(fact)
    (cd_const_ab, cd_fact_tp, cd_fact_predicate) = cd_fact
    cd_const_ba = iclr_encoder_decoder.term_for_tuple((const_b, const_a))
    cd_const_a = iclr_encoder_decoder.term_for_tuple((const_a,))
    cd_const_b = iclr_encoder_decoder.term_for_tuple((const_b,))
    node_ab = nodes.const_node_dict[cd_const_ab]
    node_a = nodes.const_node_dict[cd_const_a]
    node_b = nodes.const_node_dict[cd_const_b]
    node_ba = nodes.const_node_dict[cd_const_ba]
    row_ab = node_to_gd_row_dict[node_ab]
    row_a = node_to_gd_row_dict[node_a]
    row_b = node_to_gd_row_dict[node_b]
    row_ba = node_to_gd_row_dict[node_ba]
    cd_fact_pred_pos = can_encoder_decoder.unary_pred_position_dict[cd_fact_predicate]
    assert gnn_output_gd[2][row_ab][cd_fact_pred_pos] >= args.threshold, "Error: the fact to be explained is not derived by the model on this dataset."



    # Compute relevant positions and possible values, exploiting matrix sparsity
    c1 = can_encoder_decoder.binary_pred_colour_dict[iclr_encoder_decoder.binary_canonical[1]]
    c2 = can_encoder_decoder.binary_pred_colour_dict[iclr_encoder_decoder.binary_canonical[2]]
    c3 = can_encoder_decoder.binary_pred_colour_dict[iclr_encoder_decoder.binary_canonical[3]]
    # we consider only 8 relevant vertices:
    # output vertex v_xy
    # vertices v_x, v_Y, v_yx, connected to it via colours c1, c2, c3, respectively
    # vertices v_xz, v_zx; connected to x via colours c1 and c2, respectively
    # vertices v_yz, v_zy, connected to y via colours c1 and c2, respectively
    # the last four are not real vertices, but virtual vertices to represent the results of max aggregation for x and y

    #  Exploit matrix sparsity to identify the only relevant positions of each vector
    relevant_positions_xy_1 = relevant([cd_fact_pred_pos], model.matrix_A(2))
    relevant_positions_x_1 = relevant([cd_fact_pred_pos], model.matrix_B(2, c1))
    relevant_positions_y_1 = relevant([cd_fact_pred_pos], model.matrix_B(2, c2))
    relevant_positions_yx_1 = relevant([cd_fact_pred_pos], model.matrix_B(2, c3))
    relevant_positions_xy_0_for_xy_1 = {}
    relevant_positions_yx_0_for_xy_1 = {}
    relevant_positions_xz_0_for_x_1 = {}
    relevant_positions_zx_0_for_x_1 = {}
    relevant_positions_yz_0_for_y_1 = {}
    relevant_positions_zy_0_for_y_1 = {}
    relevant_positions_yx_0_for_yx_1 = {}
    relevant_positions_xy_0_for_yx_1 = {}
    for i in relevant_positions_xy_1:
        relevant_positions_xy_0_for_xy_1[i] = relevant([i], model.matrix_A(1))
        relevant_positions_yx_0_for_xy_1[i] = relevant([i], model.matrix_B(1, c3))
    for i in relevant_positions_x_1:
        relevant_positions_xz_0_for_x_1[i] = relevant([i], model.matrix_B(1, c1))
        relevant_positions_zx_0_for_x_1[i] = relevant([i], model.matrix_B(1, c2))
    for i in relevant_positions_y_1:
        relevant_positions_yz_0_for_y_1[i] = relevant([i], model.matrix_B(1, c1))
        relevant_positions_zy_0_for_y_1[i] = relevant([i], model.matrix_B(1, c2))
    for i in relevant_positions_yx_1:
        relevant_positions_yx_0_for_yx_1[i] = relevant([i], model.matrix_A(1))
        relevant_positions_xy_0_for_yx_1[i] = relevant([i], model.matrix_B(1, c3))

    #  Extract the rules
    rules = set([])
    new_variables = 0
    layer_1_to_2_rule_body = set([])
    print("Block with heads R(x,y)")
    for i in relevant_positions_xy_1:
        alpha = gnn_output_gd[1][row_ab][i].item()
        if alpha > 0:
            special_predicate = "U_{}_xy_{}".format(i, alpha)
            layer_1_to_2_rule_body.add(("X", special_predicate, "Y"))
            defining_rule_body_pos = set([])
            defining_rule_head = ("X", special_predicate, "Y")
            for j in relevant_positions_xy_0_for_xy_1[i]:
                if gnn_output_gd[0][row_ab][j].item() == 1:
                    defining_rule_body_pos.add(("X", iclr_encoder_decoder.get_data_predicate(
                        can_encoder_decoder.position_unary_pred_dict[j]), "Y"))
            for j in relevant_positions_yx_0_for_xy_1[i]:
                if gnn_output_gd[0][row_ba][j].item() == 1:
                    defining_rule_body_pos.add(("Y", iclr_encoder_decoder.get_data_predicate(
                        can_encoder_decoder.position_unary_pred_dict[j]), "X"))
            rules.add((defining_rule_head, frozenset(defining_rule_body_pos)))
    print("Block with heads U(x)")
    edge_mask_c1 = gd_edge_colour_list == c1
    colour_edges_c1 = gd_edge_list[:, edge_mask_c1]
    neighbours_c1 = set(colour_edges_c1[:, colour_edges_c1[1] == row_a][0].tolist())
    edge_mask_c2 = gd_edge_colour_list == c2
    colour_edges_c2 = gd_edge_list[:, edge_mask_c2]
    neighbours_c2 = set(colour_edges_c1[:, colour_edges_c2[1] == row_a][0].tolist())
    for i in relevant_positions_x_1:
        alpha = gnn_output_gd[1][row_a][i].item()
        if alpha > 0:
            special_predicate = "U_{}_x_{}".format(i, alpha)
            layer_1_to_2_rule_body.add(("X", type_pred, special_predicate))
            defining_rule_body_pos = set([])
            defining_rule_head = ("X", type_pred, special_predicate)
            for j in relevant_positions_xz_0_for_x_1[i]:
                new_variables += 1
                z = "Z{}".format(new_variables)
                contributing_neighbour_exists = False
                for neighbour in neighbours_c1:
                    element = gnn_output_gd[0][neighbour][j].item()
                    if element > 0:
                        contributing_neighbour_exists = True
                        break
                if contributing_neighbour_exists:
                    defining_rule_body_pos.add(("X", iclr_encoder_decoder.get_data_predicate(
                        can_encoder_decoder.position_unary_pred_dict[j]), z))
            for j in relevant_positions_zx_0_for_x_1[i]:
                new_variables += 1
                z = "Z{}".format(new_variables)
                contributing_neighbour_exists = False
                for neighbour in neighbours_c2:
                    element = gnn_output_gd[0][neighbour][j].item()
                    if element > 0:
                        contributing_neighbour_exists = True
                        break
                if contributing_neighbour_exists:
                    defining_rule_body_pos.add((z, iclr_encoder_decoder.get_data_predicate(
                        can_encoder_decoder.position_unary_pred_dict[j]), "X"))
            rules.add((defining_rule_head, frozenset(defining_rule_body_pos)),)
    print("Block with heads U(y)")
    edge_mask_c1 = gd_edge_colour_list == c1
    colour_edges_c1 = gd_edge_list[:, edge_mask_c1]
    neighbours_c1 = set(colour_edges_c1[:, colour_edges_c1[1] == row_b][0].tolist())
    edge_mask_c2 = gd_edge_colour_list == c2
    colour_edges_c2 = gd_edge_list[:, edge_mask_c2]
    neighbours_c2 = set(colour_edges_c1[:, colour_edges_c2[1] == row_b][0].tolist())
    for i in relevant_positions_y_1:
        alpha = gnn_output_gd[1][row_b][i].item()
        if alpha > 0:
            special_predicate = "U_{}_y_{}".format(i, alpha)
            layer_1_to_2_rule_body.add(("Y", type_pred, special_predicate))
            defining_rule_body_pos = set([])
            defining_rule_head = ("X", type_pred, special_predicate)
            for j in relevant_positions_yz_0_for_y_1[i]:
                new_variables += 1
                z = "Z{}".format(new_variables)
                contributing_neighbour_exists = False
                for neighbour in neighbours_c1:
                    element = gnn_output_gd[0][neighbour][j].item()
                    if element > 0:
                        contributing_neighbour_exists = True
                        break
                if contributing_neighbour_exists:
                    defining_rule_body_pos.add(("X", iclr_encoder_decoder.get_data_predicate(
                        can_encoder_decoder.position_unary_pred_dict[j]), z))
            for j in relevant_positions_zy_0_for_y_1[i]:
                new_variables += 1
                z = "Z{}".format(new_variables)
                contributing_neighbour_exists = False
                for neighbour in neighbours_c2:
                    element = gnn_output_gd[0][neighbour][j].item()
                    if element > 0:
                        contributing_neighbour_exists = True
                        break
                if contributing_neighbour_exists:
                    defining_rule_body_pos.add((z, iclr_encoder_decoder.get_data_predicate(
                        can_encoder_decoder.position_unary_pred_dict[j]), "X"))
            rules.add((defining_rule_head, frozenset(defining_rule_body_pos)),)
    print("Block with heads R(y,x)")
    for i in relevant_positions_yx_1:
        alpha = gnn_output_gd[1][row_ba][i].item()
        if alpha > 0:
            special_predicate = "U_{}_yx_{}".format(i, alpha)
            layer_1_to_2_rule_body.add(("Y", special_predicate, "X"))
            defining_rule_body_pos = set([])
            defining_rule_head = ("X", special_predicate, "Y")
            for j in relevant_positions_yx_0_for_yx_1[i]:
                if gnn_output_gd[0][row_ba][j].item() == 1:
                    defining_rule_body_pos.add(("X", iclr_encoder_decoder.get_data_predicate(
                        can_encoder_decoder.position_unary_pred_dict[j]), "Y"))
            for j in relevant_positions_xy_0_for_yx_1[i]:
                if gnn_output_gd[0][row_ab][j].item() == 1:
                    defining_rule_body_pos.add(("Y", iclr_encoder_decoder.get_data_predicate(
                        can_encoder_decoder.position_unary_pred_dict[j]), "X"))
            rules.add((defining_rule_head, frozenset(defining_rule_body_pos)),)
    rules.add((("X", iclr_encoder_decoder.get_data_predicate(cd_fact_predicate), "Y"), frozenset(layer_1_to_2_rule_body)),)

    print("Extraction completed. Total number of rules: {}".format(len(rules)+1))

    print(datetime.datetime.now())

    # Prepare output
    if args.output is not None:
        output_file = open(args.output, 'w')

    for (head_atom, pos_body_atoms) in rules:
        body_atoms = []
        for (s, p, o) in pos_body_atoms:
            if p == type_pred:
                body_atoms.append("<{}>[?{}]".format(o, s))
            else:
                body_atoms.append("<{}>[?{},?{}]".format(p, s, o))
        head = "<{}>[?X,?Y]".format(iclr_encoder_decoder.unary_canonical_to_input_predicate_dict[cd_fact_predicate])
        rule = head + " :- " + ", ".join(body_atoms) + " .\n"

        if args.output is not None:
            output_file.write("{}\n".format(fact))
            output_file.write(rule + '\n')


    # values = {xy: {}, x: {}, y: {}, yx: {}}
    #print("Computing for v_xy")
    # values_xy_1 = {}
    # for i in relevant_positions_xy_1:
    #     values[xy][i] = {}
    #     relevant_positions[0][xy][i] = relevant([i], model.matrix_A(1))
    #     relevant_positions[0][yx][i] = relevant([i], model.matrix_B(1, c3))
    #     options_x = powerset(relevant_positions[0][xy][i])
    #     options_y3 = powerset(relevant_positions[0][yx][i])
    #     for (rel_x, rel_y3) in itertools.product(*[options_x, options_y3]):
    #         suma = 0
    #         for j in rel_x:
    #             suma += model.matrix_A(1)[i, j]
    #         for j in rel_y3:
    #             suma += model.matrix_B(1, c3)[i, j]
    #         suma += model.bias(1)[i]
    #         outcome = model.activation(1)(suma)
    #         if outcome > 0:
    #             if outcome in values[xy][i]:
    #                 values[xy][i][outcome].add((rel_x, rel_y3))
    #             else:
    #                 values[xy][i][outcome] = {(rel_x, rel_y3)}
    #
    # print("Computing for y1")
    # for i in relevant_positions[1][x]:
    #     values[x][i] = {}
    #     relevant_positions[0][x][i] = relevant([i], model.matrix_A(1))
    #     relevant_positions[0][xz][i] = relevant([i], model.matrix_B(1, c1))
    #     relevant_positions[0][zx][i] = relevant([i], model.matrix_B(1, c2))
    #     options_y1 = powerset(relevant_positions[0][x])
    #     options_z11 = powerset(relevant_positions[0][xz])
    #     options_z12 = powerset(relevant_positions[0][zx])
    #     for (rel_y1, rel_z11, rel_z12) in itertools.product(*[options_y1, options_z11, options_z12]):
    #         suma = 0
    #         for j in rel_y1:
    #             suma += model.matrix_A(1)[i, j]
    #         for j in rel_z11:
    #             suma += model.matrix_B(1,c1)[i, j]
    #         for j in rel_z12:
    #             suma += model.matrix_B(1,c2)[i, j]
    #         suma += model.bias(1)[i]
    #         outcome = model.activation(1)(suma)
    #         if outcome > 0:
    #             if outcome in values[x][i]:
    #                 values[x][i][outcome].add((rel_y1, rel_z11, rel_z12))
    #             else:
    #                 values[x][i][outcome] = {(rel_y1, rel_z11, rel_z12)}
    #
    # print("Computing for y2")
    # for i in relevant_positions[1][y]:
    #     values[y][i] = set()
    #     relevant_positions[0][y] = relevant([i], model.matrix_A(1))
    #     relevant_positions[0][yz] = relevant([i], model.matrix_B(1, c1))
    #     relevant_positions[0][zy] = relevant([i], model.matrix_B(1, c2))
    #     options_y2 = powerset(relevant_positions[0][y])
    #     options_z21 = powerset(relevant_positions[0][yz])
    #     options_z22 = powerset(relevant_positions[0][zy])
    #     for (rel_y2, rel_z21, rel_z22) in itertools.product(*[options_y2, options_z21, options_z22]):
    #         suma = 0
    #         for j in rel_y2:
    #             suma += model.matrix_A(1)[i, j]
    #         for j in rel_z21:
    #             suma += model.matrix_B(1, c1)[i, j]
    #         for j in rel_z22:
    #             suma += model.matrix_B(1, c2)[i, j]
    #         suma += model.bias(1)[i]
    #         outcome = model.activation(1)(suma)
    #         if outcome > 0:
    #             if outcome in values[y][i]:
    #                 values[y][i][outcome].add((rel_y2, rel_z21, rel_z22))
    #             else:
    #                 values[y][i][outcome] = {(rel_y2, rel_z21, rel_z22)}
    #
    # print("Computing for y3")
    # for i in relevant_positions[1][yx]:
    #     values[yx][i] = set()
    #     relevant_positions[0][yx] = relevant([i], model.matrix_A(1))
    #     relevant_positions[0][xy] = relevant([i], model.matrix_B(1, c3))
    #     options_y3 = powerset(relevant_positions[0][yx])
    #     options_x = powerset(relevant_positions[0][xy])
    #     for (rel_y3, rel_x) in itertools.product(*[options_y3, options_x]):
    #         suma = 0
    #         for j in rel_y3:
    #             suma += model.matrix_A(1)[i, j]
    #         for j in rel_x:
    #             suma += model.matrix_B(1, c3)[i, j]
    #         suma += model.bias(1)[i]
    #         outcome = model.activation(1)(suma)
    #         if outcome > 0:
    #             if outcome in values[xy][i]:
    #                 values[yx][i][outcome].add((rel_y3, rel_x))
    #             else:
    #                 values[yx][i][outcome] = {(rel_y3, rel_x)}
    #
    # for i in relevant_positions[1][x]:
    #     alpha = gnn_output_gd[1][row_a][i]
    #     first_rule_body.append((x, type_pred, "U_{}_y1_{}".format(i, alpha)))
    # for i in relevant_positions[1][y]:
    #     alpha = gnn_output_gd[1][row_b][i]
    #     first_rule_body.append((y, type_pred, "U_{}_y2_{}".format(i, alpha)))
    # for i in relevant_positions[1][yx]:
    #     alpha = gnn_output_gd[1][row_ba][i]
    #     first_rule_body.append((yx, type_pred, "U_{}_y3_{}".format(i, alpha)))

    # Great optimisation: no negations needed in the first layer.

