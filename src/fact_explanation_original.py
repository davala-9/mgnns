import torch
from torch_geometric.data import Data
import numpy as np
import argparse
import os.path
import data_parser
import nodes
from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder


from utils import load_predicates
parser = argparse.ArgumentParser(description="Extract a rule captured by the GNN which derives a given fact on this dataset")

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

args = parser.parse_args()

type_pred = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'

# This is the unoptimised algorithm from Section 4 of the paper.
if __name__ == "__main__":

    dataset_path = args.dataset
    assert os.path.exists(dataset_path)
    print("Loading graph data from {}".format(dataset_path))
    dataset = data_parser.parse(dataset_path)

    model_name = args.load_model_name[:-3]  # Remove extension '.pt'

    if args.encoding_scheme == 'canonical':
        cd_dataset = dataset
    else:
        iclr_encoder_decoder = ICLREncoderDecoder(load_from_document=args.iclr22_encoder_file)
        cd_dataset = iclr_encoder_decoder.encode_dataset(dataset)

    can_encoder_decoder = CanonicalEncoderDecoder(load_from_document=args.canonical_encoder_file)

    # gd stands for "graph dataset"
    (gd_features, node_to_gd_row_dict, gd_edge_list, gd_edge_colour_list) = can_encoder_decoder.encode_dataset(cd_dataset)
    gd_row_to_node_dict = {node_to_gd_row_dict[node]: node for node in node_to_gd_row_dict}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gd_data = Data(x=gd_features, edge_index=gd_edge_list, edge_type=gd_edge_colour_list).to(device)
    model = torch.load(args.load_model_name).to(device)
    model.eval()
    gnn_output_gd = model.all_labels(gd_data)

    if args.output is not None:
        output_file = open(args.output, 'w')

    facts_path = args.facts
    assert os.path.exists(facts_path), "No fact found in file {}".format(facts_path)
    print("Loading dataset from {}".format(facts_path))
    lines = open(facts_path, 'r').readlines()
    num_read_lines = 0
    for line in lines:
        num_read_lines += 1
        if num_read_lines > 10:
            continue
        if num_read_lines%10 == 0:
            print(num_read_lines)

        ent1, ent2, ent3 = line.split()
        fact = (ent1, ent2, ent3)
        if args.encoding_scheme == 'canonical':
            cd_fact = fact
        else:
            cd_fact = iclr_encoder_decoder.get_canonical_equivalent(fact)
        (cd_fact_constant, cd_fact_tp, cd_fact_predicate) = cd_fact
        assert cd_fact_tp == type_pred, "Error, the canonised fact to be explained should be unary."
        cd_fact_node = nodes.const_node_dict[cd_fact_constant]
        assert cd_fact_node in node_to_gd_row_dict, "Error: the canonised fact mentions a constant not in the (canonised) dataset."
        L = model.num_layers
        cd_fact_gd_row = node_to_gd_row_dict[cd_fact_node]
        cd_fact_pred_pos = can_encoder_decoder.unary_pred_position_dict[cd_fact_predicate]
        assert gnn_output_gd[L][cd_fact_gd_row][cd_fact_pred_pos] > args.threshold, "Error: the fact to be explained is not derived by the model on this dataset."

        rule_body = []
        variable_counter = 1
        x1 = "X1"
        # theta of the paper: substitution that grounds the extracted rule (constants expressed as nodes)
        theta_variable_to_node_dict = {x1: cd_fact_node}
        theta_node_to_variable_dict = {cd_fact_node: x1}
        variables_for_next_layer = [x1]
        for current_layer in range(L, 0, -1):
            variables_for_this_layer = variables_for_next_layer.copy()
            for y in variables_for_this_layer:
                for c in can_encoder_decoder.colours:
                    for j in range(model.layer_dimension(current_layer - 1)):
                        y_node_as_row = node_to_gd_row_dict[theta_variable_to_node_dict[y]]
                        gd_c_edge_list = gd_edge_list[:, gd_edge_colour_list == c]
                        neighbours_as_rows = gd_c_edge_list[:, gd_c_edge_list[1] == y_node_as_row][0].tolist()
                        value_neighbour_pairs = []
                        for neighbour in neighbours_as_rows:
                            value_neighbour_pairs.append((gnn_output_gd[current_layer-1][neighbour][j].item(), neighbour))
                        sorted_value_neighbour_pairs = sorted(value_neighbour_pairs, reverse=True)  # Max is the first
                        if sorted_value_neighbour_pairs and sorted_value_neighbour_pairs[0][0] > 0:
                            if model.aggregation_function(current_layer) == 'max':
                                max_neighbour = sorted_value_neighbour_pairs[0][1]
                                variable_counter += 1
                                z = "X" + str(variable_counter)
                                theta_variable_to_node_dict[z] = gd_row_to_node_dict[max_neighbour]
                                theta_node_to_variable_dict[gd_row_to_node_dict[max_neighbour]] = z
                                rule_body.append((y, can_encoder_decoder.colour_binary_pred_dict[c], z))
                                variables_for_next_layer.append(z)
                            else:
                                assert(model.aggregation_function(current_layer)) == 'sum'
                                new_variable_list = []
                                for (_, neighbour) in sorted_value_neighbour_pairs:
                                    variable_counter += 1
                                    z = "X" + str(variable_counter)
                                    theta_variable_to_node_dict[z] = gd_row_to_node_dict[neighbour]
                                    theta_node_to_variable_dict[gd_row_to_node_dict[neighbour]] = z
                                    rule_body.append((y, can_encoder_decoder.colour_binary_pred_dict[c], z))
                                    variables_for_next_layer.append(z)
                                    new_variable_list.append(z)
                                for ind1 in range(len(new_variable_list)-1):
                                    for ind2 in range(len(new_variable_list[ind1+1:])):
                                        rule_body.append((new_variable_list[ind1], "owl:differentFrom", new_variable_list[ind2]))
        for y in variables_for_next_layer:
            for j in range(model.layer_dimension[0]):
                if gnn_output_gd[0][node_to_gd_row_dict[theta_variable_to_node_dict[y]][j]].item() == 1:
                    rule_body.append((y, type_pred, can_encoder_decoder.position_unary_pred_dict[j]))

        # Unfold extracted rules with the encoder's rules
        if args.encoding_scheme == "iclr22":
            rule_body, can_variable_to_data_variable, top_facts = iclr_encoder_decoder.unfold(rule_body, cd_fact_predicate)
            # Process top_facts
            for pair in top_facts:
                [y1, y2] = list(pair)
                cvar_list = iclr_encoder_decoder.find_canonical_variable(can_variable_to_data_variable, y1, y2)
                if len(cvar_list) == 1:
                    # y1 and y2 come from a binary canonical variable
                    ab = nodes.node_const_dict[theta_variable_to_node_dict[cvar_list[0]]]
                    a, b = iclr_encoder_decoder.term_tuple_dict[ab]
                    ba = iclr_encoder_decoder.tuple_term_dict[(b, a)]
                else:
                    # y1 and y2 come from unary canonical variables
                    a = theta_variable_to_node_dict[cvar_list[0]]
                    b = theta_variable_to_node_dict[cvar_list[1]]
                    ab = iclr_encoder_decoder.tuple_term_dict[(a, b)]
                    ba = iclr_encoder_decoder.tuple_term_dict[(b, a)]
                fact_found = False
                for (a1, a2, a3) in cd_dataset:
                    if not fact_found and a2 == type_pred:
                        if a1 == ab:
                            fact_found = True
                            rule_body.append((y1, iclr_encoder_decoder.unary_canonical_to_input_predicate_dict[a3], y2))
                        elif a1 == ba:
                            fact_found = True
                            rule_body.append((y2, iclr_encoder_decoder.unary_canonical_to_input_predicate_dict[a3], y1))
                assert fact_found

        # Write the rule
        body_atoms = []
        for (s, p, o) in rule_body:
            if p == type_pred:
                body_atoms.append("<{}>[?{}]".format(o, s))
            else:
                body_atoms.append("<{}>[?{},?{}]".format(p, s, o))
        if args.encoding_scheme == 'canonical':
            head = "<{}>[?X1]".format(cd_fact_predicate)
        else:
            if iclr_encoder_decoder.data_predicate_to_arity[iclr_encoder_decoder.unary_canonical_to_input_predicate_dict[cd_fact_predicate]] == 1:
                head = "<{}>[?X1]".format(iclr_encoder_decoder.unary_canonical_to_input_predicate_dict[cd_fact_predicate])
            else:
                head = "<{}>[?X1,?X2]".format(iclr_encoder_decoder.unary_canonical_to_input_predicate_dict[cd_fact_predicate])
        rule = head + " :- " + ", ".join(body_atoms) + " .\n"

        if args.output is not None:
            output_file.write("{}\n".format(fact))
            output_file.write(rule + '\n')

    output_file.close()
