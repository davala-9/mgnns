import torch
from torch_geometric.data import Data
import numpy as np
import argparse
import os.path
import data_parser
import nodes
import datetime

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
parser.add_argument('--minimal-rule',
                    default=False,
                    action='store_true')

args = parser.parse_args()

type_pred = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'

# Optimised rule extraction algorithm: the hope is that the extracted rules are the shortest/strongest among the
# explanatory rules, and so they are the ones that make most sense.
# IMPORTANT NOTE: Currently supports only max aggregation.
if __name__ == "__main__":

    print(datetime.datetime.now())

    dataset_path = args.dataset
    assert os.path.exists(dataset_path)
    print("Loading graph data from {}".format(dataset_path))
    dataset = data_parser.parse(dataset_path)

    model_name = args.load_model_name[:-3] # Remove extension '.pt'

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
        # Explain the first 10 facts
        num_read_lines += 1
        if num_read_lines > 10:
            continue
        if num_read_lines%10 == 0:
            print(num_read_lines)

        ent1, ent2, ent3 = line.split()
        fact = (ent1, ent2, ent3)
        if args.encoding_scheme == 'canonical':
            cd_fact = fact
            assert cd_fact_tp == type_pred, "Error, with the canonical encoding, the fact to be explained should be unary."
        else:
            cd_fact = iclr_encoder_decoder.get_canonical_equivalent(fact)
        (cd_fact_constant, cd_fact_tp, cd_fact_predicate) = cd_fact
        cd_fact_node = nodes.const_node_dict[cd_fact_constant]
        assert cd_fact_node in node_to_gd_row_dict, "Error: the encoded fact mentions a constant not in the encoded dataset."
        L = model.num_layers
        cd_fact_gd_row = node_to_gd_row_dict[cd_fact_node]
        cd_fact_pred_pos = can_encoder_decoder.unary_pred_position_dict[cd_fact_predicate]

        # Sanity check: ensure the fact is a consequence of the model and the dataset
        assert gnn_output_gd[L][cd_fact_gd_row][cd_fact_pred_pos] >= args.threshold, "Error: the fact to be explained is not derived by the model on this dataset."
        rule_body = []

        # Compute simplified version of Gamma_i
        print("Computing Gamma_i...")
        variable_counter = 1
        x1 = "X1"
        # The following two dictionaries capture a substitution \nu of variables in the rule to terms that unify with them (expressed as nodes)
        nu_variable_to_node_dict = {x1: cd_fact_node}
        nu_node_to_variable_dict = {cd_fact_node: x1}
        # We construct a tree where nodes are variables of Gamma_i. Each node at depth ell has a label for each layer from \ell to 0.
        # The label is the paper's \mu: a list of the elements of the corresponding feature vector that are relevant.
        # Each node has a successor by colour c_i, position j to the variable representing the constant that contributes
        # with the maximum value aggregating over c_i in position j.
        successors = {}  # (var, layer, colour, pos) -> var
        predecessors = {}  # (var, layer) -> (var, colour, pos)
        labels = {(x1, L): {cd_fact_pred_pos}}
        variables_for_next_layer = [x1]
        for current_layer in range(L, 0, -1):
            variables_for_this_layer = variables_for_next_layer.copy()
            for y in variables_for_this_layer:
                current_label = labels[(y, current_layer)]
                # Update label for same variable
                new_label = set()
                for j in range(model.layer_dimension(current_layer - 1)):
                    relevant = False
                    for i in current_label:
                        if not relevant and model.matrix_A(current_layer)[i][j].item() * gnn_output_gd[current_layer - 1][node_to_gd_row_dict[nu_variable_to_node_dict[y]]][j].item() > 0:
                            new_label.add(j)
                            relevant = True
                labels[(y, current_layer-1)] = new_label
                # Introduce new variables
                for colour in can_encoder_decoder.colours:
                    edge_mask = gd_edge_colour_list == colour
                    colour_edges = gd_edge_list[:, edge_mask]
                    neighbours = colour_edges[:, colour_edges[1] == node_to_gd_row_dict[nu_variable_to_node_dict[y]]][0].tolist()
                    for j in range(model.layer_dimension(current_layer-1)):
                        # Check if the value in position j of the aggregation is used in the matrix multiplications.
                        relevant = False
                        for i in current_label:
                            if not relevant and model.matrix_B(current_layer, colour)[i][j].item() > 0:
                                relevant = True
                        if relevant:
                            # Find the neighbour that contributes maximum to aggregation
                            max_neighbour = None
                            max_value = 0
                            for neighbour in neighbours:
                                element = gnn_output_gd[current_layer - 1][neighbour][j].item()
                                # TODO ITP: can break ties using number of neighbours
                                if element > max_value:
                                    max_neighbour = neighbour
                                    max_value = element
                            if max_neighbour is not None:
                                variable_counter += 1
                                z = "X" + str(variable_counter)
                                nu_variable_to_node_dict[z] = gd_row_to_node_dict[max_neighbour]
                                nu_node_to_variable_dict[gd_row_to_node_dict[max_neighbour]] = z
                                successors[(y, current_layer, colour, j)] = z
                                predecessors[(z, current_layer)] = (y, colour, j)
                                variables_for_next_layer.append(z)
                                new_label = {j}
                                labels[(z, current_layer-1)] = new_label

        def get_conjunction_for_contributor(var_y, pos_j):
            conj = set()
            conj.add((var_y, type_pred, can_encoder_decoder.position_unary_pred_dict[pos_j]))
            vari = var_y
            for cur_lay in range(1, L + 1, 1):
                if (vari, cur_lay) in predecessors:
                    (new_vari, colr, _) = predecessors[(vari, cur_lay)]
                    conj.add((vari, can_encoder_decoder.colour_binary_pred_dict[colr], new_vari))
                    vari = new_vari
            return conj

        def test_gr_dataset(body):
            if not body:
                gr_features = torch.FloatTensor(np.zeros((1, model.layer_dimension(0))))
                gr_edge_list = torch.LongTensor(2, 0)
                gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=torch.LongTensor([])).to(device)
                gnn_output_gr = model(gr_dataset)
                return gnn_output_gr[0][cd_fact_pred_pos] >= args.threshold
            else:
                # Note that here variables are treated as constants. Thus, to recover the relevant node in
                # the gr graph, we need to call nodes.const_node_dict[z], not nu_variable_to_node[z].
                (gr_features, node_to_gr_row_dict, gr_edge_list, gr_colour_list) = can_encoder_decoder.encode_dataset(body)
                gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(device)
                gnn_output_gr = model(gr_dataset)
                a = gnn_output_gr[node_to_gr_row_dict[nodes.const_node_dict[x1]]][cd_fact_pred_pos]
                b = args.threshold
                return a >= b

        # Sanity check: ensure Gamma_i suffices to derive the fact
        mini_dataset = []
        for y in variables_for_next_layer:
            for j in labels[(y, 0)]:
                mini_dataset.append((y, type_pred, can_encoder_decoder.position_unary_pred_dict[j]))
            variables = [y]
            for current_layer in range(1, L+1, 1):
                if (variables[-1], current_layer) in predecessors:
                    (z, colour, _) = predecessors[(variables[-1], current_layer)]
                    variables.append(z)
                    mini_dataset.append((y, can_encoder_decoder.colour_binary_pred_dict[colour], z))
                else:
                    variables.append(variables[-1])
        (gr_features, node_to_gr_row_dict, gr_edge_list, gr_colour_list) = can_encoder_decoder.encode_dataset(mini_dataset)
        gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(device)
        gnn_output_gr = model(gr_dataset)
        assert gnn_output_gr[node_to_gr_row_dict[nodes.const_node_dict[x1]]][cd_fact_pred_pos] >= args.threshold,\
            "Bug: the base rule does not derive the fact from the dataset"


        # BEST APPROXIMATION 1
        # Make a dictionary of all elements of gamma_i and compute the influence value of each
        # To compute the value, construct the minimal rule, and check the value of the node. If zero,
        # then start travelling backwards through the relevant list of colours until you find one that
        # has relevant nodes not equal to zero. Mark the layer and the sum of values (pondered by weights of the next
        # matrix multiplication-- this last value is probably not enough to beat the bias and activation function).
        print("Attempting approximation 1...")
        contributors_to_influence_dict = {}
        for y in variables_for_next_layer:
            for j in labels[(y, 0)]:
                mini_dataset = [(y, type_pred, can_encoder_decoder.position_unary_pred_dict[j])]
                # Construct test dataset
                variables = [y]
                for current_layer in range(1, L+1, 1):
                    if (variables[-1], current_layer) in predecessors:
                        (z, colour, _) = predecessors[(variables[-1], current_layer)]
                        variables.append(z)
                        mini_dataset.append((y, can_encoder_decoder.colour_binary_pred_dict[colour], z))
                    else:
                        variables.append(variables[-1])
                (gr_features, node_to_gr_row_dict, gr_edge_list, gr_colour_list) = can_encoder_decoder.encode_dataset(mini_dataset)
                gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(device)
                gnn_output_gr = model(gr_dataset)
                current_layer = L
                while (y, j) not in contributors_to_influence_dict:
                    if current_layer == 0:
                        contributors_to_influence_dict[(y, j)] = (0,0)
                    else:
                        z = variables.pop()
                        node_z = nodes.const_node_dict[z]
                        total = 0
                        for k in labels[(z, current_layer)]:
                            total = total + gnn_output_gr[node_to_gr_row_dict[node_z]][k]
                        if total > 0:
                            contributors_to_influence_dict[(y, j)] = (current_layer, total)
                        else:
                            current_layer = current_layer - 1
        # Start trying rules, adding atoms in increasing order of contribution
        contributors_list = []
        for (y, j) in contributors_to_influence_dict:
            contributors_list.append((contributors_to_influence_dict[(y, j)], y, j))
        contributors_list = sorted(contributors_list)
        current_body_set = set()
        output_body = None
        while not output_body:
            if test_gr_dataset(current_body_set):
                output_body = current_body_set
            else:
                (inf, vy, pj) = contributors_list.pop()
                current_body_set = current_body_set.union(get_conjunction_for_contributor(vy, pj))
        short_body_1 = list(output_body)
        # TODO ITP: maybe a clean-up step like in approximation 2 can help make improvements

        print("Attempting approximation 2...")
        # BEST APPROXIMATION 2
        # This uses old code, and can only be done for 2 layers and relu. Essentially it multiplies the products
        # of the matrix weights along an `influence path': for example, if we have 2 layers, position 1 in layer 0 affects
        # positions 2 and 3 in layer 1, which in turn affect position 4 in layer 2, (1->2->4) and (1->3->4) are two
        # differenc influence paths. We sort influence paths by value and add atoms in this order.
        if model.num_layers == 2 and model.activation(1) == torch.relu:

            # An input unit for a given node, i, and layer is a triple (node',col,j) where node' is connected to node
            # via a link col (or, if col=-1, node'=node), and it holds that both the j-th feature of node' in layer-1 is
            # positive, max among those features for the col and j, and the (i,j)-weight of the matrix for colour col
            # (matrix A if col=-1) is also positive. Intuitively, it captures all inputs that affect the value of the
            # ith feature of node in layer A, on this dataset.
            def get_input_units(node_as_row, ii, layer):
                return_list = []
                for clr in set(can_encoder_decoder.colours).union({-1}):
                    if clr == -1:
                        matrix = model.matrix_A(layer)
                        nghbrs = {node_as_row}
                    else:
                        matrix = model.matrix_B(layer=layer, colour=clr)
                        mask = gd_edge_colour_list == clr
                        gd_clr_edge_list = gd_edge_list[:, mask]
                        nghbrs = set(gd_clr_edge_list[:, gd_clr_edge_list[1] == node_as_row][0].tolist())
                    for jj in range(model.layer_dimension(layer - 1)):
                        if matrix[ii][jj].item() > 0:
                            mx_neighbour = None
                            mx_value = 0
                            for nghbr in nghbrs:
                                feature = gnn_output_gd[layer - 1][nghbr][jj].item()
                                if feature > mx_value:
                                    mx_neighbour = nghbr
                                    mx_value = feature
                            if mx_neighbour is not None:
                                return_list.append((mx_neighbour, clr, jj))
                return return_list

            r_body_dataset = []
            if not test_gr_dataset(r_body_dataset):
                contributions = []
                for (source2_row, col2, j2) in get_input_units(cd_fact_gd_row, cd_fact_pred_pos, 2):
                    source2_node = gd_row_to_node_dict[source2_row]
                    if col2 == -1:
                        z2 = "X1"
                    else:
                        variable_counter += 1
                        z2 = "X" + str(variable_counter)
                        nu_node_to_variable_dict[source2_node] = z2
                        nu_variable_to_node_dict[z2] = source2_node
                    next_level = get_input_units(source2_row, j2, 1)
                    if col2 == -1:
                        matrix2 = model.matrix_A(layer=2)
                    else:
                        matrix2 = model.matrix_B(layer=2, colour=col2)
                    if not next_level:
                        c_value = matrix2[cd_fact_pred_pos][j2] * gnn_output_gd[1][cd_fact_gd_row][j2]
                        contributions.append((c_value, z2, None, col2, None, j2, None))
                    for (source1_row, col1, j1) in get_input_units(source2_row, j2, 1):
                        source1_node = gd_row_to_node_dict[source1_row]
                        if col1 == -1:
                            z1 = z2
                        else:
                            variable_counter += 1
                            z1 = "X" + str(variable_counter)
                            nu_node_to_variable_dict[source1_node] = z1
                            nu_variable_to_node_dict[z1] = source1_node
                        if col1 == -1:
                            matrix1 = model.matrix_A(layer=1)
                        else:
                            matrix1 = model.matrix_B(layer=1, colour=col1)
                        contribution_value = matrix2[cd_fact_pred_pos][j2] * matrix1[j2][j1]
                        contributions.append((contribution_value, z2, z1, col2, col1, j2, j1))
                contributions = sorted(contributions, reverse=True)
                threshold_met = False
                used_contributions = []
                contributions_to_atoms_necessary = {}
                while not threshold_met and contributions:
                    contrib = contributions.pop(0)
                    contributions_to_atoms_necessary[contrib] = []
                    used_contributions.append(contrib)
                    contribution_value, z2, z1, col2, col1, j2, j1 = contrib
                    if col2 == -1:
                        if col1 is None:
                            pass
                        elif col1 == -1:
                            atom = ("X1", type_pred, can_encoder_decoder.position_unary_pred_dict[j1])
                            r_body_dataset.append(atom)
                            contributions_to_atoms_necessary[contrib].append(atom)
                            threshold_met = test_gr_dataset(r_body_dataset)
                        else:
                            binary_atom_1 = (z1, can_encoder_decoder.colour_binary_pred_dict[col1], "X1")
                            contributions_to_atoms_necessary[contrib].append(binary_atom_1)
                            if binary_atom_1 not in r_body_dataset:
                                r_body_dataset.append(binary_atom_1)
                                threshold_met = test_gr_dataset(r_body_dataset)
                            if not threshold_met:
                                atom = (z1, type_pred, can_encoder_decoder.position_unary_pred_dict[j1])
                                contributions_to_atoms_necessary[contrib].append(atom)
                                r_body_dataset.append(atom)
                                threshold_met = test_gr_dataset(r_body_dataset)
                    else:
                        binary_atom_2 = (z2, can_encoder_decoder.colour_binary_pred_dict[col2], "X1")
                        contributions_to_atoms_necessary[contrib].append(binary_atom_2)
                        if binary_atom_2 not in r_body_dataset:
                            r_body_dataset.append(binary_atom_2)
                            threshold_met = test_gr_dataset(r_body_dataset)
                        if not threshold_met and col1 is not None:
                            if col1 == -1:
                                atom = (z2, type_pred, can_encoder_decoder.position_unary_pred_dict[j1])
                                contributions_to_atoms_necessary[contrib].append(atom)
                                r_body_dataset.append(atom)
                                threshold_met = test_gr_dataset(r_body_dataset)
                            else:
                                binary_atom_1 = (z1, can_encoder_decoder.colour_binary_pred_dict[col1], z2)
                                contributions_to_atoms_necessary[contrib].append(binary_atom_1)
                                if binary_atom_1 not in r_body_dataset:
                                    r_body_dataset.append(binary_atom_1)
                                    threshold_met = test_gr_dataset(r_body_dataset)
                                if not threshold_met:
                                    atom = (z1, type_pred, can_encoder_decoder.position_unary_pred_dict[j1])
                                    contributions_to_atoms_necessary[contrib].append(atom)
                                    r_body_dataset.append(atom)
                                    threshold_met = test_gr_dataset(r_body_dataset)
                (gr_features, node_to_gr_row_dict, gr_edge_list, gr_colour_list) = can_encoder_decoder.encode_dataset(
                    r_body_dataset)
                gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(device)
                gnn_output_gr = model.all_labels(gr_dataset)
                necessary_body_atoms = set()
                while used_contributions:
                    contrib = used_contributions.pop()
                    (contribution_value, z2, z1, col2, col1, j2, j1) = contrib
                    # if gnn_output_gr[1][node_to_gr_row_dict[nodes.const_node_dict[z2]]][j2] != 0:
                    for atom in contributions_to_atoms_necessary[contrib]:
                        necessary_body_atoms.add(atom)
                short_body_2 = list(necessary_body_atoms)

        # SELECT FINAL RULE
        rule_body = []
        if not args.minimal_rule:
            if short_body_2 and len(short_body_2) < len(short_body_1):
                print("Approximation 2 wins")
                rule_body = short_body_2
            else:
                print("Approximation 1 wins")
                rule_body = short_body_1
        else:
            if short_body_2 and len(short_body_2) < len(short_body_1):
                print("Approximation 2 wins")
                max_number_of_body_atoms = len(short_body_2)
            else:
                print("Approximation 1 wins")
                max_number_of_body_atoms = len(short_body_1)

            # Dictionary mapping each variables in gamma_i to its relevant positions (given by label of layer 0).
            total_variable_to_positions_dict = {}
            for (y, j) in contributors_to_influence_dict:
                if y not in total_variable_to_positions_dict:
                    total_variable_to_positions_dict[y] = {j}
                else:
                    total_variable_to_positions_dict[y].add(j)

            def get_successors(partial_variable_to_positions_dict, conjunction_form):
                successors = []
                for y in total_variable_to_positions_dict:
                    if y not in partial_variable_to_positions_dict:
                        for j in total_variable_to_positions_dict[y]:
                            new_successor = partial_variable_to_positions_dict.copy()
                            new_successor[y] = {j}
                            new_conjunction_form = conjunction_form.union(get_conjunction_for_contributor(y,j))
                            successors.append((len(new_conjunction_form),new_successor, new_conjunction_form))
                    else:
                        for j in total_variable_to_positions_dict[y]:
                            if j not in partial_variable_to_positions_dict[y]:
                                new_successor = partial_variable_to_positions_dict.copy()
                                new_successor[y].add(j)
                                new_conjunction_form = conjunction_form.union(get_conjunction_for_contributor(y,j))
                                successors.append((len(new_conjunction_form),new_successor, new_conjunction_form))
                return successors

            frontier = get_successors({},set())
            while frontier:
                # Sort in decreasing order of cost
                frontier = sorted(frontier, reverse=True)
                (score, dictionary, conjunction) = frontier.pop()
                (gr_features, node_to_gr_row_dict, gr_edge_list, gr_colour_list) = can_encoder_decoder.encode_dataset(conjunction)
                gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(device)
                gnn_output_gr = model(gr_dataset)
                if gnn_output_gr[node_to_gr_row_dict[nodes.const_node_dict[x1]]][cd_fact_pred_pos] >= args.threshold:
                    frontier = None
                    rule_body = conjunction
                else:
                    frontier = list(set(frontier).union(set(get_successors(dictionary,conjunction))))

        rule_body = remove_redundant_atoms(rule_body)

        # Correctness check: check that each atom is grounded in the canonical dataset via \nu
        for (s, p, o) in rule_body:
            if p == type_pred:
                constant = nodes.node_const_dict[nu_variable_to_node_dict[s]]
                assert (constant, p, o) in cd_dataset, "ERROR: This rule does not unify with the dataset. Bug."
            else:
                origin_constant = nodes.node_const_dict[nu_variable_to_node_dict[s]]
                dest_constant = nodes.node_const_dict[nu_variable_to_node_dict[o]]
                assert (origin_constant, p, dest_constant) in cd_dataset,\
                    "ERROR: the extracted rule does not unify with the dataset. Bug."
        # Correctness check: ensure that the rule is captured.
        if not rule_body:
            gr_features = torch.FloatTensor(np.zeros((1, model.layer_dimension(0))))
            gr_edge_list = torch.LongTensor(2, 0)
            gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=torch.LongTensor([])).to(device)
            gnn_output_gr = model(gr_dataset)
            assert gnn_output_gr[0][cd_fact_pred_pos] >= args.threshold,\
                "ERROR: the extracted rule seems not to be captured by the model. This means there is a bug."
        else:
            (gr_features, node_to_gr_row_dict, gr_edge_list, gr_colour_list) = can_encoder_decoder.encode_dataset(rule_body)
            gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(device)
            gnn_output_gr = model(gr_dataset)
            assert gnn_output_gr[node_to_gr_row_dict[nodes.const_node_dict[x1]]][cd_fact_pred_pos] > args.threshold, \
                "ERROR: the extracted rule seems not to be captured by the model. This means there is a bug."

        # Unfold extracted rules with the encoder's rules
        if args.encoding_scheme == "iclr22":
            rule_body, can_variable_to_data_variable, top_facts = iclr_encoder_decoder.unfold(rule_body, cd_fact_predicate)
            # Process top_facts
            for pair in top_facts:
                [y1, y2] = list(pair)
                cvar_list = iclr_encoder_decoder.find_canonical_variable(can_variable_to_data_variable, y1, y2)
                if len(cvar_list) == 1:
                    # y1 and y2 come from a binary canonical variable
                    ab = nodes.node_const_dict[nu_variable_to_node_dict[cvar_list[0]]]
                    a, b = iclr_encoder_decoder.term_tuple_dict[ab]
                    ba = iclr_encoder_decoder.tuple_term_dict[(b, a)]
                else:
                    # y1 and y2 come from unary canonical variables
                    a = nu_variable_to_node_dict[cvar_list[0]]
                    b = nu_variable_to_node_dict[cvar_list[1]]
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
