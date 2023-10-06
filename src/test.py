#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ----
"""
import torch
from torch_geometric.data import Data
from numpy import arange
from numpy import trapz
from numpy import nan_to_num
import argparse
import data_parser
import os.path
import rdflib as rdf
from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder
from utils import load_predicates

parser = argparse.ArgumentParser(description="Evaluate a trained GNNs")
parser.add_argument('--load-model-name',
                    help='Filename of trained model to load')
parser.add_argument('--threshold',
                    type=float,
                    default=0,
                    help='threshold of the GNN. The default value is 0 (all facts with positive scores are derived)')
parser.add_argument('--predicates',
                    help='File with the fixed, ordered list of predicates we consider.')
parser.add_argument('--test-graph',
                    help='Filename of graph test data')
parser.add_argument('--test-positive-examples',
                    help='Filename of positive examples.')
parser.add_argument('--test-negative-examples',
                    help='Filename of negative examples.')
parser.add_argument('--output',
                    default=None,
                    help='Print the classification metrics.')
parser.add_argument('--encoding-scheme',
                    default='canonical',
                    nargs='?',
                    choices=['iclr22', 'canonical'],
                    help='Choose the encoder-decoder that will be applied to the data (canonical by default).')
parser.add_argument('--canonical-encoder-file',
                    help='File with the canonical encoder/decoder used to train the model.')
parser.add_argument('--iclr22-encoder-file',
                    default=None,
                    help='File with the iclr22 encoder/decoder used to train the model, if it was used.')
parser.add_argument('--print-entailed-facts',
                    default=None,
                    help='Print the facts that have been derived in the provided filename.')
args = parser.parse_args()


def precision(tp, fp, tn, fn):
    value = 0
    try:
        value = tp / (tp + fp)
    except:
        value = float("NaN")
    finally:
        return value


def recall(tp, fp, tn, fn):
    value = 0
    try:
        value = tp / (tp + fn)
    except:
        value = float("NaN")
    finally:
        return value


def accuracy(tp, fp, tn, fn):
    value = 0
    try:
        value = (tn + tp) / (tp + fp + tn + fn)
    except:
        value = float("NaN")
    finally:
        return value


def f1score(tp, fp, tn, fn):
    value = 0
    try:
        value = tp / (tp + 0.5 * (fp + fn))
    except:
        value = float("NaN")
    finally:
        return value


def auprc(precision_vector, recall_vector):
    return -1 * trapz(precision_vector, recall_vector)


def parse_triple(line):
    temp_string = line[1:]
    bits = temp_string.split('>')
    ent1 = bits[0]
    print(ent1)
    ent2 = bits[1][2:]
    ent3 = bits[2][2:]
    ent4 = bits[3][1:-2]
    return ent1, ent2, ent3, ent4


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_graph_path = args.test_graph
    assert os.path.exists(test_graph_path)
    print("Loading graph data from {}".format(test_graph_path))
    test_graph_dataset = data_parser.parse(test_graph_path)

    if args.encoding_scheme == 'canonical':
        cd_dataset = test_graph_dataset
    else:
        iclr_encoder_decoder = ICLREncoderDecoder(load_from_document=args.iclr22_encoder_file)
        cd_dataset = iclr_encoder_decoder.encode_dataset(test_graph_dataset)

    can_encoder_decoder = CanonicalEncoderDecoder(load_from_document=args.canonical_encoder_file)

    (test_x, test_nodes, test_edge_list, test_edge_colour_list) = can_encoder_decoder.encode_dataset(cd_dataset)

    test_data = Data(x=test_x, edge_index=test_edge_list, edge_type=test_edge_colour_list).to(device)

    print("Evaluating model {} on dataset {} using threshold={}".format(args.load_model_name,
                                                                        args.test_graph, args.threshold))
    model = torch.load(args.load_model_name).to(device)
    model.eval()

    # gnn_output : torch.FloatTensor of size i x j, with i = num graph nodes, j = length of feature vectors
    # importantly, the ith row of gnn_output and test_x represent the same node
    gnn_output = model(test_data)

    cd_output_dataset_scores_dict = can_encoder_decoder.decode_graph(test_nodes, gnn_output, args.threshold)
    # facts_scores_dict:  a dictionary mapping triples (s,p,o) to a value (in str) score
    if args.encoding_scheme == 'canonical':
        facts_scores_dict = cd_output_dataset_scores_dict
    elif args.encoding_scheme == 'iclr22':
        facts_scores_dict = {}
        for (s, p, o) in cd_output_dataset_scores_dict:
            ss, pp, oo = iclr_encoder_decoder.decode_fact(s, p, o)
            facts_scores_dict[(ss, pp, oo)] = cd_output_dataset_scores_dict[(s, p, o)]

    # Print from the fact with the highest score to that with the least
    if args.print_entailed_facts is not None:
        to_print = []
        for (s, p, o) in facts_scores_dict:
            to_print.append((facts_scores_dict[s, p, o], (s, p, o)))
        to_print = sorted(to_print, reverse=True)
        with open(args.print_entailed_facts, 'w') as output:
            for (score, (s, p, o)) in to_print:
                output.write("{}\t{}\t{}\n".format(s, p, o))
        with open(args.print_entailed_facts + '_scored', 'w') as output2:
            for (score, (s, p, o)) in to_print:
                output2.write("{}\t{}\t{}\t{}\n".format(s, p, o, score))
        output.close()

    threshold_list = [0.0000000001, 0.000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001] \
                     + arange(0.01, 1, 0.01).tolist()
    threshold_list = [round(elem, 10) for elem in threshold_list]
    number_of_positives = 0
    number_of_negatives = 0
    counter_all = 0
    counter_scored = 0
    # Each threshold is mapped to a 4-tuple containing true and false positives and negatives.
    threshold_to_counter = {0: [0, 0, 0, 0]}
    for threshold in threshold_list:
        threshold_to_counter[threshold] = [0, 0, 0, 0]
    entry_for = {"true_positives": 0, "false_positives": 1, "true_negatives": 2, "false_negatives": 3}

    test_positive_examples_path = args.test_positive_examples
    assert os.path.exists(test_positive_examples_path)
    print("Loading examples data from {}".format(test_positive_examples_path))
    test_positive_examples_dataset = data_parser.parse(test_positive_examples_path)

    test_negative_examples_path = args.test_negative_examples
    assert os.path.exists(test_negative_examples_path)
    print("Loading examples data from {}".format(test_negative_examples_path))
    test_negative_examples_dataset = data_parser.parse(test_negative_examples_path)

    test_examples_dataset = [(ex, '1') for ex in test_positive_examples_dataset] + \
                            [(ex, '0') for ex in test_negative_examples_dataset]

    for ((s, p, o), score) in test_examples_dataset:
        counter_all += 1
        # Check that the target fact has a score
        if (s, p, o) in facts_scores_dict:
            counter_scored += 1
        if score == '1':
            # Positive example
            number_of_positives += 1
            # First consider threshold 0
            # True positive
            if facts_scores_dict.get((s, p, o), 0) > 0:
                threshold_to_counter[0][entry_for["true_positives"]] += 1
            # False negative
            else:
                threshold_to_counter[0][entry_for["false_negatives"]] += 1
            # Consider all other thresholds
            for threshold in threshold_list:
                # True positive
                if facts_scores_dict.get((s, p, o), 0) > threshold:
                    threshold_to_counter[threshold][entry_for["true_positives"]] += 1
                # False negative
                else:
                    threshold_to_counter[threshold][entry_for["false_negatives"]] += 1
        # Negative example
        else:
            assert score == '0'
            number_of_negatives += 1
            # First consider threshold 0
            # False positive
            if facts_scores_dict.get((s, p, o), 0) > 0:
                threshold_to_counter[0][entry_for["false_positives"]] += 1
            # True negative
            else:
                threshold_to_counter[0][entry_for["true_negatives"]] += 1
            # Consider all other thresholds
            for threshold in threshold_list:
                # False positive
                if facts_scores_dict.get((s, p, o), 0) > threshold:
                    threshold_to_counter[threshold][entry_for["false_positives"]] += 1
                # True negative
                else:
                    threshold_to_counter[threshold][entry_for["true_negatives"]] += 1

    #  Compute and print result
    recall_vector = []
    precision_vector = []
    print("Total examples: {}".format(counter_all))
    print("Scored examples: {}".format(counter_scored))

    with open(args.output, 'w') as f:
        f.write("Threshold" + '\t' + "Precision" + '\t' + "Recall" + '\t' + "Accuracy" + '\t' + "F1 Score" + '\n')
        for threshold in threshold_to_counter:
            tp, fp, tn, fn = threshold_to_counter[threshold]
            f.write("{}\t{}\t{}\t{}\t{}\n".format(threshold, precision(tp, fp, tn, fn),
                                                  recall(tp, fp, tn, fn), accuracy(tp, fp, tn, fn),
                                                  f1score(tp, fp, tn, fn)))
            recall_vector.append(recall(tp, fp, tn, fn))
            precision_vector.append(precision(tp, fp, tn, fn))
        # Add extremal points for AUC. This ensures a perfect classifier has AUC 1, a random classifier has AUC 0.5,
        # and an `always wrong' classifier has an AUC 0.
        # Without this, a perfect classifier would have a score of 0!!
        precision_vector.insert(0, 0)
        precision_vector.append(1)
        recall_vector.insert(0, 1)
        recall_vector.append(0)
        # Get rid of NaNs
        recall_vector = nan_to_num(recall_vector)
        precision_vector = nan_to_num(precision_vector)
        f.write("Area under precision recall curve: {}\n".format(auprc(precision_vector, recall_vector)))

    f.close()
