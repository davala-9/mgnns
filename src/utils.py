#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ----
"""
import csv
import time
import torch
from torch_geometric.data import Data

import numpy as np

from itertools import combinations

import requests

import re

from tqdm import tqdm

# Code in this file for interfacing with RDFox is based off that found here: 
# https://docs.oxfordsemantic.tech/getting-started.html

rdfox_server = "http://localhost:8080"
RDF_type_string = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'


def assert_response_ok(response, message):
    '''Helper function to raise an exception if the REST endpoint returns an
    unexpected status code.'''
    if not response.ok:
        raise Exception(
            message + "\nStatus received={}\n{}".format(response.status_code,
                                                        response.text))

def dlog_to_RDF(dlog_array, prefix_dict, append_str=''):
    '''Convert an array of predicates in Datalog form to RDF form.'''
    RDF_strings = []
    for dlog_pred in dlog_array:
        # For each Datalog predicate, separate it into its prefix and main
        # component
        prefix, pred = dlog_pred.split(':')
        # Get the variables, remove the ? at the front, make the letters
        # lowercase:
        constants = [x[1:].lower() + append_str for x in pred.split('[')[1][:-1].split(',')]
        # Always of form Predicate[?X, ?Y], therefore by splitting at [ we get
        # the predicate name
        pred_name = pred.split('[')[0]
        if len(constants) == 2:  # Then binary predicate
            RDF_string = '<{}> <{}> <{}> .'.format(constants[0],
                                                   prefix_dict[prefix] + pred_name,
                                                   constants[1])
        else:
            assert (len(constants)) == 1
            RDF_string = '<{}> <{}> <{}> .'.format(constants[0],
                                                   RDF_type_string,
                                                   prefix_dict[prefix] + pred_name)
        RDF_strings.append(RDF_string)

    return RDF_strings


def load_predicates(predicates_file):
    '''Load the predicates from their file into memory, return them.'''
    # Lists to store binary and unary predicates
    binary_predicates = []
    unary_predicates = []

    try:
        with open(predicates_file, 'r') as f:
            for line in f:
                # Every line is of form "predicate,arity"
                pair = line.split(',')
                if int(pair[1][:-1]) == 1:  # [:-1] to get rid of \n
                    unary_predicates.append(pair[0])
                else:
                    binary_predicates.append(pair[0])
        return binary_predicates, unary_predicates

    except FileNotFoundError:
        raise FileNotFoundError('Predicates file {} not found.'.format(predicates_file))








def decode_and_get_threshold(node_dict, num_binary, num_unary, binaryPredicates, unaryPredicates,
                             feature_vectors, threshold):
    '''Decode feature vectors back into a dataset.
    Additionally report back the threshold at which all facts in the dataset would no longer be predicted'''
    threshold_indices = torch.nonzero(feature_vectors >= threshold)
    GNN_dataset = set()
    for i, index in enumerate(threshold_indices):
        index = index.tolist()
        const_index = index[0]
        pred_index = index[1]
        extraction_threshold = feature_vectors[index[0], index[1]]
        const = node_dict[const_index]
        if type(const) is tuple:  # Then we just want to consider this if it's in the binary preds
            if pred_index < num_binary:
                predicate = binaryPredicates[pred_index]
                RDF_triplet = "{} {} {}".format(const[0], predicate, const[1])
                GNN_dataset.add((RDF_triplet, extraction_threshold))
        else:  # Then we're dealing with a unary predicate (second section of the vec)
            if pred_index >= num_binary:
                predicate = unaryPredicates[pred_index - num_binary]
                RDF_triplet = "{} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> {}".format(const, predicate)
                GNN_dataset.add((RDF_triplet, extraction_threshold))
    return GNN_dataset

def output_scores(encoding_scheme, model, binaryPredicates, unaryPredicates, incomplete_graph, examples, device='cpu'):
    '''Give the scores for the facts in the query dataset.'''
    num_binary = len(binaryPredicates)
    num_unary = len(unaryPredicates)
    print("Encoding input dataset...")
    (dataset_x, edge_list, edge_type,
     node_to_const_dict, dataset_const_to_node_dict, pred_dict) = encode_input_dataset(encoding_scheme,
                                                                                       incomplete_graph,
                                                                                       examples,
                                                                                       binaryPredicates,
                                                                                       unaryPredicates)
    print("Encapsulating input data...")
    test_data = Data(x=dataset_x, edge_index=edge_list, edge_type=edge_type).to(device)
    print("Applying model to data...")
    entailed_facts_encoded = model(test_data)
    print("Decoding...")
    nonzero_scores_and_facts = decode_with_scores(node_to_const_dict, num_binary, num_unary, binaryPredicates,
                                                  unaryPredicates, entailed_facts_encoded)
    print("Done.")
    return nonzero_scores_and_facts
