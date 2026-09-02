#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ----
"""
import csv
import time
import torch
from pathlib import Path
from torch_geometric.data import Data

import numpy as np

from itertools import combinations

import requests

import re

from tqdm import tqdm
import os.path
import sys
import bisect

from src.utils.bitset import BitSet

# Code in this file for interfacing with RDFox is based off that found here: 
# https://docs.oxfordsemantic.tech/getting-started.html

rdfox_server = "http://localhost:8080"
TYPE_PRED = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'

def find_index_and_insert(sorted_list: list[int], element: int) -> tuple[int, bool]:
    idx = bisect.bisect_left(sorted_list, element)
    if idx == len(sorted_list) or sorted_list[idx] != element:
        sorted_list.insert(idx, element)
        return idx, True
    return idx, False

def check(path: Path, fileid):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {fileid} file not found in {path}")
    else:
        return path

# Clamp to 0 values of a matrix with absolute value smaller or equal than a given threshold.
# Use option negative_only to clamp only the negative values
def threshold_matrix_values(matrix: torch.tensor, threshold: float, negative_only=False):
    below_threshold_mask = matrix <= -threshold
    above_threshold_mask = matrix >= threshold
    if negative_only:
        outside_threshold_mask = torch.logical_or(below_threshold_mask, matrix >= 0)
    else:
        outside_threshold_mask = torch.logical_or(below_threshold_mask, above_threshold_mask)
    inside_threshold_mask = torch.logical_not(outside_threshold_mask)
    matrix[inside_threshold_mask] = 0
    return torch.sum(outside_threshold_mask == False), torch.numel(outside_threshold_mask)

def assert_response_ok(response, message):
    '''Helper function to raise an exception if the REST endpoint returns an
    unexpected status code.'''
    if not response.ok:
        raise Exception(
            message + "\nStatus received={}\n{}".format(response.status_code, response.text))


def load_predicates(predicates_file):
    '''Load the predicates from their file into memory, return them.'''
    # Lists to store binary and unary predicates
    binary_predicates = []
    unary_predicates = []

    # IT IS CRITICAL TO RESPECT THE ORDER OF PREDICATES IN THE LIST!!
    try:
        with open(predicates_file, 'r') as f:
            for line in f:
                # Every line is of form "predicate,arity"
                pair = line.split(',')
                if int(pair[1][:-1]) == 1:  # [:-1] to get rid of \n
                    unary_predicates.append(pair[0])
                else:
                    binary_predicates.append(pair[0])

        # TODO: sanity check - no duplicates
        return binary_predicates, unary_predicates

    except FileNotFoundError:
        raise FileNotFoundError('Predicates file {} not found.'.format(predicates_file))


# Simplifies redundant atoms in the body of a rule, where atoms are written as triples
def remove_redundant_atoms(rule_body):
    x1 = "X1"
    variable_to_unary_predicates = {x1: set()}
    parent_to_children = {}
    for (s, p, o) in rule_body:
        if p == TYPE_PRED:
            if s in variable_to_unary_predicates:
                variable_to_unary_predicates[s].add(o)
            else:
                variable_to_unary_predicates[s] = {o}
        else:
            if s not in variable_to_unary_predicates:
                variable_to_unary_predicates[s] = set()
            if o not in variable_to_unary_predicates:
                variable_to_unary_predicates[o] = set()
            if o in parent_to_children:
                parent_to_children[o].add((p, s))
            else:
                parent_to_children[o] = {(p, s)}
    var_to_level = {x1: 0}
    frontier = [x1]
    while frontier:
        y = frontier.pop()
        if y in parent_to_children:
            for (_, z) in parent_to_children[y]:
                var_to_level[z] = var_to_level[y] + 1
                frontier.append(z)
    new_parent_to_children = {}
    var_type = {}
    for level in range(max(var_to_level.values()), -1, -1):
        for y in var_to_level:
            if var_to_level[y] == level:
                subtrees = set()
                if y in parent_to_children:
                    for (R, z) in parent_to_children[y]:
                        if (R, var_type[z]) not in subtrees:
                            subtrees.add((R, var_type[z]))
                            if y in new_parent_to_children:
                                new_parent_to_children[y].add((R, z))
                            else:
                                new_parent_to_children[y] = {(R, z)}
                var_type[y] = frozenset(subtrees.union(variable_to_unary_predicates[y]))
    new_rule_body = []
    frontier = [x1]
    while frontier:
        y = frontier.pop()
        for pred in variable_to_unary_predicates[y]:
            new_rule_body.append((y, TYPE_PRED, pred))
        if y in new_parent_to_children:
            for (R, z) in new_parent_to_children[y]:
                new_rule_body.append((z, R, y))
                frontier.append(z)
    return new_rule_body

# Takes as input:
# -- a bitset of dimension m representing a relevant subset of features in some layer
# -- a matrix of dimension m x n (typically a matrix for the previous layer)
# -- optionally, a feature vector of dimension n that represent specific activations in the previous layer
# It computes the relevant subset of features in the previous layer
def backpropagate_relevance(current_relevant: BitSet, matrix, previous_activations=None):
    if matrix.shape[0] != current_relevant.dimension:
        raise ValueError(f"Left vector dimension {current_relevant.dimension} does not match matrix row dimension"
                         f"{matrix.shape[0]}")
    if previous_activations is not None and matrix.shape[1] != previous_activations.shape[0]:
        raise ValueError(f"Right vector dimension {previous_activations.shape[0]} does not match matrix column dimension"
                         f"{matrix.shape[1]}")
    if not current_relevant.is_empty():
        matrix_relevant_rows = matrix[current_relevant.elements(), :]
        any_positive = (matrix_relevant_rows > 0).any(dim=0) # dim_{l-1} Boolean vector
        if previous_activations is not None:
            mask = (previous_activations > 0) & any_positive
        else:
            mask = any_positive
        return BitSet.from_subset(matrix.shape[1], set(torch.where(mask)[0].tolist()))
    else:
        # For some reason the above formula does not work in the degenerate case where it's all zeroes.
        return BitSet.from_subset(matrix.shape[1], set())



 # def threshold_matrix_values(matrix: torch.tensor, threshold: float, negative_only=False):
    #     below_threshold_mask = matrix <= -threshold
    #     above_threshold_mask = matrix >= threshold
    #     if negative_only:
    #         outside_threshold_mask = torch.logical_or(below_threshold_mask, matrix >= 0)
    #     else:
    #         outside_threshold_mask = torch.logical_or(below_threshold_mask, above_threshold_mask)
    #     inside_threshold_mask = torch.logical_not(outside_threshold_mask)
    #     matrix[inside_threshold_mask] = 0
    #
    # #Model clamping
    # if args.model_clamping:
    #     for layer in range(model.num_layers + 1):
    #         mat = model.matrix_A(layer)
    #         threshold_matrix_values(model.matrix_A(layer), float(args.model_clamping))
    #         for colour in range(model.num_colours + 1):
    #             threshold_matrix_values(model.matrix_B(layer, colour), float(args.model_clamping))

 #    # Auxiliary method - I think this is for one of the methods below?
 #    def get_conjunction_for_contributor(var_y, pos_j):
 #        conj = set()
 #        conj.add((var_y, type_pred, can_encoder_decoder.position_unary_pred_dict[pos_j]))
 #        vari = var_y
 #        for cur_lay in range(1, L + 1, 1):
 #            if (vari, cur_lay) in predecessors:
 #                (new_vari, colr, _) = predecessors[(vari, cur_lay)]
 #                conj.add((vari, can_encoder_decoder.colour_binary_pred_dict[colr], new_vari))
 #                vari = new_vari
 #        return conj
 #
 #    # Auxiliary method - I think this is also for one of the methods below
 #    def test_gr_dataset(body):
 #        if not body:
 #            gr_features = torch.FloatTensor(np.zeros((1, model.layer_dimension(0))))
 #            gr_edge_list = torch.LongTensor(2, 0)
 #            gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=torch.LongTensor([])).to(device)
 #            gnn_output_gr = model(gr_dataset)
 #            return gnn_output_gr[0][cd_fact_pred_pos] >= cfg.derivation_threshold
 #        else:
 #            # Note that here variables are treated as constants. Thus, to recover the relevant node in
 #            # the gr graph, we need to call nodes.const_node_dict[z], not nu_variable_to_node[z].
 #            (gr_features, node_to_gr_row_dict, gr_edge_list, gr_colour_list) = can_encoder_decoder.encode_dataset(body)
 #            gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(device)
 #            gnn_output_gr = model(gr_dataset)
 #            a = gnn_output_gr[node_to_gr_row_dict[nodes.const_node_dict[x1]]][cd_fact_pred_pos]
 #            b = cfg.derivation_threshold
 #            return a >= b
 #
 #
 # # Weight clamping. Note that this modifies the model.
 #        if cfg.clamping > 0:
 #            n_clamped_weights = 0
 #            n_total_weights = 0
 #            for layer in range(1, model.num_layers + 1):
 #                a, b = threshold_matrix_values(model.matrix_A(layer), cfg.clamping)
 #                n_clamped_weights += a
 #                n_total_weights += b
 #                for colour in range(model.num_colours):
 #                    a, b = threshold_matrix_values(model.matrix_B(layer, colour), cfg.clamping)
 #                    n_clamped_weights += a
 #                    n_total_weights += b
 #            print("Percentage of clamped weights: {}".format(n_clamped_weights / n_total_weights))

#RULE CORRECTNESS CHECKS
# Correctness check: check that each atom is grounded in the canonical dataset via \nu
# for (s, p, o) in rule_body:
#     if p == TYPE_PRED:
#         constant = nodes.node_const_dict[nu_variable_to_node_dict[s]]
#         assert (constant, p, o) in cd_dataset, "ERROR: This rule does not unify with the dataset. Bug."
#     else:
#         origin_constant = nodes.node_const_dict[nu_variable_to_node_dict[s]]
#         dest_constant = nodes.node_const_dict[nu_variable_to_node_dict[o]]
#         assert (origin_constant, p, dest_constant) in cd_dataset, \
#             "ERROR: the extracted rule does not unify with the dataset. Bug."
# # Correctness check: ensure that the rule is captured.
# if not rule_body:
#     gr_features = torch.FloatTensor(np.zeros((1, model.layer_dimension(0))))
#     gr_edge_list = torch.LongTensor(2, 0)
#     gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=torch.LongTensor([])).to(device)
#     gnn_output_gr = model(gr_dataset)
#     assert gnn_output_gr[0][cd_pred_pos] >= cfg.derivation_threshold, \
#         "ERROR: the extracted rule seems not to be captured by the model. This means there is a bug."
# else:
#     (gr_features, node_to_gr_row_dict, gr_edge_list,
#      gr_colour_list) = can_encoder_decoder.encode_dataset(rule_body)
#     gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(device)
#     gnn_output_gr = model(gr_dataset)
#     assert (gnn_output_gr[node_to_gr_row_dict[nodes.const_node_dict[x1]]][cd_pred_pos] >
#             cfg.derivation_threshold), \
#         "ERROR: the extracted rule seems not to be captured by the model. This means there is a bug."