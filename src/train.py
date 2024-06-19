#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ----
"""

import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import argparse
import os.path
from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder
import nodes
import sys
import data_parser

from utils import load_predicates
from gnn_architectures import GNN

parser = argparse.ArgumentParser(description="Train the GNNs")
parser.add_argument('--model-name',
                    help='Name of the model to be learned.')
parser.add_argument('--model-folder',
                    help='Name of the folder where the learned model will be stored')
parser.add_argument('--predicates',
                    help='File with the fixed, ordered list of predicates we consider.')
parser.add_argument('--train-graph',
                    nargs='?',
                    default=None,
                    help='Filename of training data with input graph, including extension.')
parser.add_argument('--train-examples',
                    nargs='?',
                    default=None,
                    help='Filename of training data with positive examples (facts), including extension.')
parser.add_argument('--encoding-scheme',
                    default='canonical',
                    nargs='?',
                    choices=['iclr22', 'canonical'],
                    help='Choose the encoder-decoder that will be applied to the data (canonical by default).')
parser.add_argument('--encoder-folder',
                    help='Name of the folder where the used encoder/decoder(s) will be stored')
parser.add_argument('--aggregation',
                    default='max-max',
                    choices=['max-max', 'max-sum', 'sum-max', 'sum-sum'],
                    help='Aggregation function to be used by the model')
parser.add_argument('--train-with-dummies',
                    default=False,
                    action='store_true'),
# parser.add_argument('--model-clamping',
#                     default=None,
#                     help='Clamp to zero all weights with absolute value under this number at the end')
parser.add_argument('--non-negative-weights',
                    choices=['True', 'False'],
                    help='Restrict matrix weights so that they are non-monotonic')
args = parser.parse_args()

saved_model_name = args.model_name

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading predicates from {}".format(args.predicates))
    data_binary_predicates, data_unary_predicates = load_predicates(args.predicates)
    print("{} unary predicates and {} binary predicates in the signature.".format(len(data_unary_predicates),
                                                                                  len(data_binary_predicates)))

    train_graph_path = args.train_graph
    assert os.path.exists(train_graph_path)
    print("Loading graph data from {}".format(train_graph_path))
    train_graph_dataset = data_parser.parse(train_graph_path)

    # 'cd' is short for (col,\delta), referring to the (col,\delta)-signature
    if args.encoding_scheme == 'canonical':
        cd_unary_predicates = data_unary_predicates
        cd_binary_predicates = data_binary_predicates
        cd_dataset = train_graph_dataset
        print("Using canonical encoding scheme.")
    else:
        iclr_encoder_decoder = ICLREncoderDecoder(load_from_document=None,
                                                  unary_predicates=data_unary_predicates,
                                                  binary_predicates=data_binary_predicates)
        iclr_encoder_decoder.save_to_file(args.encoder_folder + '/' + saved_model_name + '_iclr22' + '.tsv')
        cd_unary_predicates = iclr_encoder_decoder.canonical_unary_predicates()
        cd_binary_predicates = iclr_encoder_decoder.canonical_binary_predicates()
        cd_dataset = iclr_encoder_decoder.encode_dataset(train_graph_dataset)
        print("Using ICLR22 encoding scheme.")
        print("{} unary predicates and {} binary predicates in the (col,delta) signature.".format(
            len(cd_unary_predicates), len(cd_binary_predicates)))

    can_encoder_decoder = CanonicalEncoderDecoder(load_from_document=None,
                                                  unary_predicates=cd_unary_predicates,
                                                  binary_predicates=cd_binary_predicates)
    can_encoder_decoder.save_to_file(args.encoder_folder + '/' + saved_model_name + '_canonical' + '.tsv')

    # train_x : torch.FloatTensor of size i x j, with i the number of graph nodes, j the length of feature vectors
    # train_nodes: dictionary mapping each node in the graph to the corresponding row of train_x
    # train_edge_list : torch.LongTensor with all edges in the graph, each edge is a pair of nodes (integers)
    # train_edge_colour_list : torch.LongTensor where the ith component is the colour of the ith edge in train_edge_list
    (train_x, train_nodes, train_edge_list, train_edge_colour_list) = \
        can_encoder_decoder.encode_dataset(cd_dataset, use_dummy_constants=args.train_with_dummies)

    train_examples_path = args.train_examples
    assert os.path.exists(train_examples_path)
    print("Loading graph data from {}".format(train_examples_path))
    train_examples_dataset = []
    examples = data_parser.parse(train_examples_path)
    examples_excluded = 0
    for s, p, o in examples:
        # TODO: we can revise this and instead encode all of these in the input, see if that improves performance
        # NOTE: Drop all examples introducing nodes not in the training graph, as no predictions are generated for them
        if args.encoding_scheme == 'canonical':
            _, e_nodes, _, _ = can_encoder_decoder.encode_dataset([(str(s), str(p), str(o))])
        elif args.encoding_scheme == 'iclr22':
            cd_dataset_examples = iclr_encoder_decoder.encode_dataset([(str(s), str(p), str(o))])
            _, e_nodes, _, _ = can_encoder_decoder.encode_dataset(cd_dataset_examples)
        exclude_example = False
        for node in e_nodes:
            if node not in train_nodes:
                exclude_example = True
        if exclude_example:
            examples_excluded += 1
        else:
            train_examples_dataset.append((str(s), str(p), str(o)))
    if args.encoding_scheme == 'canonical':
        cd_dataset_examples = train_examples_dataset
    elif args.encoding_scheme == 'iclr22':
        cd_dataset_examples = iclr_encoder_decoder.encode_dataset(train_examples_dataset)
    # train_y : torch.FloatTensor of the same size as train_x
    # examples are encoded as graphs equal to train_x where all labels are 0 except those corresp to facts in examples
    train_y = torch.zeros_like(train_x)
    (new_y, examples_nodes, _, _) = can_encoder_decoder.encode_dataset(cd_dataset_examples)
    for node in examples_nodes:
        train_y[train_nodes[node]] = new_y[examples_nodes[node]]

    # Convert to PyTorch Geometric Data objects
    # Data: "A plain old python object modeling a single graph with various (optional) attributes"
    #        Please note that edge_type is a custom attribute of the function, NOT related to the optional
    #        attribute edge_attr.
    train_data = Data(x=train_x, y=train_y, edge_index=train_edge_list, edge_type=train_edge_colour_list)
    # DataLoader: "Data loader which merges data objects from a torch_geometric.data.dataset to a mini-batch."
    #  Note that list train_data.to(device) is a Dataset. DataLoader only uses two methods within
    #  the dataset argument: __length__, and __getitem__, so it works with a list like this.
    train_loader = DataLoader(dataset=[train_data.to(device)], batch_size=1)

    model = GNN(feature_dimension=len(cd_unary_predicates), num_edge_colours=len(cd_binary_predicates),
                aggregation=args.aggregation).to(device)
    # Select Adam as the optimisation algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    checkpoints_folder = args.model_folder + "/checkpoints"
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    def train():
        # Set module in training mode (this method is inherited from torch.nn.Module)
        model.train()
        
        total_loss = 0

        # Notice how here we are iterating over the elements of train_loader, according to the documentation is
        # a DataLoader, which in turn means that iteration is entirely controlled by the iterable data structure
        # that implements whichever Dataset argument was used on creation on the DataLoader. In our case, the Dataset
        # is a Pytorch Geometric Data object, which provides an iterable method where it simply provides a tuple with
        # attributes, their names and values. In short, a batch here is iterating through 4-tuples of the form
        #  x, y, edge_index, edge_type
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            y = batch.y
            # Construct a weight matrix with weight of 5.0 wherever there is a
            # 1 output in the y vector, 0.5 where there is a 0.
            weight = torch.tensor([0.5, 5.0]).to(device)
            # .data is a tensor method that gives you the values; .long() transforms it to long format
            # ALSO: bear in mind that y is going to be a single number, because it is just one element in the batch
            # ALSO: view_as is an operation of tensors to make it look the same size as y: so essentially we are looking
            #       at weight as a tensor of the same size as y.
            weight_ = weight[y.data.long()].view_as(y)
            # Compute GNN output
            # Instances of modules are callable, and what happens on the call depends on whether there are `hooks`.
            # There aren't in this case, in which case the call uses the `forward` method inside the model. And indeed,
            # the forward method extracts named attributes from its input which coincide with the names of the
            # attributes in the object `batch` that we pass as input to the instance `model` of this Module
            output = model(batch)
            # Target label
            label = y.to(device)
            lossFunc = torch.nn.BCELoss(reduction='none')
            # Compute loss matrix, to be reduced later
            loss = lossFunc(output, label)
            
            # Double check we're not getting NaNs
            assert(not (loss != loss).any())
            loss = loss * weight_
            # Use sum reduction on loss, backpropagate
            loss.sum().backward()
            optimizer.step()
            # Any weight components < 0 are immediately "clamped" to 0, but not the bias
            for name, param in model.named_parameters():
                if 'bias' not in name and args.non_negative_weights == 'True':
                    param.data.clamp_(0)
            total_loss += batch.num_graphs * loss.sum().item()
        return total_loss

    # Train for a maximum of 50000 epochs, but expect to stop early

    # How often we'll report progress of GNN
    divisor = 200

    # Implementing a form of early stopping. Keep track of the lowest loss
    # achieved, if we've had n epochs (to be specified) only achieving higher
    # losses than the lowest one recorded, then stop early.
    min_loss = None
    num_bad_iterations = 0
    # Maximum number of epochs reporting higher loss than lowest achieved before we stop early
    max_num_bad = 50

    print("Training model {}.".format(args.model_name))

    for epoch in range(50000):
        loss = train()
        if min_loss is None: min_loss = loss
        if epoch % divisor == 0:
            print('Epoch: {:03d}, Loss: {:.5f}'.
                  format(epoch, loss))
            if epoch % 1000 == 0:
                torch.save(model,
                           checkpoints_folder + '/' +
                           "{}_Epoch{}.pt".format(args.model_name, epoch))
        if loss >= min_loss:
            num_bad_iterations += 1
            if num_bad_iterations > max_num_bad:
                print("Stopping early")
                break
        else:
            num_bad_iterations = 0
            min_loss = loss

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

    torch.save(model, args.model_folder + '/' + saved_model_name + '.pt')
