#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ----
"""

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os.path
from src.encodings.canonical import CanonicalEncoderDecoder
from src.config.config import ExperimentConfig
from src.model.cd_graph import CDGraph
from src.utils.utils import TYPE_PRED


def train(cfg: ExperimentConfig, device, internal_encoder: CanonicalEncoderDecoder, model,
          cd_graph: CDGraph, train_examples, experiment_folder) :

    # Create positive examples for training
    train_y = torch.zeros_like(cd_graph.features)
    examples_excluded = 0
    for s, p, o in train_examples:
        if p == TYPE_PRED and s in cd_graph.node_names:
            train_y[cd_graph.node_names.index(s)][internal_encoder.unary_pred_position_dict[o]] = 1
        else:
            # We drop cd_examples mentioning new constants. Note that if we use ICLR as external decoder, this means
            # dropping all facts of the form R(a,b) where a and b never occur together in the training set.
            examples_excluded += 1

    # Convert to PyTorch Geometric Data objects
    # Data: "A plain old python object modeling a single graph with various (optional) attributes"
    #        Please note that edge_type is a custom attribute of the function, NOT related to the optional
    #        attribute edge_attr.
    train_data = Data(x=cd_graph.features, y=train_y, edge_index=cd_graph.edges, edge_type=cd_graph.edge_colours)
    # DataLoader: "Data loader which merges data objects from a torch_geometric.data.dataset to a mini-batch."
    #  Note that list train_data.to(device) is a Dataset. DataLoader only uses two methods within
    #  the dataset argument: __length__, and __getitem__, so it works with a list like this.
    train_loader = DataLoader(dataset=[train_data.to(device)], batch_size=1)

    # Select Adam as the optimisation algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    checkpoints_folder = experiment_folder / "checkpoints"
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    def train_epoch():
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

            # Compute GNN output
            # Instances of modules are callable, and what happens on the call depends on whether there are `hooks`.
            # There aren't in this case, in which case the call uses the `forward` method inside the model. And indeed,
            # the forward method extracts named attributes from its input which coincide with the names of the
            # attributes in the object `batch` that we pass as input to the instance `model` of this Module
            output, _ = model(batch) # Ignore the second output, which are the activations in intermediate layer

            # Target label
            label = y.to(device)
            lossFunc = torch.nn.BCELoss(reduction='none')
            # Compute loss matrix, to be reduced later
            loss = lossFunc(output, label)
            # Double check we're not getting NaNs
            assert(not (loss != loss).any())
            # We give different weight to positive and negative examples; we construct a weight matrix with weight of
            # 5.0 wherever there is a 1 output in the y vector and a 1.0 where there is a 0 (previously 0.5/5 or 0.1/10)
            # weight = torch.tensor([1.0, 5.0]).to(device)
            weight = torch.tensor([0.5, 5.0]).to(device)
            weight_ = weight[y.data.long()].view_as(y)
            loss = loss * weight_
            # Use sum reduction on loss, backpropagate
            loss.sum().backward()
            optimizer.step()
            # MONOTONICITY: Any weight components < 0 are immediately "clamped" to 0, but not the bias
            for name, param in model.named_parameters():
                if 'bias' not in name and cfg.non_negative_weights:
                    param.data.clamp_(0)
            total_loss += batch.num_graphs * loss.sum().item()

        return total_loss

    divisor = 200 # How often we'll report progress of GNN

    # Implementing a form of early stopping. Keep track of the lowest loss achieved, if we've had max_num_bad epochs
    # only achieving higher losses than the lowest one recorded, then stop early.
    min_loss = None
    num_bad_iterations = 0
    max_num_bad = 50

    print("Training model")
    # Train for a maximum of 20000 epochs, but expect to stop early
    for epoch in range(20000): # TODO: include these numbers in the experiment configuration
        loss = train_epoch()
        if min_loss is None: min_loss = loss
        if epoch % divisor == 0:
            print('Epoch: {:03d}, Loss: {:.5f}'.
                  format(epoch, loss))
            if epoch % 1000 == 0: # Save checkpoint for each 1000 epochs
                torch.save(model, checkpoints_folder / "{}_Epoch{}.pt".format("model", epoch))
        if loss >= min_loss:
            num_bad_iterations += 1
            if num_bad_iterations > max_num_bad:
                print("Stopping early")
                break
        else:
            num_bad_iterations = 0
            min_loss = loss

    torch.save(model, experiment_folder / "model.pt")

