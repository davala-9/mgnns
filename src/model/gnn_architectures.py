#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file contains the GNN architecture, as the GNN class.
One of the fundamental steps in the GNN's update rule is the
use of an appropriate convolution. We define 2 convolutions,
one for coloured edges, and one for colourless edges.

@author: ----
"""
import torch

from torch_geometric.nn import MessagePassing

import torch.nn.functional as F
from torch.nn import Parameter

# Define a convolution step (will be used in each layer of the model)
class EC_GCNConv(MessagePassing):

    # in_channels (int) - Size of each input sample
    # out_channels (int) - Size of each output sample
    def __init__(self, in_channels, out_channels, edge_colours, aggregation):

        aggr_name = getattr(aggregation, "value", aggregation)
        if aggr_name not in {"sum","max"}:
            raise ValueError(f"Unsupported aggregation mode: {aggr_name!r}")
        super(EC_GCNConv, self).__init__(aggr=aggr_name)
        self.weights = Parameter(torch.Tensor(edge_colours, out_channels, in_channels))
        self.weights.data.normal_(0, 0.001)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_colours = edge_colours
        
    def forward(self, x, edge_index, edge_colour):
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        for i in range(self.edge_colours):
            edge_mask = edge_colour == i
            temp_edges = edge_index[:, edge_mask]
            out += F.linear(self.propagate(temp_edges, x=x, size=(x.size(0), x.size(0))), self.weights[i], bias=None)
        return out

class GNN(torch.nn.Module):

    # The last layer computes sigmoid(x - OUTPUT_SHIFT) rather than sigmoid(x). Since the biases are unconstrained
    # this is mathematically irrelevant, but the models were trained with it, so bias(2) includes it too.
    # Note: saved model.pt files pickle whole GNN instances, so new state must not be added to __init__.
    OUTPUT_SHIFT = 10

    def __init__(self, feature_dimension, num_edge_colours, aggregation_1, aggregation_2):
        super(GNN, self).__init__()

        self.num_layers = 2 # Currently hardcoded!

        self.num_colours = num_edge_colours
        # From layer 0 (left) to layer L (right)
        self.dimensions = [feature_dimension, 2*feature_dimension, feature_dimension]

        self.agg_1 = aggregation_1
        self.agg_2 = aggregation_2

        self.conv1 = EC_GCNConv(self.dimensions[0], self.dimensions[1], num_edge_colours, self.agg_1)
        self.conv2 = EC_GCNConv(self.dimensions[1], self.dimensions[2], num_edge_colours, self.agg_2)

        self.lin_self_1 = torch.nn.Linear(self.dimensions[0], self.dimensions[1])
        self.lin_self_2 = torch.nn.Linear(self.dimensions[1], self.dimensions[2])
        
        self.output = torch.nn.Sigmoid()

    # One thing to keep in mind is that since this is a torch.nn.Module, you can call a GNN by writing model([yourdata])
    # and this essentially calls this forward. So think of this as a __call__ method

    # Note also that unlike most "forward" implementations, this returns all feature vectors of intermediate layers.

    def forward(self, data):
        x, edge_index, edge_colour = data.x, data.edge_index, data.edge_type

        # Layer 1
        x = self.lin_self_1(x) + self.conv1(x, edge_index, edge_colour)
        x = torch.relu(x)
        features_1 = x.detach().clone() # Detached so that it does not participate in the computation graph

        # Layer 2
        x = self.lin_self_2(x) + self.conv2(x, edge_index, edge_colour)
        x = self.output(x - self.OUTPUT_SHIFT)

        return x, features_1

    def layer_dimension(self, layer):
        return self.dimensions[layer]

    # Returns the element of `per_layer` (a pair: layer 1, layer 2) that belongs to `layer`.
    def _for_layer(self, layer, per_layer):
        if layer not in (1, 2):
            raise ValueError(f"invalid layer: {layer!r} (model has {self.num_layers} layers)")
        return per_layer[layer - 1]

    def matrix_A(self, layer):
        return self._for_layer(layer, (self.lin_self_1, self.lin_self_2)).weight.detach()

    def matrix_B(self, layer, colour):
        return self._for_layer(layer, (self.conv1, self.conv2)).weights[colour].detach()

    def bias(self, layer):
        bias = self._for_layer(layer, (self.lin_self_1, self.lin_self_2)).bias.detach()
        return bias - self.OUTPUT_SHIFT if layer == 2 else bias

    def activation(self, layer):
        return self._for_layer(layer, (torch.relu, torch.nn.Sigmoid()))

    def aggregation_function(self, layer):
        return self._for_layer(layer, (self.agg_1, self.agg_2))
