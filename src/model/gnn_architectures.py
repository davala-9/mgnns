#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file contains the GNN architecture, as the GNN class.
One of the fundamental steps in the GNN's update rule is the
use of an appropriate convolution. We define 2 convolutions,
one for coloured edges, and one for colourless edges.

@author: ----
"""
from os import write

import torch

from torch_geometric.nn import MessagePassing

import torch.nn.functional as F
from torch.nn import Parameter

# Define a convolution step (will be used in each layer of the model)
class EC_GCNConv(MessagePassing):

    # in_channels (int) - Size of each input sample
    # out_channels (int) - Size of each output sample
    def __init__(self, in_channels, out_channels, edge_colours, aggregation):

        self.aggr = aggregation
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

    def __init__(self, dimensions, num_edge_colours, aggregations, conv_builder=EC_GCNConv):
        super(GNN, self).__init__()

        self.num_layers = len(dimensions) - 1

        self.num_colours = num_edge_colours
        # From layer 0 (left) to layer L (right)
        self.dimensions = dimensions

        if len(aggregations) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} aggregations, but got {len(aggregations)}.")
        else:
            self.aggregations = aggregations

        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(self.num_layers):
            in_dim = self.dimensions[i]
            out_dim = self.dimensions[i + 1]
            agg = self.aggregations[i]

            self.convs.append(conv_builder(in_dim, out_dim, num_edge_colours, agg))
            self.lins.append(torch.nn.Linear(in_dim, out_dim))
        
        self.output = torch.nn.Sigmoid()

    # One thing to keep in mind is that since this is a torch.nn.Module, you can call a GNN by writing model([yourdata])
    # and this essentially calls this forward. So think of this as a __call__ method

    # Note also that unlike most "forward" implementations, this returns all feature vectors of intermediate layers.

    def forward(self, data):
        x, edge_index, edge_colour = data.x, data.edge_index, data.edge_type

        intermediate_features = []

        for i in range(self.num_layers):
            x = self.lins[i](x) + self.convs[i](x, edge_index, edge_colour)

            # Apply ReLU and save detached features for all but the final layer
            if i < self.num_layers - 1:
                x = torch.relu(x)
                intermediate_features.append(x.detach().clone())
            else:
                x = self.output(x - 10)

        return x, intermediate_features

    def layer_dimension(self, layer):
        return self.dimensions[layer]

    def matrix_A(self, layer):
        # Adjusted for 0-indexing internally while keeping 1-indexing externally
        idx = layer - 1
        if 0 <= idx < self.num_layers:
            return self.lins[idx].weight.detach()
        return None

    def matrix_B(self, layer, colour):
        idx = layer - 1
        if 0 <= idx < self.num_layers:
            # Note: This assumes the custom convolution has a 'weights' attribute
            if hasattr(self.convs[idx], 'weights'):
                return self.convs[idx].weights[colour].detach()
        return None

    def bias(self, layer):
        idx = layer - 1
        if 0 <= idx < self.num_layers:
            bias_val = self.lins[idx].bias.detach()
            if idx == self.num_layers - 1:
                return bias_val - 10
            return bias_val
        return None

    def activation(self, layer):
        idx = layer - 1
        if 0 <= idx < self.num_layers - 1:
            return torch.relu
        elif idx == self.num_layers - 1:
            return torch.nn.Sigmoid()
        return None

    def aggregation_function(self, layer):
        idx = layer - 1
        if 0 <= idx < self.num_layers:
            return self.aggregations[idx]
        return None
#
