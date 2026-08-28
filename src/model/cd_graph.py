import torch
from dataclasses import dataclass, field

# Implementation of a (col,d)-graph. It simply packages many variables into one for convenience, with some checks.
# Colours are ALWAYS represented by integers 1...n (this is what the pytorch geometric model needs)
# Features is a |nodes| x delta matrix. Each row represents a node and its feature.
# Edges is a 2 x |edges| matrix where each column is of the form [i,j] representing an edge from the node
# represented by row i of self.features to the node represented by row j
# Edge_colours is a |edges|-sized list where the ith element is the colour of edge in column i of edges.
# Node_names is a |nodes|-sized list where the ith element is the name of the node in row i of self.features.
class CDGraph:

    def __init__(self, col_size: int, delta: int, features: torch.FloatTensor, edges: torch.LongTensor,
                 edge_colours: torch.LongTensor, node_names: list):

        # Sanity checks on the input
        assert col_size > 0
        assert delta > 0
        assert features.shape[0] == len(node_names)
        assert features.shape[1] == delta
        assert edges.shape[1] == edge_colours.shape[0]
        assert edges.shape[0] == 2
        assert all(colour in range(col_size) for colour in edge_colours)
        assert len(node_names) == len(set(node_names)) # No repeated node names

        self.col_size = col_size
        self.delta = delta
        self.features = features
        self.edges = edges
        self.edge_colours = edge_colours
        self.node_names = node_names
        self.node_names_to_indices = {n_name: index for index, n_name in enumerate(self.node_names)}


    def clone(self):
        return CDGraph(col_size=self.col_size,delta=self.delta,features=self.features,
                       edges=self.edges,edge_colours=self.edge_colours,node_names=self.node_names)

@dataclass
class TraceCollector:
    cd_graph: CDGraph = None
    activations: dict[int, torch.Tensor] = field(default_factory=dict)
