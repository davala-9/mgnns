import numpy as np
import torch
import pytest

from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.iclr22 import ICLREncoderDecoder
from src.encodings.noncanonical.identity import IdentityEncoderDecoder
from src.model.cd_graph import TraceCollector, CDGraph
from src.model.gnn_architectures import GNN
from src.model.gnn_transformation import apply_model, apply_nc_decoder
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, Variable, walk
from src.rule_extraction.fact_explanation import FactExplainer, FactContext
from src.utils.utils import TYPE_PRED
from src.utils.bitset import BitSet

# ----------------------------------------------------------------------------------------------------------
# EXAMPLE 1 - CANONICAL ENCODING
# ----------------------------------------------------------------------------------------------------------

# Assume a signature with A, B (unary), R, S (binary)
# Assume also a canonical encoding so (\col,\delta) = (2,2)
# Layer_1:
# Feature 1: current node has feature A
# Feature 2: R-neighbour has feature A
# Feature 3: R-neighbour has feature B and S-neighbour has feature B
# Feature 4: fixed value of 1
# Layer 2:
# Feature 1: current node has features 1 & 2 and S-neighbour with feature 1
# Feature 2: has an S-neighbour with feature 3 & S-neighbour with feature 4
def example_model_1(device):
    model = GNN(
        dimensions=[2,4,2],
        num_edge_colours=2,
        aggregations=["max","max"]).to(device)
    # Checks
    assert model.num_layers == 2
    assert model.num_colours == 2
    assert model.dimensions == [2,4,2]
    # Matrix B1-c1
    model.convs[0].weights.data[0] = torch.tensor([
    [0., 0.],
    [1., 0.],
    [0., 1.],
    [0., 0.]
    ])
    # Matrix B1-c2
    model.convs[0].weights.data[1] = torch.tensor([
        [0., 0.],
        [0., 0.],
        [0., 1.],
        [0., 0.]
    ])
    # Matrix B2-c1
    model.convs[1].weights.data[0] = torch.tensor([
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]
    ])
    # Matrix B2-c2
    model.convs[1].weights.data[1] = torch.tensor([
        [1., 0., 0., 0.],
        [0., 0., 1., 1.]
    ])
    # Matrix A1
    model.lins[0].weight.data = torch.tensor([
        [1., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])
    # Matrix A2
    model.lins[1].weight.data = torch.tensor([
        [1., 1., 0., 0.],
        [0., 0., 0., 0.]
    ])
    # Bias b1
    model.lins[0].bias.data = torch.tensor([0.,0.,-1.,1.])
    # Bias b2
    model.lins[1].bias.data = torch.tensor([-2.,-1.])
    return model

def example_cdgraph_1():
    # R(b,a), R(c,a), S(b,a), A(a), A(b), B(b), B(c)
    features = torch.tensor([
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
    ], dtype=torch.float)
    edges = torch.tensor([
     [1,2,1],
     [0,0,0]
    ], dtype=torch.long)
    edge_colours = torch.tensor([0,0,1], dtype=torch.long)
    node_names = ["a","b","c"]
    cdgraph = CDGraph(col_size=2,delta=2,features=features,edges=edges,edge_colours=edge_colours,node_names=node_names)
    assert cdgraph.node_names_to_indices["a"] == 0
    assert cdgraph.node_names_to_indices["b"] == 1
    assert cdgraph.node_names_to_indices["c"] == 2
    return cdgraph


def ex_1_1_activation_0():
    return torch.tensor([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=torch.float)

def ex_1_1_activation_1():
    return torch.tensor([
        [1., 1., 1., 1.],
        [1., 0., 0., 1.],
        [0., 0., 0., 1.]
    ], dtype=torch.float)

def ex_1_1_activation_2():
    return torch.sigmoid(torch.tensor([
        [-9,-10],
        [-11,-11],
        [-12,-11]
    ], dtype=torch.float)) # Sorry for magic numbers. This just matches the GNN architecture AND the selected bias

# Assuming here a canonical encoding with signature A, B (unary), R, S (binary)
def make_mock_fe_1():
    trace = TraceCollector()
    external_encoder = IdentityEncoderDecoder(load_from_document=None, unary_predicates=["A","B"],
                                              binary_predicates=["R","S"])
    internal_encoder = CanonicalEncoderDecoder(load_from_document=None,unary_predicates=["A","B"],
                                               binary_predicates=["R","S"])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = example_model_1(device)
    cd_graph = example_cdgraph_1()
    threshold = 0.000123 # sigmoid of 1-10 (GNN architecture does this -10)
    apply_model(cd_graph, device, model, trace)
    # Check the activations match what we expect
    assert torch.equal(trace.fl0,ex_1_1_activation_0())
    assert torch.equal(trace.fl1,ex_1_1_activation_1())
    assert torch.equal(trace.fl2,ex_1_1_activation_2())
    fe = FactExplainer(device,model,threshold,trace,external_encoder,internal_encoder)
    return fe

def get_fact_ex1():
    return "a",TYPE_PRED, "A"


# ----------------------------------------------------------------------------------------------------------
# EXAMPLE 2 - ICLR22 ENCODING
# ----------------------------------------------------------------------------------------------------------

# Assume a signature with A, (unary), R, S (binary)
# Assume the iclr22 encoding so (\col,\delta) = (4,3)
# Layer_1:
# Feature 1: is an instance of A c1 connected to a pair instance of S
# Feature 2: is a pair instance of predicate S
# Feature 3: nothing
# Feature 4: nothing
# Feature 5: nothing
# Feature 6: nothing
# Layer 2:
# Feature 1: nothing
# Feature 2: current node is c3-connected to one with feature 2 and is c1-connected to a node with feature 1
# Feature 3: nothing
def example_model_2(device):
    model = GNN(
        dimensions=[3,6,3],
        num_edge_colours=4,
        aggregations=["max","max"]).to(device)
    # Checks
    assert model.num_layers == 2
    assert model.num_colours == 4
    assert model.dimensions == [3,6,3]
    # Matrix B1-c1
    model.convs[0].weights.data[0] = torch.tensor([
    [0., 0., 1.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.]
    ])
    # Matrix B1-c2
    model.convs[0].weights.data[1] = torch.tensor([
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.]
    ])
    # Matrix B1-c3
    model.convs[0].weights.data[2] = torch.tensor([
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]
    ])
    # Matrix B1-c4
    model.convs[0].weights.data[3] = torch.tensor([
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]
    ])
    # Matrix B2-c1
    model.convs[1].weights.data[0] = torch.tensor([
        [0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]
    ])
    # Matrix B2-c2
    model.convs[1].weights.data[1] = torch.tensor([
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]
    ])
    # Matrix B2-c3
    model.convs[1].weights.data[2] = torch.tensor([
        [0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]
    ])
    # Matrix B2-c4
    model.convs[1].weights.data[3] = torch.tensor([
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]
    ])
    # Matrix A1
    model.lins[0].weight.data = torch.tensor([
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]
    ])
    # Matrix A2
    model.lins[1].weight.data = torch.tensor([
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]
    ])
    # Bias b1
    model.lins[0].bias.data = torch.tensor([-1,0.,0.,0.,0.,0.])
    # Bias b2
    model.lins[1].bias.data = torch.tensor([0.,-1.,0.])
    return model

def example_cdgraph_2():
    # nodes: a, term-for-a-b, b, term-for-b-a  (order determined by careful examination of iclr22 encoder and canonical,
    # assuming input order of facts is A(a), S(a,b), and S(b,a)
    # A(a), unary-for-S(term-for-a-b), unary-for-S(term-for-b-a)
    features = torch.tensor([
    # A, R, S
    [1., 0., 0.], # a
    [0., 0., 1.], # ab
    [0., 0., 0.], # b
    [0., 0., 1.] # ba
    ], dtype=torch.float)
    edges = torch.tensor([
     [0,1,2,3,1,2,0,3,1,3,0,2],
     [1,0,3,2,2,1,3,0,3,1,2,0]
    ], dtype=torch.long)
    edge_colours = torch.tensor([0,0,0,0,1,1,1,1,2,2,3,3], dtype=torch.long)
    node_names = ["a","term-for-a-b","b","term-for-b-a"]
    cdgraph = CDGraph(col_size=4,delta=3,features=features,edges=edges,edge_colours=edge_colours,node_names=node_names)
    assert cdgraph.node_names_to_indices["a"] == 0
    assert cdgraph.node_names_to_indices["term-for-a-b"] == 1
    assert cdgraph.node_names_to_indices["b"] == 2
    assert cdgraph.node_names_to_indices["term-for-b-a"] == 3
    return cdgraph


def ex_2_2_activation_0():
    return torch.tensor([
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 0., 0.],
        [0., 0., 1.]
    ], dtype=torch.float)

def ex_2_2_activation_1():
    return torch.tensor([
        [1., 0., 0., 0.,0., 0.],
        [0., 1., 0., 0.,0., 0.],
        [0., 0., 0., 0.,0., 0.],
        [0., 1., 0., 0., 0., 0.]
    ], dtype=torch.float)

def ex_2_2_activation_2():
    return torch.sigmoid(torch.tensor([
        [-10,-11,-10],
        [-10,-9,-10],
        [-10,-11,-10],
        [-10, -10, -10]
    ], dtype=torch.float)) # Sorry for magic numbers. This just matches the GNN architecture AND the selected bias

# Assuming here a canonical encoding with signature A, B (unary), R, S (binary)
def make_mock_fe_2():
    trace = TraceCollector()
    external_encoder = ICLREncoderDecoder(load_from_document=None, unary_predicates=["A"],
                                              binary_predicates=["R","S"])
    internal_encoder = CanonicalEncoderDecoder(load_from_document=None,
                                               unary_predicates=["A","unary-for-R", "unary-for-S"],
                                               binary_predicates=["binary-pred-1",
                                                                  "binary-pred-2",
                                                                  "binary-pred-3",
                                                                  "binary-pred-4"])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = example_model_2(device)
    cd_graph = example_cdgraph_2()
    threshold = 0.000123 # sigmoid of 1-10 (GNN architecture does this -10)
    apply_model(cd_graph, device, model, trace)
    # Check the activations match what we expect
    assert torch.equal(trace.fl0,ex_2_2_activation_0())
    assert torch.equal(trace.fl1,ex_2_2_activation_1())
    assert torch.equal(trace.fl2,ex_2_2_activation_2())
    fe = FactExplainer(device,model,threshold,trace,external_encoder,internal_encoder)
    return fe

def get_fact_ex2():
    return "a", "R", "b"


class TestFactContext:

    def test_fact_context_ex1(self):
        fe = make_mock_fe_1()
        fact = get_fact_ex1()
        fact_context = FactContext(fact, fe.external_encoder, fe.internal_encoder, fe.node_to_index)
        assert fact_context.cd_ent1 == "a"
        assert fact_context.cd_ent3 == "A"
        assert fact_context.cd_fact_const_index == 0
        assert fact_context.cd_fact_pred_pos == 0

    def test_fact_context_ex2(self):
        fe = make_mock_fe_2()
        fact = get_fact_ex2()
        fact_context = FactContext(fact, fe.external_encoder, fe.internal_encoder, fe.node_to_index)
        assert fact_context.cd_ent1 == "term-for-a-b"
        assert fact_context.cd_ent3 == "unary-for-R"
        assert fact_context.cd_fact_const_index == 1
        assert fact_context.cd_fact_pred_pos == 1


class TestFactExplainer:

    def test_initialisation_ex1(self):
        fe = make_mock_fe_1()
        assert fe.node_to_index == {"a": 0, "b": 1, "c": 2}

    def test_basic_explanation_ex1(self):
        fe = make_mock_fe_1()
        fc = FactContext(get_fact_ex1(), fe.external_encoder, fe.internal_encoder, fe.node_to_index)
        conjunction, var_const_idx, var_layer_mask = fe.get_basic_explanation(fc)
        assert isinstance(conjunction, TreeShapedConjunction)
        assert len(conjunction) == 3
        vx = conjunction.root_node
        assert isinstance(vx, Variable)
        assert vx.features == BitSet.from_subset(dimension=2, subset={0})
        assert vx.level == 2
        assert (2, 1, 0) in vx.children  # In layer 2, we introduce a child variable via S (1) for position 0
        assert (1, 0, 0) in vx.children  # In layer 1, we introduce a child variable via R (0) for position 0
        vy = vx.children[(2, 1, 0)]
        assert isinstance(vy, Variable)
        assert vy.features == BitSet.from_subset(dimension=2, subset={0})
        assert vy.level == 1
        assert vy.children == {}
        vz = vx.children[(1, 0, 0)]
        assert isinstance(vz, Variable)
        assert vz.features == BitSet.from_subset(dimension=2, subset={0})
        assert vz.level == 0
        assert vz.children == {}

        # Verify the mapping of variables to constants (the paper's \nu)
        assert vx in var_const_idx
        assert var_const_idx[vx] == 0
        assert vy in var_const_idx
        assert var_const_idx[vy] == 1
        assert vz in var_const_idx
        assert var_const_idx[vz] == 1

        # Verify the (var,layer) to mask mapping (the paper's \mu)
        # These expected results were done by hand.
        assert var_layer_mask[(vx, 2)] == BitSet.from_subset(2, {0})
        assert var_layer_mask[(vx, 1)] == BitSet.from_subset(4, {0, 1})
        assert var_layer_mask[(vx, 0)] == BitSet.from_subset(2, {0})
        assert (vy, 2) not in var_layer_mask
        assert var_layer_mask[(vy, 1)] == BitSet.from_subset(4, {0})
        assert var_layer_mask[(vy, 0)] == BitSet.from_subset(2, {0})
        assert (vz, 2) not in var_layer_mask
        assert (vz, 1) not in var_layer_mask
        assert var_layer_mask[(vz, 0)] == BitSet.from_subset(2, {0})

    def test_fact_explainer_ex1(self):
        fe = make_mock_fe_1()
        rule = fe.explain_fact(get_fact_ex1())
        head, body = rule.rstrip(' .\n').split(' :- ')
        body_atoms = set(body.split(', '))
        assert head ==  "<A>[?X0]"
        assert body_atoms == {"<A>[?X0]", "<S>[?X1,?X0]", "<A>[?X1]", "<R>[?X2,?X0]", "<A>[?X2]"}


    def test_initialisation_ex2(self):
        fe = make_mock_fe_2()
        assert fe.node_to_index == {"a": 0, "term-for-a-b":1, "b":2, "term-for-b-a":3}

    def test_fact_basic_explanation_ex2(self):
        # The conjunction should have 4 variables: x0 (for ab), x1 (for a), x2 (for ba), x3 (for az)
        fe = make_mock_fe_2()
        fc = FactContext(get_fact_ex2(), fe.external_encoder, fe.internal_encoder, fe.node_to_index)
        conjunction, var_const_idx, var_layer_mask = fe.get_basic_explanation(fc)
        assert isinstance(conjunction, TreeShapedConjunction)
        assert len(conjunction) == 4
        vx = conjunction.root_node  # represents ab
        assert isinstance(vx, Variable)
        assert vx.features == BitSet.from_subset(dimension=3, subset={})
        assert vx.level == 2
        assert (2, 0, 0) in vx.children  # In layer 2, we introduce a child variable via c1 (0) for position 0
        vy = vx.children[(2, 0, 0)]  # represents a
        assert isinstance(vy, Variable)
        assert vy.features == BitSet.from_subset(dimension=3, subset={0})
        assert vy.level == 1
        assert (1, 0, 2) in vy.children  # In layer 1, we introduce a child variable via c1 (0) for position 2
        vz = vy.children[(1, 0, 2)]  # represents az
        assert isinstance(vz, Variable)
        assert vz.features == BitSet.from_subset(dimension=3, subset={2})
        assert vz.level == 0
        assert vz.children == {}
        assert (2, 2, 1) in vx.children  # In layer 2, we introduce a child variable via c3 (2) for position 1
        vt = vx.children[(2, 2, 1)]  # represents ba
        assert isinstance(vt, Variable)
        assert vt.features == BitSet.from_subset(dimension=3, subset={2})
        assert vt.level == 1
        assert vt.children == {}

        # Verify the mapping of variables to constants (the paper's \nu)
        # Recall constant order is a, ab, b, ba
        assert vx in var_const_idx
        assert var_const_idx[vx] == 1  # ab
        assert vy in var_const_idx
        assert var_const_idx[vy] == 0  # a
        assert vz in var_const_idx
        assert var_const_idx[vz] == 1  # ab
        assert vt in var_const_idx
        assert var_const_idx[vt] == 3  # ba

        # Verify the (var,layer) to mask mapping (the paper's \mu)
        # These expected results were done by hand.
        assert var_layer_mask[(vx, 2)] == BitSet.from_subset(3, {1})
        assert var_layer_mask[(vx, 1)] == BitSet.from_subset(6, {})
        assert var_layer_mask[(vx, 0)] == BitSet.from_subset(3, {})
        assert (vy, 2) not in var_layer_mask
        assert var_layer_mask[(vy, 1)] == BitSet.from_subset(6, {0})
        assert var_layer_mask[(vy, 0)] == BitSet.from_subset(3, {0})
        assert (vz, 2) not in var_layer_mask
        assert (vz, 1) not in var_layer_mask
        assert var_layer_mask[(vz, 0)] == BitSet.from_subset(3, {2})
        assert (vt, 2) not in var_layer_mask
        assert var_layer_mask[(vt, 1)] == BitSet.from_subset(6, {1})
        assert var_layer_mask[(vt, 0)] == BitSet.from_subset(3, {2})

    def test_fact_explainer_ex2(self):
        fe = make_mock_fe_2()
        rule = fe.explain_fact(get_fact_ex2())
        head, body = rule.rstrip(' .\n').split(' :- ')
        body_atoms = set(body.split(', '))
        assert head ==  "<R>[?X0,?X1]"
        assert body_atoms == {"<S>[?X1,?X0]", "<A>[?X0]", "<S>[?X0,?X2]"}
