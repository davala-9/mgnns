import pytest
import torch
from torch_geometric.data import Data

from src.config.config import AggregationType
from src.model.gnn_architectures import EC_GCNConv, GNN


# --- EC_GCNConv ---

@pytest.mark.parametrize("aggregation,expected", [
    ("max", "max"),
    ("sum", "sum"),
    (AggregationType.MAX, "max"),
    (AggregationType.SUM, "sum"),
])
def test_ec_gcnconv_constructs_with_valid_aggregation(aggregation, expected):
    conv = EC_GCNConv(in_channels=4, out_channels=6, edge_colours=3, aggregation=aggregation)
    assert conv.aggr == expected

def test_ec_gcnconv_weights_shape():
    conv = EC_GCNConv(in_channels=4, out_channels=6, edge_colours=3, aggregation="max")
    assert conv.weights.shape == (3, 6, 4)

def test_ec_gcnconv_rejects_unsupported_aggregation():
    with pytest.raises(ValueError, match="Unsupported aggregation mode"):
        EC_GCNConv(in_channels=4, out_channels=6, edge_colours=3, aggregation="mean")


# --- GNN construction ---

def make_model(feature_dimension=4, num_edge_colours=3):
    return GNN(feature_dimension=feature_dimension, num_edge_colours=num_edge_colours,
               aggregation_1=AggregationType.MAX, aggregation_2=AggregationType.SUM)

def test_gnn_dimensions():
    model = make_model(feature_dimension=4)
    assert model.dimensions == [4, 8, 4]

def test_gnn_num_layers():
    model = make_model()
    assert model.num_layers == 2

def test_gnn_conv_layers_wired_to_correct_dimensions():
    model = make_model(feature_dimension=4, num_edge_colours=5)
    assert model.conv1.weights.shape == (5, 8, 4)
    assert model.conv2.weights.shape == (5, 4, 8)


# --- forward ---

def make_data(x, edge_index, edge_type):
    return Data(x=x, edge_index=edge_index, edge_type=edge_type)

def test_forward_output_shapes():
    torch.manual_seed(0)
    model = make_model(feature_dimension=4, num_edge_colours=3)
    x = torch.rand(3, 4)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_type = torch.tensor([0, 1], dtype=torch.long)
    out, feat1 = model(make_data(x, edge_index, edge_type))
    assert out.shape == (3, 4)
    assert feat1.shape == (3, 8)

def test_forward_output_in_unit_range():
    torch.manual_seed(0)
    model = make_model()
    x = torch.rand(3, 4)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_type = torch.tensor([0, 1], dtype=torch.long)
    out, _ = model(make_data(x, edge_index, edge_type))
    assert torch.all(out >= 0) and torch.all(out <= 1)

def test_forward_handles_no_edges_without_nan():
    torch.manual_seed(0)
    model = make_model()
    x = torch.rand(3, 4)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_type = torch.empty((0,), dtype=torch.long)
    out, feat1 = model(make_data(x, edge_index, edge_type))
    assert not torch.isnan(out).any()
    assert not torch.isnan(feat1).any()

def test_forward_handles_missing_edge_colours_without_nan():
    # Edges only use colour 0; colours 1 and 2 have no edges at all
    torch.manual_seed(0)
    model = make_model(feature_dimension=4, num_edge_colours=3)
    x = torch.rand(3, 4)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_type = torch.tensor([0, 0], dtype=torch.long)
    out, feat1 = model(make_data(x, edge_index, edge_type))
    assert not torch.isnan(out).any()
    assert not torch.isnan(feat1).any()


# --- accessor methods ---

def test_layer_dimension():
    model = make_model(feature_dimension=4)
    assert model.layer_dimension(0) == 4
    assert model.layer_dimension(1) == 8
    assert model.layer_dimension(2) == 4

def test_matrix_A_matches_linear_layers():
    model = make_model()
    assert torch.equal(model.matrix_A(1), model.lin_self_1.weight.detach())
    assert torch.equal(model.matrix_A(2), model.lin_self_2.weight.detach())

def test_matrix_B_matches_conv_weights():
    model = make_model(num_edge_colours=3)
    assert torch.equal(model.matrix_B(1, 0), model.conv1.weights[0].detach())
    assert torch.equal(model.matrix_B(2, 2), model.conv2.weights[2].detach())

def test_bias_layer_1_matches_linear_bias():
    model = make_model()
    assert torch.equal(model.bias(1), model.lin_self_1.bias.detach())

def test_bias_layer_2_applies_the_forward_translation_constant():
    # forward() computes self.output(x - 10) for layer 2; bias(2) must apply the same -10
    # so anything reconstructing layer 2's affine map from bias(2) matches forward()'s behaviour.
    model = make_model()
    assert torch.equal(model.bias(2), model.lin_self_2.bias.detach() - 10)

def test_activation_functions():
    model = make_model()
    assert model.activation(1) is torch.relu
    assert isinstance(model.activation(2), torch.nn.Sigmoid)

def test_aggregation_function_returns_configured_aggregations():
    model = GNN(feature_dimension=4, num_edge_colours=3,
                aggregation_1=AggregationType.MAX, aggregation_2=AggregationType.SUM)
    assert model.aggregation_function(1) == AggregationType.MAX
    assert model.aggregation_function(2) == AggregationType.SUM

@pytest.mark.parametrize("accessor", ["matrix_A", "bias", "activation", "aggregation_function"])
@pytest.mark.parametrize("invalid_layer", [0, 3, -1])
def test_single_arg_accessors_reject_invalid_layer(accessor, invalid_layer):
    model = make_model()
    with pytest.raises(ValueError, match="invalid layer"):
        getattr(model, accessor)(invalid_layer)

@pytest.mark.parametrize("invalid_layer", [0, 3, -1])
def test_matrix_B_rejects_invalid_layer(invalid_layer):
    model = make_model()
    with pytest.raises(ValueError, match="invalid layer"):
        model.matrix_B(invalid_layer, 0)
