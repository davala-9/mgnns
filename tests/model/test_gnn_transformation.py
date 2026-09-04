import torch

from src.config.config import AggregationType
from src.model.cd_graph import CDGraph, TraceCollector
from src.model.gnn_architectures import GNN
from src.model.gnn_transformation import (
    apply_nc_encoder, apply_c_encoder, apply_model, apply_c_decoder,
    apply_nc_decoder, apply_gnn_transformation,
)


def make_cd_graph():
    torch.manual_seed(0)
    return CDGraph(
        col_size=3, delta=4,
        features=torch.rand(3, 4),
        edges=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_colours=torch.tensor([0, 1], dtype=torch.long),
        node_names=["a", "b", "c"],
    )

def make_model():
    torch.manual_seed(0)
    return GNN(feature_dimension=4, num_edge_colours=3,
               aggregation_1=AggregationType.MAX, aggregation_2=AggregationType.SUM)


# --- simple delegation wrappers ---

def test_apply_nc_encoder_delegates_to_external_encoder():
    calls = []
    class FakeExternalEncoder:
        def encode_dataset(self, dataset):
            calls.append(dataset)
            return {("cd1", "p", "cd2")}
    result = apply_nc_encoder({("a", "p", "b")}, FakeExternalEncoder())
    assert result == {("cd1", "p", "cd2")}
    assert calls == [{("a", "p", "b")}]

def test_apply_c_encoder_delegates_to_internal_encoder():
    calls = []
    class FakeInternalEncoder:
        def encode_dataset(self, cd_dataset):
            calls.append(cd_dataset)
            return "a-cd-graph"
    result = apply_c_encoder({("cd1", "p", "cd2")}, FakeInternalEncoder())
    assert result == "a-cd-graph"
    assert calls == [{("cd1", "p", "cd2")}]

def test_apply_c_decoder_delegates_to_internal_encoder():
    calls = []
    class FakeInternalEncoder:
        def decode_graph(self, cd_graph, threshold):
            calls.append((cd_graph, threshold))
            return {("cd1", "p", "cd2"): 0.9}
    result = apply_c_decoder("a-cd-graph", 0.5, FakeInternalEncoder())
    assert result == {("cd1", "p", "cd2"): 0.9}
    assert calls == [("a-cd-graph", 0.5)]


# --- apply_model ---

def test_apply_model_returns_new_graph_with_transformed_features():
    cd_graph = make_cd_graph()
    output = apply_model(cd_graph, torch.device("cpu"), make_model())

    assert output.features.shape == cd_graph.features.shape
    assert output.features is not cd_graph.features
    assert not torch.equal(output.features, cd_graph.features)

def test_apply_model_preserves_graph_structure():
    cd_graph = make_cd_graph()
    output = apply_model(cd_graph, torch.device("cpu"), make_model())

    assert output.col_size == cd_graph.col_size
    assert output.delta == cd_graph.delta
    assert torch.equal(output.edges, cd_graph.edges)
    assert torch.equal(output.edge_colours, cd_graph.edge_colours)
    assert output.node_names == cd_graph.node_names

def test_apply_model_output_does_not_require_grad():
    cd_graph = make_cd_graph()
    output = apply_model(cd_graph, torch.device("cpu"), make_model())
    assert output.features.requires_grad is False

def test_apply_model_does_not_mutate_input_graph():
    cd_graph = make_cd_graph()
    original_features = cd_graph.features.clone()
    apply_model(cd_graph, torch.device("cpu"), make_model())
    assert torch.equal(cd_graph.features, original_features)

def test_apply_model_without_trace_collector():
    cd_graph = make_cd_graph()
    output = apply_model(cd_graph, torch.device("cpu"), make_model(), trace_collector=None)
    assert output is not None

def test_apply_model_populates_trace_collector():
    cd_graph = make_cd_graph()
    model = make_model()
    trace = TraceCollector()
    apply_model(cd_graph, torch.device("cpu"), model, trace_collector=trace)

    assert trace.cd_graph is cd_graph
    assert set(trace.activations.keys()) == {0, 1, 2}
    assert trace.activations[0].shape == cd_graph.features.shape
    assert trace.activations[1].shape == (3, model.dimensions[1])
    assert trace.activations[2].shape == cd_graph.features.shape
    assert all(not t.requires_grad for t in trace.activations.values())


# --- apply_nc_decoder ---

def test_apply_nc_decoder_translates_facts_and_keeps_scores():
    class FakeExternalEncoder:
        def decode_fact(self, s, p, o):
            return (f"data_{s}", p, o)

    cd_facts_scores = {
        ("c1", "p", "c2"): 0.7,
        ("c3", "p", "c4"): 0.3,
    }
    result = apply_nc_decoder(cd_facts_scores, FakeExternalEncoder())
    assert result == {
        ("data_c1", "p", "c2"): 0.7,
        ("data_c3", "p", "c4"): 0.3,
    }

def test_apply_nc_decoder_drops_facts_that_decode_to_none():
    class FakeExternalEncoder:
        def decode_fact(self, s, p, o):
            return None if s == "c1" else (s, p, o)

    cd_facts_scores = {
        ("c1", "p", "c2"): 0.9,
        ("c3", "p", "c4"): 0.5,
    }
    result = apply_nc_decoder(cd_facts_scores, FakeExternalEncoder())
    assert result == {("c3", "p", "c4"): 0.5}


# --- apply_gnn_transformation ---

def test_apply_gnn_transformation_pipes_data_through_all_steps():
    cd_graph = make_cd_graph()
    model = make_model()

    class FakeExternalEncoder:
        def __init__(self):
            self.encode_calls = []
        def encode_dataset(self, dataset):
            self.encode_calls.append(dataset)
            return {("cd1", "p", "cd2")}
        def decode_fact(self, s, p, o):
            return (f"orig_{s}", p, o)

    class FakeInternalEncoder:
        def __init__(self, cd_graph_to_return, decode_result):
            self.cd_graph_to_return = cd_graph_to_return
            self.decode_result = decode_result
            self.encode_calls = []
            self.decode_calls = []
        def encode_dataset(self, cd_dataset):
            self.encode_calls.append(cd_dataset)
            return self.cd_graph_to_return
        def decode_graph(self, output_cd_graph, threshold):
            self.decode_calls.append((output_cd_graph, threshold))
            return self.decode_result

    external_encoder = FakeExternalEncoder()
    internal_encoder = FakeInternalEncoder(cd_graph, decode_result={("cd1", "p", "cd2"): 0.8})

    input_dataset = {("a", "p", "b")}
    result = apply_gnn_transformation(input_dataset, external_encoder, internal_encoder, model,
                                      threshold=0.5, device=torch.device("cpu"))

    # step 1: external encoder receives the original dataset
    assert external_encoder.encode_calls == [input_dataset]
    # step 2: internal encoder's encode_dataset receives step 1's output
    assert internal_encoder.encode_calls == [{("cd1", "p", "cd2")}]
    # step 4: internal encoder's decode_graph receives the threshold and apply_model's output graph
    assert len(internal_encoder.decode_calls) == 1
    decoded_graph, threshold_used = internal_encoder.decode_calls[0]
    assert threshold_used == 0.5
    assert decoded_graph.node_names == cd_graph.node_names
    # step 5: final result is decoded back through the external encoder
    assert result == {("orig_cd1", "p", "cd2"): 0.8}

def test_apply_gnn_transformation_forwards_trace_collector_to_apply_model():
    cd_graph = make_cd_graph()
    model = make_model()

    class FakeExternalEncoder:
        def encode_dataset(self, dataset):
            return {("cd1", "p", "cd2")}
        def decode_fact(self, s, p, o):
            return (s, p, o)

    class FakeInternalEncoder:
        def encode_dataset(self, cd_dataset):
            return cd_graph
        def decode_graph(self, output_cd_graph, threshold):
            return {}

    trace = TraceCollector()
    apply_gnn_transformation({("a", "p", "b")}, FakeExternalEncoder(), FakeInternalEncoder(), model,
                             threshold=0.5, device=torch.device("cpu"), trace_collector=trace)
    assert trace.cd_graph is cd_graph
    assert set(trace.activations.keys()) == {0, 1, 2}
