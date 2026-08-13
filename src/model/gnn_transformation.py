from inspect import trace
from src.model.cd_graph import CDGraph
from torch_geometric.data import Data

# This is the transformation T_M defined in the paper. It has the following steps:
# 1) Apply the external (non-canonical) encoder,
# 2) Apply the internal (canonical) encoder,
# 3) Apply the model
# 4) Apply the internal (canonical) decoder,
# 5) Apply the external (non-canonical) decoder.
# Each step is implemented separately to enable re-use of individual steps.

# Non-Canonical Encoding
def apply_nc_encoder(dataset: set[tuple[str,str,str]], external_encoder):
    # dataset in input signature -> dataset in cd-signature
    return external_encoder.encode_dataset(dataset)

# Canonical Encoding
def apply_c_encoder(cd_dataset: set[tuple[str,str,str]], internal_encoder):
    # dataset in cd-signature -> cd_graph
    return internal_encoder.encode_dataset(cd_dataset)

# Apply Model
def apply_model(cd_graph: CDGraph, device, model, trace_collector=None):
    # PyTorch Encoding: cd_graph -> pytorch geometric graph
    data = Data(x=cd_graph.features, edge_index=cd_graph.edges, edge_type=cd_graph.edge_colours).to(device)

    # Apply model
    model.eval()

    # Capture the final output tensor and the list of intermediate layer tensors
    final_features, hidden_features = model(data)

    if trace_collector is not None:
        trace_collector.cd_graph = cd_graph
        trace_collector.input_features = data.x.detach().clone()

        # Safely clone each tensor in the hidden features list
        trace_collector.hidden_features = [layer.detach().clone() for layer in hidden_features]

        trace_collector.final_features = final_features.detach().clone()

    # PyTorch Decoding: pytorch geometric graph -> cd_graph
    return CDGraph(cd_graph.col_size, cd_graph.delta, final_features.detach().clone(), cd_graph.edges,
                   cd_graph.edge_colours, cd_graph.node_names)

# Canonical Decoding
def apply_c_decoder(cd_graph, threshold, internal_encoder):
    # cd_graph -> [dataset in cd-signature: score]
    return internal_encoder.decode_graph(cd_graph, threshold)

# Non-Canonical Decoding
def apply_nc_decoder(cd_facts_scores_dict, external_encoder):
    # [cd_dataset: score] -> [dataset in input signature: score]
    facts_scores_dict = {}
    for (s, p, o), score in cd_facts_scores_dict.items():
        result = external_encoder.decode_fact(s, p, o)
        # TODO: this could be a set, of many or none
        if result is not None: # Some canonical facts dont turn into facts
            ss, pp, oo = result
            facts_scores_dict[(ss, pp, oo)] = cd_facts_scores_dict[(s, p, o)]
    return facts_scores_dict


def apply_gnn_transformation(dataset: set[tuple[str, str, str]], external_encoder, internal_encoder, model, threshold,
                             device, trace_collector=None):

    cd_dataset = apply_nc_encoder(dataset,external_encoder) # Step 1
    cd_graph = apply_c_encoder(cd_dataset,internal_encoder) # Step 2
    output_cd_graph = apply_model(cd_graph, device, model, trace_collector) # Step 3
    cd_dataset_facts_scores_dict = apply_c_decoder(output_cd_graph, threshold, internal_encoder) # Step 4
    dataset_facts_scores_dict =  apply_nc_decoder(cd_dataset_facts_scores_dict,external_encoder) # Step 5

    return dataset_facts_scores_dict


