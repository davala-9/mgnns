import torch
import pytest

from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.identity import IdentityEncoderDecoder
from src.model.cd_graph import TraceCollector, CDGraph
from src.model.gnn_architectures import GNN
from src.model.gnn_transformation import apply_model
from src.rule_extraction.fact_explanation import FactExplainer, FactContext
from src.rule_extraction.rule_optimisation_1 import apply_optimisation
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction
from src.utils.bitset import BitSet
from src.utils.utils import TYPE_PRED

# ----------------------------------------------------------------------------------------------------------
# Fixtures below mirror EXAMPLE 1 from test_fact_explanation.py (same model/fact), kept self-contained here
# rather than cross-importing between test modules.
# ----------------------------------------------------------------------------------------------------------

def example_model_1(device):
    model = GNN(feature_dimension=2, num_edge_colours=2, aggregation_1="max", aggregation_2="max").to(device)
    # Matrix B1-c1
    model.conv1.weights.data[0] = torch.tensor([
        [0., 0.],
        [1., 0.],
        [0., 1.],
        [0., 0.]
    ])
    # Matrix B1-c2
    model.conv1.weights.data[1] = torch.tensor([
        [0., 0.],
        [0., 0.],
        [0., 1.],
        [0., 0.]
    ])
    # Matrix B2-c1
    model.conv2.weights.data[0] = torch.tensor([
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]
    ])
    # Matrix B2-c2
    model.conv2.weights.data[1] = torch.tensor([
        [1., 0., 0., 0.],
        [0., 0., 1., 1.]
    ])
    # Matrix A1
    model.lin_self_1.weight.data = torch.tensor([
        [1., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])
    # Matrix A2
    model.lin_self_2.weight.data = torch.tensor([
        [1., 1., 0., 0.],
        [0., 0., 0., 0.]
    ])
    model.lin_self_1.bias.data = torch.tensor([0., 0., -1., 1.])
    model.lin_self_2.bias.data = torch.tensor([-2., -1.])
    return model


def example_cdgraph_1():
    # R(b,a), R(c,a), S(b,a), A(a), A(b), B(b), B(c)
    features = torch.tensor([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=torch.float)
    edges = torch.tensor([
        [1, 2, 1],
        [0, 0, 0]
    ], dtype=torch.long)
    edge_colours = torch.tensor([0, 0, 1], dtype=torch.long)
    node_names = ["a", "b", "c"]
    return CDGraph(col_size=2, delta=2, features=features, edges=edges,
                   edge_colours=edge_colours, node_names=node_names)


def make_mock_fe_1():
    trace = TraceCollector()
    external_encoder = IdentityEncoderDecoder(load_from_document=None, unary_predicates=["A", "B"],
                                               binary_predicates=["R", "S"])
    internal_encoder = CanonicalEncoderDecoder(load_from_document=None, unary_predicates=["A", "B"],
                                                binary_predicates=["R", "S"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = example_model_1(device)
    cd_graph = example_cdgraph_1()
    threshold = 0.000123  # sigmoid of 1-10 (GNN architecture does this -10)
    apply_model(cd_graph, device, model, trace)
    return FactExplainer(device, model, threshold, trace, external_encoder, internal_encoder)


def get_fact_ex1():
    return "a", TYPE_PRED, "A"


def get_basic_explanation_ex1(fe):
    fact_context = FactContext(get_fact_ex1(), fe.external_encoder, fe.internal_encoder, fe.constant_to_index)
    rule_body, var_const_idx, var_layer_mask = fe.get_basic_explanation(fact_context)
    return fact_context, rule_body, var_const_idx, var_layer_mask


# ----------------------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------------------

def test_apply_optimisation_returns_a_sound_smaller_or_equal_rule():
    fe = make_mock_fe_1()
    fact_context, rule_body, _, var_layer_mask = get_basic_explanation_ex1(fe)

    result = apply_optimisation(fe, rule_body, fact_context.cd_fact_pred_pos, var_layer_mask)

    assert isinstance(result, TreeShapedConjunction)
    assert len(result) <= len(rule_body)

    # The optimised rule must still derive the head fact above threshold when grounded on its own
    output_graph = apply_model(result.as_cd_graph, fe.device, fe.model)
    assert output_graph.features[0][fact_context.cd_fact_pred_pos] > fe.threshold


def test_apply_optimisation_is_deterministic():
    fe = make_mock_fe_1()
    fact_context, rule_body, _, var_layer_mask = get_basic_explanation_ex1(fe)

    result_1 = apply_optimisation(fe, rule_body, fact_context.cd_fact_pred_pos, var_layer_mask)
    result_2 = apply_optimisation(fe, rule_body, fact_context.cd_fact_pred_pos, var_layer_mask)

    assert result_1.features == result_2.features
    assert result_1.parent == result_2.parent


def test_apply_optimisation_raises_when_rule_body_is_insufficient():
    # apply_optimisation assumes its input rule_body is already a SOUND explanation, and only tries
    # to shrink it. Root alone (without the two supporting variables the real derivation needs, per
    # test_fact_explainer_ex1's 5-atom body) is not sound, so the greedy search must exhaust every
    # candidate atom without ever crossing the threshold, hitting the function's own sanity-check
    # AssertionError rather than silently returning something unsound.
    fe = make_mock_fe_1()
    fact_context, rule_body, _, var_layer_mask = get_basic_explanation_ex1(fe)

    root_only = TreeShapedConjunction(
        n_colours=rule_body.n_colours,
        features=(rule_body.features[0],),
        levels=(rule_body.levels[0],),
        children=({},),
        parent=(-1,),
    )
    trimmed_var_layer_mask = {k: v for k, v in var_layer_mask.items() if k[0] == 0}

    with pytest.raises(AssertionError):
        apply_optimisation(fe, root_only, fact_context.cd_fact_pred_pos, trimmed_var_layer_mask)
