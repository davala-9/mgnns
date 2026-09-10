import torch
import pytest

from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.identity import IdentityEncoderDecoder
from src.model.cd_graph import TraceCollector, CDGraph
from src.model.gnn_architectures import GNN
from src.model.gnn_transformation import apply_model
from src.rule_extraction.fact_explanation import FactExplainer, FactContext
from src.rule_extraction.rule_optimisation_2 import (
    apply_optimisation, compute_path_weights, path_weight_score_fn, unrestricted_var_layer_mask,
)
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, TreeShapedConjunctionBuilder
from src.utils.bitset import BitSet
from src.utils.utils import TYPE_PRED

# ----------------------------------------------------------------------------------------------------------
# Fixtures below mirror EXAMPLE 1 from test_fact_explanation.py / test_rule_optimisation_1.py (same
# model/fact), kept self-contained here rather than cross-importing between test modules.
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
# compute_path_weights: hand-verified against the model's actual matrices
# ----------------------------------------------------------------------------------------------------------

def test_compute_path_weights_matches_hand_derivation():
    fe = make_mock_fe_1()
    fact_context, rule_body, _, var_layer_mask = get_basic_explanation_ex1(fe)

    # Structure (per test_basic_explanation_ex1): root=0 (level 2), vy=1 (level 1, edge (2,1,0) from root),
    # vz=2 (level 0, edge (1,0,0) from root). All three have a single relevant position: 0.
    weights = compute_path_weights(fe.model, rule_body, var_layer_mask, fact_context.cd_fact_pred_pos)

    # Root: one-hot at predicate_position=0, propagated through matrix_A(2) then matrix_A(1).
    # level2->1: [1,0] @ lin_self_2.weight -> row 0 = [1,1,0,0]
    # level1->0: [1,1,0,0] @ lin_self_1.weight -> row0 + row1 = [1,0] + [0,0] = [1,0]
    assert weights[(0, 0)] == pytest.approx(1.0)

    # vy: edge (2, colour=1, pos=0) from root. edge_weight = root_vec_at_L[0] * matrix_B(2,1)[0][0] = 1*1 = 1
    # no further self-hop needed at pos 0 beyond the edge itself (level 1 -> already the atom's own level... )
    assert weights[(1, 0)] == pytest.approx(1.0)

    # vz: edge (1, colour=0, pos=0) from root, using root's OWN propagated vector AT LAYER 1 ([1,1,0,0]),
    # summed (weighted) over var_layer_mask[(0,1)] = {0,1}:
    # edge_weight = 1*matrix_B(1,0)[0][0] + 1*matrix_B(1,0)[1][0] = 1*0 + 1*1 = 1
    assert weights[(2, 0)] == pytest.approx(1.0)


def test_compute_path_weights_only_covers_atoms_present_in_rule_body():
    fe = make_mock_fe_1()
    fact_context, rule_body, _, var_layer_mask = get_basic_explanation_ex1(fe)
    weights = compute_path_weights(fe.model, rule_body, var_layer_mask, fact_context.cd_fact_pred_pos)
    expected_atoms = {
        (var_id, pos)
        for var_id in range(len(rule_body))
        for pos in rule_body.features[var_id].elements()
    }
    assert set(weights.keys()) == expected_atoms


# ----------------------------------------------------------------------------------------------------------
# path_weight_score_fn: a tiny hand-built tree, independent of any real model, to isolate the scoring
# logic itself (root -- var 0, universe {0,1} -- with one child, var 1, universe {0}).
# ----------------------------------------------------------------------------------------------------------

def make_tiny_base_tree():
    builder = TreeShapedConjunctionBuilder(n_colours=1)
    builder.add(features=BitSet.from_subset(2, {0, 1}), level=1, parent=-1)
    builder.add(features=BitSet.from_subset(1, {0}), level=0, parent=0, edge=(1, 0, 0))
    return builder.build()


# ----------------------------------------------------------------------------------------------------------
# unrestricted_var_layer_mask: no real model needed -- it only ever calls model.layer_dimension(l).
# ----------------------------------------------------------------------------------------------------------

class FakeModelWithLayerDimensions:
    def __init__(self, dimensions: dict[int, int]):
        self._dimensions = dimensions

    def layer_dimension(self, l):
        return self._dimensions[l]


def test_unrestricted_var_layer_mask_covers_exactly_what_compute_path_weights_needs():
    base_tree = make_tiny_base_tree()  # root (var 0, level 1), one child (var 1, level 0)
    mask = unrestricted_var_layer_mask(FakeModelWithLayerDimensions({0: 2, 1: 3}), base_tree)
    # var 0 needs its own level down to 1: just {1}. var 1's own level is already 0, so it needs none
    # (compute_path_weights never looks up var_layer_mask for a var that starts out at layer 0).
    assert set(mask.keys()) == {(0, 1)}
    assert mask[(0, 1)].elements() == [0, 1, 2]  # every position at layer 1, unrestricted


def test_unrestricted_var_layer_mask_works_as_a_drop_in_for_a_real_model_and_tree():
    # This is what RuleOptimisation3.greedy_climb_frontier does when base_tree has no \mu of its own.
    fe = make_mock_fe_1()
    fact_context, rule_body, _, _ = get_basic_explanation_ex1(fe)
    mask = unrestricted_var_layer_mask(fe.model, rule_body)
    weights = compute_path_weights(fe.model, rule_body, mask, fact_context.cd_fact_pred_pos)
    expected_atoms = {
        (var_id, pos) for var_id in range(len(rule_body)) for pos in rule_body.features[var_id].elements()
    }
    assert set(weights.keys()) == expected_atoms


def test_path_weight_score_fn_root_scores_zero():
    base_tree = make_tiny_base_tree()
    score = path_weight_score_fn(base_tree, weights={(0, 0): 3.0, (0, 1): 5.0, (1, 0): 7.0})
    assert score(base_tree.initial_compact) == 0.0


def test_path_weight_score_fn_adds_up_weights_of_set_bits_along_a_path():
    base_tree = make_tiny_base_tree()
    weights = {(0, 0): 3.0, (0, 1): 5.0, (1, 0): 7.0}
    score = path_weight_score_fn(base_tree, weights)

    root = base_tree.initial_compact
    turn_on_var0_pos0 = next(iter(root.get_successors(base_tree)))  # var 0's first relevant position
    assert score(turn_on_var0_pos0) == pytest.approx(3.0)  # 0 (root) + weight of the atom just turned on

    turn_on_var0_pos1 = next(s for s in turn_on_var0_pos0.get_successors(base_tree) if len(s.var_ids) == 1)
    assert score(turn_on_var0_pos1) == pytest.approx(8.0)  # 3 (parent) + 5 (second atom)


def test_path_weight_score_fn_a_freshly_added_child_scores_like_its_parent():
    base_tree = make_tiny_base_tree()
    weights = {(0, 0): 3.0, (0, 1): 5.0, (1, 0): 7.0}
    score = path_weight_score_fn(base_tree, weights)

    root = base_tree.initial_compact
    add_child = next(s for s in root.get_successors(base_tree) if len(s.var_ids) == 2)
    assert score(add_child) == 0.0  # the new child's mask is empty -- nothing to weigh in yet

    # The successor where the CHILD's (var 1's) mask changes, as opposed to var 0's.
    turn_on_child_pos0 = next(s for s in add_child.get_successors(base_tree) if s.masks[1] != add_child.masks[1])
    assert score(turn_on_child_pos0) == pytest.approx(7.0)  # 0 (add_child) + weight of the child's atom


def test_path_weight_score_fn_requires_a_scored_predecessor():
    base_tree = make_tiny_base_tree()
    score = path_weight_score_fn(base_tree, weights={})
    add_child = next(s for s in base_tree.initial_compact.get_successors(base_tree) if len(s.var_ids) == 2)
    # add_child's own score is never requested, so its only predecessor (add_child itself) isn't cached.
    orphan = next(add_child.get_successors(base_tree))
    with pytest.raises(AssertionError):
        score(orphan)


# ----------------------------------------------------------------------------------------------------------
# apply_optimisation: end-to-end
# ----------------------------------------------------------------------------------------------------------

def test_apply_optimisation_returns_a_sound_smaller_or_equal_rule():
    fe = make_mock_fe_1()
    fact_context, rule_body, _, var_layer_mask = get_basic_explanation_ex1(fe)

    result = apply_optimisation(fe, rule_body, fact_context.cd_fact_pred_pos, var_layer_mask)

    assert isinstance(result, TreeShapedConjunction)
    assert len(result) <= len(rule_body)

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
    # Same rationale as the equivalent test in test_rule_optimisation_1.py: apply_optimisation assumes
    # its input rule_body is already sound. Root alone (without the two supporting variables the real
    # derivation needs) is not sound, so the greedy search must exhaust every candidate atom without
    # ever crossing the threshold, hitting the function's own sanity-check AssertionError.
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


def test_apply_optimisation_rejects_models_that_are_not_2_layer_relu():
    fe = make_mock_fe_1()
    fact_context, rule_body, _, var_layer_mask = get_basic_explanation_ex1(fe)
    fe.model.num_layers = 3  # simulate an unsupported architecture

    with pytest.raises(AssertionError):
        apply_optimisation(fe, rule_body, fact_context.cd_fact_pred_pos, var_layer_mask)
