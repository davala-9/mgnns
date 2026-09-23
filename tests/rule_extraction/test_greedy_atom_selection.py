import torch

from src.model.gnn_architectures import GNN
from src.rule_extraction.greedy_atom_selection import add_atoms_until_sound
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunctionBuilder
from src.utils.bitset import BitSet

DEVICE = torch.device("cpu")
THRESHOLD = 0.5


# One unary predicate (position 0), one colour. The model derives position 0 at a node iff that node itself
# has position 0 in its input: hidden = x, output = sigmoid(20 * hidden - 10). Neighbours are ignored.
def self_only_model():
    model = GNN(feature_dimension=1, num_edge_colours=1, aggregation_1="max", aggregation_2="max").to(DEVICE)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.lin_self_1.weight[0, 0] = 1.0
        model.lin_self_2.weight[0, 0] = 20.0
    return model


# Root (var 0) and one child (var 1), both with the single atom at position 0.
def root_and_child_tree():
    builder = TreeShapedConjunctionBuilder(n_colours=1)
    builder.add(features=BitSet.from_subset(1, {0}), level=2, parent=-1)
    builder.add(features=BitSet.from_subset(1, {0}), level=1, parent=0, edge=(2, 0, frozenset({0})))
    return builder.build()


def test_stops_at_the_first_sound_prefix():
    result = add_atoms_until_sound(root_and_child_tree(), [(0, 0), (1, 0)], DEVICE, self_only_model(), THRESHOLD, 0)
    assert len(result) == 1  # the child was never needed, so it was never added
    assert result.features[0] == BitSet.from_subset(1, {0})


def test_keeps_adding_until_sound():
    result = add_atoms_until_sound(root_and_child_tree(), [(1, 0), (0, 0)], DEVICE, self_only_model(), THRESHOLD, 0)
    assert len(result) == 2
    assert result.features[0] == BitSet.from_subset(1, {0})
    assert result.features[1] == BitSet.from_subset(1, {0})


def test_returns_none_if_no_prefix_is_sound():
    assert add_atoms_until_sound(root_and_child_tree(), [(1, 0)], DEVICE, self_only_model(), THRESHOLD, 0) is None
