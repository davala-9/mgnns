import time
from typing import Callable

from src.rule_extraction.lattice_search import (
    Frontier, SinglePathFrontier, GreedyBestSuccessorFrontier, DFSFrontier, BFSFrontier, FirstResultPolicy, search,
)
from src.rule_extraction.rule_optimisation_2 import compute_path_weights, path_weight_score_fn, \
    unrestricted_var_layer_mask
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction

# Takes a TreeShapedConjunction and attempts to find a minimal sound subtree, as another #TreeShapedConjunction.
# It spends some amount of time trying minimal extraction with BFS, then it gives up and uses greedy path climb.
class RuleOptimisation3:

    def __init__(self, device, model, threshold, pred_position, base_tree: TreeShapedConjunction):
        self.device = device
        self.model = model
        self.threshold = threshold
        self.pred_position = pred_position
        self.base_tree = base_tree

    def _check_soundness(self, node):
        return node.check_soundness(self.base_tree, self.device, self.model, self.threshold, self.pred_position)

    def graph_search(self, frontier: Frontier, timeout: float | None = None):
        deadline = time.monotonic() + timeout if timeout is not None else None
        return next(search(self.base_tree, self._check_soundness, frontier, FirstResultPolicy(), deadline), None)

    # A frontier_factory for minimise_rule()'s fallback stage: a greedy climb (see
    # GreedyBestSuccessorFrontier) guided by Optimisation 2's per-atom weights, instead of an arbitrary
    # path. base_tree has no \mu of its own here (it's already a reduced, renumbered subtree, not the
    # original explanation tree \mu was computed for), so we use an unrestricted one -- see
    # rule_optimisation_2.unrestricted_var_layer_mask.
    def greedy_climb_frontier(self):
        var_layer_mask = unrestricted_var_layer_mask(self.model, self.base_tree)
        weights = compute_path_weights(self.model, self.base_tree, var_layer_mask, self.pred_position)
        return GreedyBestSuccessorFrontier(path_weight_score_fn(self.base_tree, weights))

    # Returns a minimal compact subtree
    def minimise_rule(self, fallback_frontier_factory: Callable[[], Frontier] = SinglePathFrontier):
        result = self.graph_search(BFSFrontier(),timeout=120)
        if result is None:
            print("BFS rule simplification timed out, trying a greedy climb...")
            result = self.graph_search(fallback_frontier_factory())
        return result
