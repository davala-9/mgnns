import time

from src.rule_extraction.lattice_search import (
    Frontier, SinglePathFrontier, DFSFrontier, BFSFrontier, FirstResultPolicy, search,
)
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

    # Returns a minimal compact subtree
    def minimise_rule(self):
        result = self.graph_search(BFSFrontier(),timeout=120)
        if result is None:
            print("BFS rule simplification timed out, trying SinglePathFrontier...")
            result = self.graph_search(SinglePathFrontier())
        return result
