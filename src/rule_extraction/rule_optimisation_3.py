from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, CompactSubTree
from typing import Protocol
from collections import deque
import time

# Try to implement in a way that does not allow duplicates, please
class Frontier(Protocol):
    def push(self, item: CompactSubTree) -> None: ...
    def pop(self) -> CompactSubTree: ...
    def is_empty(self) -> bool: ...

class DFSFrontier:
    def __init__(self):
        self._stack: list[CompactSubTree] = []
    def push(self, item: CompactSubTree) -> None:
        self._stack.append(item)
    def pop(self) -> CompactSubTree:
        return self._stack.pop()
    def is_empty(self) -> bool:
        return len(self._stack) == 0

class BFSFrontier:
    def __init__(self):
        self._queue: deque[CompactSubTree] = deque()
    def push(self, item: CompactSubTree) -> None:
        self._queue.append(item)
    def pop(self) -> CompactSubTree:
        return self._queue.popleft()
    def is_empty(self) -> bool:
        return len(self._queue) == 0

class RuleOptimisation3:

    def __init__(self, device, model, threshold, pred_position, base_tree:TreeShapedConjunction):
        self.device = device
        self.model = model
        self.threshold = threshold
        self.pred_position = pred_position
        self.base_tree = base_tree

    def graph_search(self, frontier: Frontier, timeout: float | None = None):
        start = time.monotonic()
        frontier.push(self.base_tree.initial_subtree)
        explored = set()
        while not frontier.is_empty():
            if timeout is not None and time.monotonic() - start > timeout:
                return None
            subtree = frontier.pop()
            if subtree in explored:
                continue
            if subtree.check_soundness(self.device,self.model,self.threshold,self.pred_position):
                return subtree
            explored.add(subtree)
            for successor in subtree.get_successors():
                frontier.push(successor)
        return None

    # Returns a minimal compact subtree
    def minimise_rule(self):
        result = self.graph_search(BFSFrontier(),timeout=600)
        if result is None:
            print("BFS rule simplification timed out, trying DFS...")
            result = self.graph_search(DFSFrontier())
        return result
