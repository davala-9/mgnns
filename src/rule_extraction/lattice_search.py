import heapq
import itertools
import time
from collections import deque
from typing import Callable, Protocol

from src.rule_extraction.tree_shaped_conjunction import CompactSubTree, TreeShapedConjunction


# Shared traversal engine for searching the lattice of CompactSubTrees of a TreeShapedConjunction,

class Frontier(Protocol):
    # Whether search() needs to track every pushed CompactSubTree in a `seen` set to avoid revisiting one
    # reachable via more than one path (e.g. adding atom A then B vs. B then A). True unless a Frontier
    # overrides it -- only single-path greedy climbs (no backtracking, so no path can ever reconverge)
    # can safely say False.
    needs_dedup: bool = True
    def push(self, item: CompactSubTree) -> None: ...
    def pop(self) -> CompactSubTree: ...
    def is_empty(self) -> bool: ...


class SinglePathFrontier:
    """Keeps only the first item pushed since the last pop, turning the search into a single greedy
    path climb: at each node, commit to its first successor (in get_successors() order) and ignore the
    rest -- mirroring next(node.get_successors(base_tree), None)."""
    # A single monotonically-growing path can never revisit a state (every successor is strictly bigger
    # than its predecessor), so search()'s seen-set dedup would never actually trigger here -- skip it.
    needs_dedup = False
    def __init__(self):
        self._item: CompactSubTree | None = None
    def push(self, item: CompactSubTree) -> None:
        if self._item is None:
            self._item = item
    def pop(self) -> CompactSubTree:
        item = self._item
        self._item = None
        assert item is not None
        return item
    def is_empty(self) -> bool:
        return self._item is None


class GreedyBestSuccessorFrontier:
    """Like SinglePathFrontier, a single greedy path climb -- NOT best-first search over the whole
    lattice. The difference is which successor it commits to: of all the successors of the node just
    expanded, it keeps only the highest-scoring one (by `score_fn`) and discards the rest right away,
    so they're never explored. That one successor is then, in turn, expanded the same way -- always one
    atom added per step, never backtracking to a discarded alternative."""
    # Same reasoning as SinglePathFrontier.needs_dedup: a single growing path can't revisit a state.
    needs_dedup = False
    def __init__(self, score_fn: Callable[[CompactSubTree], float]):
        self._score_fn = score_fn
        self._best_item: CompactSubTree | None = None
        self._best_score: float = float("-inf")
    def push(self, item: CompactSubTree) -> None:
        score = self._score_fn(item)
        if self._best_item is None or score > self._best_score:
            self._best_item, self._best_score = item, score
    def pop(self) -> CompactSubTree:
        item = self._best_item
        assert item is not None
        self._best_item, self._best_score = None, float("-inf")
        # If score_fn exposes a prune_to hook (see rule_optimisation_2.path_weight_score_fn), tell it
        # we've committed to item: every other alternative it may have cached is now unreachable, since
        # this frontier never backtracks to a discarded successor.
        prune_to = getattr(self._score_fn, "prune_to", None)
        if prune_to is not None:
            prune_to(item)
        return item
    def is_empty(self) -> bool:
        return self._best_item is None


class DFSFrontier:
    needs_dedup = True  # explores multiple branches, which can reconverge on the same state
    def __init__(self):
        self._stack: list[CompactSubTree] = []
    def push(self, item: CompactSubTree) -> None:
        self._stack.append(item)
    def pop(self) -> CompactSubTree:
        return self._stack.pop()
    def is_empty(self) -> bool:
        return len(self._stack) == 0


class BFSFrontier:
    needs_dedup = True  # explores multiple branches, which can reconverge on the same state
    def __init__(self):
        self._queue: deque[CompactSubTree] = deque()
    def push(self, item: CompactSubTree) -> None:
        self._queue.append(item)
    def pop(self) -> CompactSubTree:
        return self._queue.popleft()
    def is_empty(self) -> bool:
        return len(self._queue) == 0


class PriorityFrontier:
    """Pops the highest-scored pending node first (ties broken FIFO).
    NOTE: unlike BFSFrontier, this does not visit nodes in non-decreasing rank order, so pairing it
    with FirstResultPolicy does NOT guarantee the smallest sound node -- just the first one found."""
    needs_dedup = True  # explores multiple branches, which can reconverge on the same state
    def __init__(self, score_fn: Callable[[CompactSubTree], float]):
        self._heap: list[tuple[float, int, CompactSubTree]] = []
        self._counter = itertools.count()
        self._score_fn = score_fn
    def push(self, item: CompactSubTree) -> None:
        heapq.heappush(self._heap, (-self._score_fn(item), next(self._counter), item))
    def pop(self) -> CompactSubTree:
        return heapq.heappop(self._heap)[2]
    def is_empty(self) -> bool:
        return not self._heap


class ResultPolicy(Protocol):
    # Called only on nodes not already dominated. Returns True if the whole search should stop now.
    def accept(self, node: CompactSubTree) -> bool: ...
    # Called before check_soundness; returning True skips (and never expands) the node.
    def is_dominated(self, node: CompactSubTree) -> bool: ...


class FirstResultPolicy:
    """Stop at the first sound node found. See PriorityFrontier's note above for the caveat about
    what "first found" does and doesn't guarantee."""
    def accept(self, node: CompactSubTree) -> bool:
        return True
    def is_dominated(self, node: CompactSubTree) -> bool:
        return False


class AllMinimalPolicy:
    """Never stop early. Dominance-prune: a node that is a superset of an already-confirmed-sound
    witness must be sound too (monotonicity) but is redundant (not inclusion-minimal), so it's
    skipped without spending a check_soundness call or being expanded further."""
    def __init__(self):
        self.witnesses: list[CompactSubTree] = []
    def accept(self, node: CompactSubTree) -> bool:
        self.witnesses.append(node)
        return False
    def is_dominated(self, node: CompactSubTree) -> bool:
        return any(node.is_superset_of(w) for w in self.witnesses)


def search(base_tree: TreeShapedConjunction, check_soundness: Callable[[CompactSubTree], bool],
          frontier: Frontier, policy: ResultPolicy, deadline: float | None = None):
    """Generic lattice-boundary search. Yields every node accepted by `policy`, in the order
    `frontier` produces them, until the frontier is exhausted, `policy.accept` says to stop, or
    `deadline` (an absolute time.monotonic() value) passes."""
    root = base_tree.initial_compact
    # Frontiers that explore more than one branch (BFS/DFS/PriorityFrontier) need this: two different
    # paths can add the same atoms in a different order and reconverge on an equal CompactSubTree, so
    # without it the same state would be (re-)expanded repeatedly. A single-path greedy climb
    # (SinglePathFrontier/GreedyBestSuccessorFrontier, frontier.needs_dedup == False) never backtracks
    # and only ever grows, so it can never revisit a state -- tracking `seen` for one of those would
    # just hash and retain every discarded alternative for the rest of the run, for no benefit.
    needs_dedup = getattr(frontier, "needs_dedup", True)
    seen = {root} if needs_dedup else None
    frontier.push(root)
    while not frontier.is_empty():
        if deadline is not None and time.monotonic() >= deadline:
            return
        node = frontier.pop()
        if policy.is_dominated(node):
            continue
        if check_soundness(node):
            yield node
            if policy.accept(node):
                return
            continue  # sound nodes are never expanded further: any successor would be dominated too
        for successor in node.get_successors(base_tree):
            if needs_dedup:
                if successor in seen:
                    continue
                seen.add(successor)
            frontier.push(successor)
