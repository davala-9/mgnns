from typing import Callable

from src.adni.heuristic import heuristic_for_element
from src.adni.sparse_triangular_matrix import SparseTriangularMatrix


# A dense (not sparse) d x d table where table[i][j] (for 0 <= i <= j < d) is every candidate value
# 1..max_value for cell (i, j), sorted by heuristic_for_element(i, j, value, ...) in descending order --
# the order get_successors tries them in, so that adding or improving a cell always offers its
# biggest-heuristic options first. Cells below the diagonal are left as None: a triangular matrix never
# has an (i, j) with i > j, so there's nothing to order there.
def value_order_table(d: int, max_value: int, internal_encoder, model) -> list:
    table = [[None] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            table[i][j] = sorted(
                range(1, max_value + 1),
                key=lambda v: heuristic_for_element(i, j, v, internal_encoder, model),
                reverse=True,
            )
    return table


# Every successor of `matrix` under `value_order` (see value_order_table): for a cell without a value
# yet, one successor per candidate value (adding it, biggest-heuristic first); for a cell that already
# has a value, one successor per candidate value with a strictly bigger heuristic than its current one
# (replacing it, biggest first). Every successor differs from `matrix` in exactly one cell.
def get_successors(matrix: SparseTriangularMatrix, value_order: list):
    entries = matrix.to_dict()
    for i in range(matrix.d):
        for j in range(i, matrix.d):
            candidates = value_order[i][j]
            current = entries.get((i, j), 0)
            better_candidates = candidates if current == 0 else candidates[:candidates.index(current)]
            for value in better_candidates:
                yield matrix.with_value(i, j, value)


# Single-path greedy climb, no backtracking: at each step, look at every cell's best still-available
# move (its highest-remaining-heuristic candidate in value_order) and commit to whichever one's own
# heuristic_for_element score is the biggest across the whole matrix. Nothing else is ever kept --
# there's no going back to a cell not picked this round, so there'd be no point storing it. Stops as
# soon as check_soundness(matrix) succeeds, or once no cell has a move left, in which case it's a dead
# end and this returns None. With verbose=True, prints one progress line per move (which cell/value was
# picked, its heuristic score, and how many other cells still had a candidate on offer that step).
def greedy_climb(matrix: SparseTriangularMatrix, value_order: list, internal_encoder, model,
                  check_soundness: Callable[[SparseTriangularMatrix], bool], verbose: bool = False):
    step = 0
    while not check_soundness(matrix):
        entries = matrix.to_dict()
        best_move = None  # (score, i, j, value)
        cells_with_a_move = 0
        for i in range(matrix.d):
            for j in range(i, matrix.d):
                candidates = value_order[i][j]
                current = entries.get((i, j), 0)
                remaining = candidates if current == 0 else candidates[:candidates.index(current)]
                if not remaining:
                    continue
                cells_with_a_move += 1
                value = remaining[0]
                score = heuristic_for_element(i, j, value, internal_encoder, model)
                if best_move is None or score > best_move[0]:
                    best_move = (score, i, j, value)
        if best_move is None:
            if verbose:
                print(f"Greedy climb dead-ended after {step} move(s): no cell has a candidate left.")
            return None
        score, i, j, value = best_move
        step += 1
        if verbose:
            print(f"Step {step}: set ({i}, {j}) = {value} (heuristic {score:.4f}); "
                  f"{cells_with_a_move} cell(s) had a candidate to choose from.")
        matrix = matrix.with_value(i, j, value)
    if verbose:
        print(f"Greedy climb found a sound matrix after {step} move(s).")
    return matrix


# Given an already-sound matrix, greedily drops cells that turn out not to be needed: tries the
# weakest-heuristic cell first (most likely to be redundant), tentatively removes it, and keeps the
# removal permanently if the matrix is still sound without it -- otherwise leaves it and moves on to
# the next-weakest cell. A single pass over every originally-present cell (weakest to strongest)
# suffices: soundness only ever gets *harder* to keep as cells are removed, so a cell that fails this
# test can never pass it later against an even smaller matrix, and one that passes stays removed for
# good (matching AllMinimalPolicy's monotonicity assumption in lattice_search.py). Not guaranteed to
# find the smallest possible sound matrix -- like the rest of this module, it's a greedy approximation.
def minimise(matrix: SparseTriangularMatrix, internal_encoder, model,
             check_soundness: Callable[[SparseTriangularMatrix], bool], verbose: bool = False) -> SparseTriangularMatrix:
    cells = sorted(
        matrix.to_dict().items(),
        key=lambda entry: heuristic_for_element(entry[0][0], entry[0][1], entry[1], internal_encoder, model))
    for (i, j), value in cells:
        candidate = matrix.with_value(i, j, 0)
        if check_soundness(candidate):
            matrix = candidate
            if verbose:
                print(f"Removed ({i}, {j}) = {value}: still sound without it.")
    return matrix


# matrix_to_tree always makes every one of a matrix's d nodes a part_of child of the root, whether or
# not the matrix actually touches it -- so even a maximally minimised matrix (see minimise above) still
# produces a tree with d node atoms attached. This tries dropping each node that ends up with no
# incident matrix entry at all (neither as a row nor a column) entirely from the tree, re-checking
# soundness the same way as minimise. `check_soundness_with_nodes(matrix, included_nodes)` plays the
# same role as minimise's check_soundness, but also takes the current candidate set of included node
# indices (see matrix_to_tree's included_nodes parameter). Isolated nodes have no heuristic of their
# own to rank by (they contribute nothing but their unconditional part_of self-loop), so they're tried
# in an arbitrary (ascending index) order; one pass suffices, for the same monotonicity reason as
# minimise. Returns the set of node indices that should stay in the final tree.
def prune_isolated_nodes(matrix: SparseTriangularMatrix,
                          check_soundness_with_nodes: Callable[[SparseTriangularMatrix, set], bool],
                          verbose: bool = False) -> set:
    entries = matrix.to_dict()
    touched = {k for (i, j) in entries for k in (i, j)}
    isolated = sorted(set(range(matrix.d)) - touched)
    included = set(range(matrix.d))
    for k in isolated:
        candidate = included - {k}
        if check_soundness_with_nodes(matrix, candidate):
            included = candidate
            if verbose:
                print(f"Dropped isolated node {k}: still sound without it.")
    return included
