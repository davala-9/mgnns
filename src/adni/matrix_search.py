from typing import Callable

from src.adni.heuristic import heuristic_for_element
from src.adni.signature import AdniSignature
from src.adni.sparse_triangular_matrix import SparseTriangularMatrix


# A dense d x d table where table[i][j] (for 0 <= i <= j < d) is every candidate value 1..max_value for cell (i, j),
# sorted by heuristic_for_element(i, j, value, ...) in descending order
def value_order_table(signature: AdniSignature, model) -> list:
    table = [[None] * signature.d for _ in range(signature.d)]
    for i in range(signature.d):
        for j in range(i, signature.d):
            table[i][j] = sorted(
                range(1, signature.max_value + 1),
                key=lambda v: heuristic_for_element(i, j, v, signature, model),
                reverse=True,
            )
    return table


# The candidate values for cell (i, j), in value_order, that have a strictly bigger heuristic than its current
# value (all of them if the cell is still 0). Useful for potential future search strategies (unnecessary for greedy)
def _better_candidates(value_order: list, i: int, j: int, current: int) -> list:
    candidates = value_order[i][j]
    return candidates if current == 0 else candidates[:candidates.index(current)]


# Single-path greedy climb, no backtracking: at each step, look at every cell's best still-available
# move and commit to whichever one's heuristic_for_element score is the biggest across the whole matrix.
# Can print progress.
def greedy_climb(matrix: SparseTriangularMatrix, value_order: list, signature: AdniSignature, model,
                  check_soundness: Callable[[SparseTriangularMatrix], bool], verbose: bool = False):
    step = 0
    while not check_soundness(matrix):
        entries = matrix.to_dict()
        best_move = None  # (score, i, j, value)
        cells_with_a_move = 0
        for i in range(matrix.d):
            for j in range(i, matrix.d):
                remaining = _better_candidates(value_order, i, j, entries.get((i, j), 0))
                if not remaining:
                    continue
                cells_with_a_move += 1
                value = remaining[0]
                score = heuristic_for_element(i, j, value, signature, model)
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


# Given an already-sound matrix, greedily drops cells not needed: tries removing the weakest-heuristic cell first
# and keeps the removal permanently if the matrix is still sound without it; otherwise leaves it and moves on to next
def minimise(matrix: SparseTriangularMatrix, signature: AdniSignature, model,
             check_soundness: Callable[[SparseTriangularMatrix], bool], verbose: bool = False) -> SparseTriangularMatrix:
    cells = sorted(
        matrix.to_dict().items(),
        key=lambda entry: heuristic_for_element(entry[0][0], entry[0][1], entry[1], signature, model))
    for (i, j), value in cells:
        candidate = matrix.with_value(i, j, 0)
        if check_soundness(candidate):
            matrix = candidate
            if verbose:
                print(f"Removed ({i}, {j}) = {value}: still sound without it.")
    return matrix
