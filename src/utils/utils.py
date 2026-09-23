import bisect
import os.path
import sys
from pathlib import Path

import torch

from src.utils.bitset import BitSet

TYPE_PRED = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'

def find_index_and_insert(sorted_list: list[int], element: int) -> tuple[int, bool]:
    idx = bisect.bisect_left(sorted_list, element)
    if idx == len(sorted_list) or sorted_list[idx] != element:
        sorted_list.insert(idx, element)
        return idx, True
    return idx, False

def check(path: Path, fileid):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {fileid} file not found in {path}")
    else:
        return path

def load_predicates(predicates_file):
    '''Load the predicates from their file into memory, return them.'''
    # Lists to store binary and unary predicates
    binary_predicates = []
    unary_predicates = []

    # IT IS CRITICAL TO RESPECT THE ORDER OF PREDICATES IN THE LIST!!
    try:
        with open(predicates_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                # Every line is of form "predicate,arity"
                predicate, arity = line.strip().rsplit(',', 1)
                if int(arity) == 1:
                    unary_predicates.append(predicate)
                else:
                    binary_predicates.append(predicate)

        # TODO: sanity check - no duplicates
        return binary_predicates, unary_predicates

    except FileNotFoundError:
        raise FileNotFoundError('Predicates file {} not found.'.format(predicates_file))


# Takes as input:
# -- a bitset of dimension m representing a relevant subset of features in some layer
# -- a matrix of dimension m x n (typically a matrix for the previous layer)
# -- optionally, a feature vector of dimension n that represent specific activations in the previous layer
# It computes the relevant subset of features in the previous layer
def backpropagate_relevance(current_relevant: BitSet, matrix, previous_activations=None):
    if matrix.shape[0] != current_relevant.dimension:
        raise ValueError(f"Left vector dimension {current_relevant.dimension} does not match matrix row dimension"
                         f"{matrix.shape[0]}")
    if previous_activations is not None and matrix.shape[1] != previous_activations.shape[0]:
        raise ValueError(f"Right vector dimension {previous_activations.shape[0]} does not match matrix column dimension"
                         f"{matrix.shape[1]}")
    if not current_relevant.is_empty():
        matrix_relevant_rows = matrix[current_relevant.elements(), :]
        any_positive = (matrix_relevant_rows > 0).any(dim=0) # dim_{l-1} Boolean vector
        if previous_activations is not None:
            mask = (previous_activations > 0) & any_positive
        else:
            mask = any_positive
        return BitSet.from_subset(matrix.shape[1], set(torch.where(mask)[0].tolist()))
    else:
        # For some reason the above formula does not work in the degenerate case where it's all zeroes.
        return BitSet.from_subset(matrix.shape[1], set())
