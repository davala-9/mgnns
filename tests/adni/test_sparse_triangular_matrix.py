import pytest

from src.adni.sparse_triangular_matrix import SparseTriangularMatrix, bits_needed, _ij_to_index, _index_to_ij


# --- bits_needed ---

def test_bits_needed_matches_worked_example():
    # 90x90 triangular incl. diagonal has 90*91/2 = 4095 cells -> needs 12 bits (2**12 = 4096).
    assert bits_needed(4095) == 12
    # Values 0..10 is 11 possibilities -> needs 4 bits (2**4 = 16).
    assert bits_needed(11) == 4

def test_bits_needed_single_value_needs_no_bits():
    assert bits_needed(1) == 0

def test_bits_needed_rejects_non_positive_count():
    with pytest.raises(ValueError):
        bits_needed(0)


# --- (i, j) <-> linear index ---

def test_ij_to_index_covers_every_cell_of_a_small_matrix_without_gaps_or_repeats():
    d = 5
    n = d * (d + 1) // 2
    seen = {_ij_to_index(i, j, d) for i in range(d) for j in range(i, d)}
    assert seen == set(range(n))

def test_index_to_ij_is_the_inverse_of_ij_to_index():
    d = 5
    for i in range(d):
        for j in range(i, d):
            index = _ij_to_index(i, j, d)
            assert _index_to_ij(index, d) == (i, j)

def test_ij_to_index_rejects_below_diagonal_cell():
    with pytest.raises(ValueError):
        _ij_to_index(2, 1, d=5)  # j < i: not part of the upper triangle


# --- SparseTriangularMatrix ---

def test_empty_matrix_reads_back_zero_everywhere():
    matrix = SparseTriangularMatrix.empty(d=90, max_value=10)
    assert matrix.get(0, 0) == 0
    assert matrix.get(3, 89) == 0

def test_matches_the_worked_90x90_example():
    matrix = SparseTriangularMatrix.empty(d=90, max_value=10)
    assert matrix.index_bits == 12
    assert matrix.value_bits == 4

def test_with_value_sets_a_cell_without_disturbing_others():
    matrix = SparseTriangularMatrix.empty(d=10, max_value=10).with_value(2, 7, 5)
    assert matrix.get(2, 7) == 5
    assert matrix.get(0, 0) == 0

def test_with_value_overwrites_an_existing_cell():
    matrix = SparseTriangularMatrix.empty(d=10, max_value=10).with_value(2, 7, 5).with_value(2, 7, 9)
    assert matrix.get(2, 7) == 9
    assert len(matrix.packed_entries) == 1

def test_with_value_zero_removes_the_entry_entirely():
    matrix = SparseTriangularMatrix.empty(d=10, max_value=10).with_value(2, 7, 5).with_value(2, 7, 0)
    assert matrix.get(2, 7) == 0
    assert matrix.packed_entries == frozenset()

def test_with_value_rejects_a_below_diagonal_cell():
    matrix = SparseTriangularMatrix.empty(d=10, max_value=10)
    with pytest.raises(ValueError):
        matrix.with_value(7, 2, 5)

def test_with_value_rejects_a_value_outside_the_allowed_range():
    matrix = SparseTriangularMatrix.empty(d=10, max_value=10)
    with pytest.raises(ValueError):
        matrix.with_value(0, 0, 11)

def test_from_dict_and_to_dict_round_trip():
    values = {(0, 0): 3, (1, 4): 10, (9, 9): 1}
    matrix = SparseTriangularMatrix.from_dict(d=10, max_value=10, values=values)
    assert matrix.to_dict() == values

def test_equal_matrices_with_same_entries_are_equal_and_hash_equal():
    a = SparseTriangularMatrix.empty(d=10, max_value=10).with_value(1, 2, 3)
    b = SparseTriangularMatrix.empty(d=10, max_value=10).with_value(1, 2, 3)
    assert a == b
    assert hash(a) == hash(b)

def test_matrix_is_immutable():
    original = SparseTriangularMatrix.empty(d=10, max_value=10)
    updated = original.with_value(1, 2, 3)
    assert original.get(1, 2) == 0
    assert updated.get(1, 2) == 3
