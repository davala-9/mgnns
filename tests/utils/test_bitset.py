import pytest

from src.utils.bitset import BitSet


def test_init_empty():
    bs = BitSet(8)

    assert bs.dimension == 8
    assert bs.mask == 0
    assert bs.elements() == []
    assert bs.as_set() == set()


def test_from_subset():
    bs = BitSet.from_subset(8, {1, 3, 5})

    assert bs.dimension == 8
    assert bs.mask == int("00101010",2)
    assert bs.as_set() == {1, 3, 5}


@pytest.mark.parametrize("subset", [
    {-1},
    {5},
    {0, 1, 6},
])
def test_from_subset_out_of_range_raises(subset):
    with pytest.raises(ValueError):
        BitSet.from_subset(5, subset)



@pytest.mark.parametrize(
    ("subset", "x", "expected"),
    [
        ({0, 2, 4}, 0, True),
        ({0, 2, 4}, 2, True),
        ({0, 2, 4}, 1, False),
        (set(), 0, False),
    ],
)
def test_contains(subset, x, expected):
    bs = BitSet.from_subset(8, subset)
    assert bs.contains(x) is expected


def test_union():
    a = BitSet.from_subset(8, {1, 2, 3})
    b = BitSet.from_subset(8, {3, 4, 5})
    result = a.union(b)
    assert result.as_set() == {1, 2, 3, 4, 5}


def test_intersection():
    a = BitSet.from_subset(8, {1, 2, 3})
    b = BitSet.from_subset(8, {3, 4, 5})
    result = a.intersection(b)
    assert result.as_set() == {3}


def test_difference():
    a = BitSet.from_subset(8, {1, 2, 3, 4})
    b = BitSet.from_subset(8, {2, 4, 6})
    result = a.difference(b)
    assert result.as_set() == {1, 3}


def test_elements_are_sorted():
    bs = BitSet.from_subset(8, {5, 1, 3})
    assert bs.elements() == [1, 3, 5]


def test_as_set():
    bs = BitSet.from_subset(8, {2, 4, 6})
    assert bs.as_set() == {2, 4, 6}


def test_repr():
    bs = BitSet.from_subset(8, {1, 3})
    assert repr(bs) == "BitSet(8,[1, 3])"


def test_set_operations_return_new_instances():
    a = BitSet.from_subset(8, {1, 2})
    b = BitSet.from_subset(8, {2, 3})
    result = a.union(b)
    assert result is not a
    assert result is not b
    assert a.as_set() == {1, 2}
    assert b.as_set() == {2, 3}


def test_add_element():
    bs = BitSet.from_subset(8, {1, 3})
    result = bs.add_element(5)
    assert result.as_set() == {1, 3, 5}


def test_add_element_already_present_is_idempotent():
    bs = BitSet.from_subset(8, {1, 3})
    result = bs.add_element(3)
    assert result.as_set() == {1, 3}


def test_add_element_returns_new_instance():
    bs = BitSet.from_subset(8, {1, 3})
    result = bs.add_element(5)
    assert result is not bs
    assert bs.as_set() == {1, 3}  # original left untouched


def test_add_element_out_of_range_raises():
    bs = BitSet.from_subset(5, {0, 2})
    with pytest.raises(ValueError):
        bs.add_element(5)