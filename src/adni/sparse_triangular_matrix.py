from dataclasses import dataclass, field


def bits_needed(count: int) -> int:
    """Minimum number of bits needed to represent `count` distinct values (0..count-1)."""
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    return (count - 1).bit_length()


# Number of (row, col) pairs, in row-major order, that come before row i in an upper-triangular
# d x d matrix (cells (i, j) with i <= j < d).
def _row_start(i: int, d: int) -> int:
    return i * d - i * (i - 1) // 2


def _ij_to_index(i: int, j: int, d: int) -> int:
    if not 0 <= i <= j < d:
        raise ValueError(f"({i}, {j}) is not a valid upper-triangular cell for a {d}x{d} matrix")
    return _row_start(i, d) + (j - i)


def _index_to_ij(index: int, d: int) -> tuple[int, int]:
    n = d * (d + 1) // 2
    if not 0 <= index < n:
        raise ValueError(f"index {index} out of range [0, {n})")
    # _row_start is strictly increasing in i, so binary-search for the largest i whose row
    # starts at or before `index`.
    lo, hi = 0, d - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _row_start(mid, d) <= index:
            lo = mid
        else:
            hi = mid - 1
    i = lo
    j = index - _row_start(i, d) + i
    return i, j


# A d x d upper-triangular matrix (row <= col, diagonal included), entries in [0, max_value].
# Stored sparsely: only nonzero cells are kept, each packed into a single int (the cell's linear
# (row, col) index, followed by its value) -- so the whole matrix is a frozenset of small ints,
# cheap to hash/compare/copy. Same spirit as BitSet (src/utils/bitset.py), which packs many
# single-bit flags into one int instead of keeping a Python object per flag.
@dataclass(frozen=True, slots=True)
class SparseTriangularMatrix:
    d: int
    max_value: int
    packed_entries: frozenset = field(default_factory=frozenset)

    @property
    def n_cells(self) -> int:
        return self.d * (self.d + 1) // 2

    @property
    def index_bits(self) -> int:
        return bits_needed(self.n_cells)

    @property
    def value_bits(self) -> int:
        return bits_needed(self.max_value + 1)

    @classmethod
    def empty(cls, d: int, max_value: int) -> "SparseTriangularMatrix":
        return cls(d, max_value, frozenset())

    @classmethod
    def from_dict(cls, d: int, max_value: int, values: dict) -> "SparseTriangularMatrix":
        matrix = cls.empty(d, max_value)
        for (i, j), value in values.items():
            matrix = matrix.with_value(i, j, value)
        return matrix

    def get(self, i: int, j: int) -> int:
        target_index = _ij_to_index(i, j, self.d)
        shift = self.value_bits
        for packed in self.packed_entries:
            if (packed >> shift) == target_index:
                return packed & ((1 << shift) - 1)
        return 0

    def with_value(self, i: int, j: int, value: int) -> "SparseTriangularMatrix":
        if not 0 <= i <= j < self.d:
            raise ValueError(f"({i}, {j}) is not a valid upper-triangular cell for a {self.d}x{self.d} matrix")
        if not 0 <= value <= self.max_value:
            raise ValueError(f"value {value} out of range [0, {self.max_value}]")
        target_index = _ij_to_index(i, j, self.d)
        shift = self.value_bits
        remaining = frozenset(p for p in self.packed_entries if (p >> shift) != target_index)
        if value == 0:  # 0 is the default; don't store it
            return SparseTriangularMatrix(self.d, self.max_value, remaining)
        packed = (target_index << shift) | value
        return SparseTriangularMatrix(self.d, self.max_value, remaining | {packed})

    def to_dict(self) -> dict:
        shift = self.value_bits
        result = {}
        for packed in self.packed_entries:
            index = packed >> shift
            value = packed & ((1 << shift) - 1)
            result[_index_to_ij(index, self.d)] = value
        return result
