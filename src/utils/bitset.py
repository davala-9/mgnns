from dataclasses import dataclass

@dataclass(frozen=True,slots=True)
class BitSet:

    dimension: int = 1
    mask: int = 0

    @classmethod
    def from_subset(cls, dimension, subset):
        mask = 0
        for x in subset:
            if not 0 <= x <= dimension - 1:
                raise ValueError(f"Element {x} is out of range [0, {dimension - 1}]")
            mask |= 1 << x
        return cls(dimension, mask)

    def contains(self, x):
        """Check whether x is in the set."""
        return bool(self.mask & (1 << x))

    def union(self, other):
        return BitSet(self.dimension, self.mask | other.mask)

    def intersection(self, other):
        return BitSet(self.dimension, self.mask & other.mask)

    def difference(self, other):
        return BitSet(self.dimension, self.mask & ~other.mask)

    def elements(self):
        return [i for i in range(self.dimension) if self.mask & (1 << i)]

    def subsetOf(self, other):
        if self.dimension != other.dimension:
            raise ValueError("BitSets must have the same dimension")
        return (self.mask & ~other.mask) == 0

    def as_set(self):
        return set(self.elements())

    def as_vector(self):
        vector = [0] * self.dimension
        for i in self.elements():
            vector[i] = 1
        return vector

    def __eq__(self, other):
        if not isinstance(other, BitSet):
            return NotImplemented
        return (
                self.dimension == other.dimension
                and self.mask == other.mask
        )

    def new_elements(self, other):
        if not self.subsetOf(other):
            raise ValueError(f"Feature Set {self} should be a subset of feature set {other}")
        diff = other.mask & ~self.mask
        result = []

        while diff:
            lsb = diff & -diff
            result.append(lsb.bit_length() - 1)
            diff ^= lsb

        return result

    def is_empty(self):
        return self.mask == 0

    def clone(self):
        return BitSet(dimension=self.dimension, mask=self.mask)

    def to_empty_compressed(self):
        k = bin(self.mask).count("1")
        return BitSet(dimension=k)

    def from_compressed(self, other):
        expected_dim = bin(self.mask).count("1")
        if other.dimension != expected_dim:
            raise ValueError(
                f"Compressed BitSet has dimension {other.dimension}, "
                f"expected {expected_dim} )"
            )
        result_mask = 0
        y_positions = self.elements()  # positions of ones in Y, ascending order
        for j, pos in enumerate(y_positions):
            if other.mask & (1 << j):
                result_mask |= 1 << pos

        return BitSet(self.dimension, result_mask)

    # Yield all instances obtained from the current one by switching exactly one 0 to a 1.
    def successors(self):
        full_mask = (1 << self.dimension) - 1
        diff = full_mask ^ self.mask  # bits that are 0 in self

        while diff:
            lsb = diff & -diff
            yield BitSet(self.dimension, self.mask | lsb)
            diff ^= lsb


    def __repr__(self):
        return f"BitSet({self.elements()})"