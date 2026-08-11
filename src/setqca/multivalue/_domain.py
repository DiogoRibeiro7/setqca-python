"""The property space of a multi-value model.

A multi-value condition takes one of several unordered categories rather than
being present or absent. ``A{0,1,2}`` is a three-level condition, and a case
takes exactly one of those levels.

Configurations are indexed in mixed radix, generalising the binary minterm.
With levels ``(2, 3)`` the index of ``(1, 2)`` is ``1 * 3 + 2 = 5``. Big-endian,
so the first condition is the most significant, matching the binary case.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True, slots=True)
class MultiValueDomain:
    """Condition names and how many levels each takes.

    Parameters
    ----------
    conditions : tuple of str
        Condition names, in the order used for indexing.
    levels : tuple of int
        Number of categories per condition. Level values are ``0..levels-1``.
    """

    conditions: tuple[str, ...]
    levels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("At least one condition is required.")
        if len(self.conditions) != len(self.levels):
            raise ValueError("conditions and levels must have the same length.")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("Condition names must be unique.")
        if any(count < 2 for count in self.levels):
            raise ValueError("Every condition needs at least two levels.")

    @classmethod
    def from_mapping(cls, levels: Mapping[str, int]) -> MultiValueDomain:
        """Build a domain from a ``{condition: levels}`` mapping."""
        return cls(tuple(levels), tuple(levels.values()))

    @property
    def size(self) -> int:
        """Return the number of logically possible configurations."""
        total = 1
        for count in self.levels:
            total *= count
        return total

    @property
    def width(self) -> int:
        """Return the number of conditions."""
        return len(self.conditions)

    def index_of(self, values: Sequence[int]) -> int:
        """Return the mixed-radix index of one configuration.

        Raises
        ------
        ValueError
            If the length is wrong or a value is outside its condition's range.
        """
        if len(values) != self.width:
            raise ValueError(f"Expected {self.width} values, got {len(values)}.")
        index = 0
        for value, count in zip(values, self.levels, strict=True):
            if not 0 <= value < count:
                raise ValueError(f"Value {value} is outside the range 0..{count - 1}.")
            index = index * count + value
        return index

    def values_of(self, index: int) -> tuple[int, ...]:
        """Return the configuration at a mixed-radix index.

        Raises
        ------
        ValueError
            If the index is outside the property space.
        """
        if not 0 <= index < self.size:
            raise ValueError(f"Index {index} is outside the property space of size {self.size}.")
        values: list[int] = []
        remaining = index
        for count in reversed(self.levels):
            values.append(remaining % count)
            remaining //= count
        return tuple(reversed(values))

    def configurations(self) -> Iterator[tuple[int, ...]]:
        """Yield every configuration, in index order."""
        yield from product(*(range(count) for count in self.levels))

    def __str__(self) -> str:
        return ", ".join(
            f"{name}{{{','.join(str(value) for value in range(count))}}}"
            for name, count in zip(self.conditions, self.levels, strict=True)
        )
