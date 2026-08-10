"""Boolean implicant representation used by exact QMC minimisation."""

from __future__ import annotations

from dataclasses import dataclass

Bit = int | None


@dataclass(frozen=True, slots=True)
class Implicant:
    """A Boolean cube and the minterms from which it was derived."""

    pattern: tuple[Bit, ...]
    origins: frozenset[int]

    @property
    def literals(self) -> int:
        """Number of non-minimised literals."""
        return sum(value is not None for value in self.pattern)

    def covers(self, minterm: int) -> bool:
        """Return whether this cube covers a binary minterm."""
        width = len(self.pattern)
        bits = tuple((minterm >> shift) & 1 for shift in reversed(range(width)))
        return all(p is None or p == b for p, b in zip(self.pattern, bits, strict=True))

    def combine(self, other: Implicant) -> Implicant | None:
        """Combine cubes that differ in exactly one fixed literal."""
        if len(self.pattern) != len(other.pattern):
            return None
        differences: list[int] = []
        for idx, (left, right) in enumerate(zip(self.pattern, other.pattern, strict=True)):
            if left == right:
                continue
            if left is None or right is None:
                return None
            differences.append(idx)
        if len(differences) != 1:
            return None
        pattern = list(self.pattern)
        pattern[differences[0]] = None
        return Implicant(tuple(pattern), self.origins | other.origins)

    def as_expression(self, conditions: tuple[str, ...]) -> str:
        """Render using standard QCA notation, e.g. ``A*~B``."""
        if len(conditions) != len(self.pattern):
            raise ValueError("condition count does not match implicant width.")
        literals = [
            condition if bit == 1 else f"~{condition}"
            for condition, bit in zip(conditions, self.pattern, strict=True)
            if bit is not None
        ]
        return "1" if not literals else "*".join(literals)


def minterm_to_implicant(minterm: int, width: int) -> Implicant:
    """Create a fully specified implicant from a minterm integer."""
    if minterm < 0 or minterm >= 2**width:
        raise ValueError("minterm is outside the truth-table domain.")
    pattern = tuple((minterm >> shift) & 1 for shift in reversed(range(width)))
    return Implicant(pattern, frozenset({minterm}))
