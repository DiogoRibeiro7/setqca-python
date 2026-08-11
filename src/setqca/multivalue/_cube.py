"""Multi-value cubes and their exact minimisation.

A cube in a multi-value property space allows a **set** of levels for each
condition, rather than a single value or a don't-care. ``A{0,2}*B{1}`` is a
legitimate term, and a condition whose set contains every level is simply
absent from the expression.

Why not Boolean dummies
-----------------------

The obvious shortcut is to encode ``A{0,1,2}`` as three binary indicators and
reuse the binary minimiser. That transformation does **not** preserve the
semantics. The binary space contains points such as ``A_0 = A_1 = 1``, which
correspond to no configuration at all, and the minimiser is free to build
implicants across them — producing terms that look valid and describe nothing.
Recovering a multi-value expression afterwards requires exactly the
mutual-exclusivity constraints the encoding discarded.

So the cube algebra is implemented directly. Merging is the generalisation of
the binary rule:

    two cubes that agree on every condition but one merge into a single cube
    whose set at that condition is the union of the two.

Because the two cubes agree everywhere else, the merged cube covers exactly
their union and nothing more, so merging can never introduce coverage of a
configuration that was not already covered. That is the property the binary
rule relies on, and it holds unchanged for sets.

The exact cover is then solved by the same verified solver the binary engine
uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from setqca.minimize.qmc import solve_minimum_cover

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from ._domain import MultiValueDomain


@dataclass(frozen=True, slots=True)
class MultiValueCube:
    """A conjunction allowing a set of levels for each condition."""

    pattern: tuple[frozenset[int], ...]

    @classmethod
    def from_configuration(cls, values: tuple[int, ...]) -> MultiValueCube:
        """Build the cube covering exactly one configuration."""
        return cls(tuple(frozenset({value}) for value in values))

    def literals(self, domain: MultiValueDomain) -> int:
        """Return the number of conditions the cube actually constrains."""
        return sum(
            1
            for allowed, count in zip(self.pattern, domain.levels, strict=True)
            if len(allowed) < count
        )

    def is_tautology(self, domain: MultiValueDomain) -> bool:
        """Return whether the cube constrains nothing."""
        return self.literals(domain) == 0

    def covers_values(self, values: tuple[int, ...]) -> bool:
        """Return whether a configuration falls inside the cube."""
        return all(value in allowed for value, allowed in zip(values, self.pattern, strict=True))

    def covers(self, index: int, domain: MultiValueDomain) -> bool:
        """Return whether the configuration at an index falls inside the cube."""
        return self.covers_values(domain.values_of(index))

    def contains(self, other: MultiValueCube) -> bool:
        """Return whether this cube covers everything ``other`` covers."""
        return all(theirs <= mine for mine, theirs in zip(self.pattern, other.pattern, strict=True))

    def merge(self, other: MultiValueCube) -> MultiValueCube | None:
        """Merge two cubes differing at exactly one condition.

        Returns ``None`` when they differ at none or several, in which case the
        union would cover configurations neither cube covers.
        """
        differing = [
            index
            for index, (mine, theirs) in enumerate(zip(self.pattern, other.pattern, strict=True))
            if mine != theirs
        ]
        if len(differing) != 1:
            return None
        position = differing[0]
        pattern = list(self.pattern)
        pattern[position] = self.pattern[position] | other.pattern[position]
        return MultiValueCube(tuple(pattern))

    def as_expression(self, domain: MultiValueDomain) -> str:
        """Render in multi-value QCA notation, for example ``A{0,2}*B{1}``.

        Conditions allowing every level are omitted, since they constrain
        nothing. A cube constraining nothing renders as ``1``.
        """
        parts = [
            f"{name}{{{','.join(str(value) for value in sorted(allowed))}}}"
            for name, allowed, count in zip(
                domain.conditions, self.pattern, domain.levels, strict=True
            )
            if len(allowed) < count
        ]
        return "*".join(parts) if parts else "1"


@dataclass(frozen=True, slots=True)
class MultiValueSolution:
    """A minimal cover of multi-value configurations."""

    cubes: tuple[MultiValueCube, ...]

    def literal_count(self, domain: MultiValueDomain) -> int:
        """Return the total number of constrained conditions."""
        return sum(cube.literals(domain) for cube in self.cubes)

    def as_expression(self, domain: MultiValueDomain) -> str:
        """Render the whole cover, for example ``A{0}*B{1} + A{2}``."""
        return " + ".join(cube.as_expression(domain) for cube in self.cubes)

    def covers(self, index: int, domain: MultiValueDomain) -> bool:
        """Return whether any cube covers a configuration."""
        return any(cube.covers(index, domain) for cube in self.cubes)


def prime_cubes(
    on_set: set[int], dont_cares: set[int], domain: MultiValueDomain
) -> tuple[MultiValueCube, ...]:
    """Generate every prime cube for a multi-value problem.

    Parameters
    ----------
    on_set : set of int
        Configuration indices that must be covered.
    dont_cares : set of int
        Configurations usable but not required.
    domain : MultiValueDomain
        The property space.

    Returns
    -------
    tuple of MultiValueCube
        Prime cubes, ordered by literal count then rendered form, and filtered
        to those covering at least one required configuration.

    Raises
    ------
    ValueError
        If the two sets overlap.
    """
    if on_set & dont_cares:
        raise ValueError("on_set and dont_cares must be disjoint.")
    universe = on_set | dont_cares
    if not universe:
        return ()

    generated = {MultiValueCube.from_configuration(domain.values_of(index)) for index in universe}
    frontier = set(generated)

    while frontier:
        produced: set[MultiValueCube] = set()
        ordered = sorted(frontier, key=lambda cube: tuple(sorted(sorted(s) for s in cube.pattern)))
        for position, left in enumerate(ordered):
            for right in ordered[position + 1 :]:
                merged = left.merge(right)
                if merged is not None and merged not in generated:
                    produced.add(merged)
        generated |= produced
        frontier = produced

    # A cube contained in another is not prime. Equality is excluded so that
    # two identical cubes do not eliminate each other.
    primes = [
        cube
        for cube in generated
        if not any(other != cube and other.contains(cube) for other in generated)
    ]
    useful = [cube for cube in primes if any(cube.covers(index, domain) for index in on_set)]
    return tuple(
        sorted(useful, key=lambda cube: (cube.literals(domain), cube.as_expression(domain)))
    )


def minimize_multivalue(
    on_set: set[int],
    *,
    domain: MultiValueDomain,
    dont_cares: set[int] | None = None,
    max_solutions: int = 256,
) -> tuple[MultiValueSolution, ...]:
    """Return every exact minimum cover of a multi-value problem.

    Parameters
    ----------
    on_set : set of int
        Configuration indices that must be covered.
    domain : MultiValueDomain
        The property space.
    dont_cares : set of int, optional
        Configurations usable but not required, typically logical remainders.
    max_solutions : int, default 256
        Upper bound on tied minimum covers.

    Returns
    -------
    tuple of MultiValueSolution
        Every cover of provably minimal cost.
    """
    required = set(on_set)
    if not required:
        return (MultiValueSolution(()),)

    primes = prime_cubes(required, set(dont_cares or ()), domain)
    covered = [
        frozenset(index for index in required if cube.covers(index, domain)) for cube in primes
    ]
    literals = [cube.literals(domain) for cube in primes]
    choices = solve_minimum_cover(covered, literals, required, max_solutions=max_solutions)
    return tuple(MultiValueSolution(tuple(primes[i] for i in indices)) for indices in choices)
