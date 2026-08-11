"""Exact classical Quine-McCluskey minimisation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .complexity import estimate_complexity, warn_if_complex
from .implicant import Implicant


@dataclass(frozen=True, slots=True)
class BooleanSolution:
    """Exact minimal Boolean cover expressed as a set of prime implicants."""

    implicants: tuple[Implicant, ...]

    @property
    def literal_count(self) -> int:
        """Return the total number of literals across all implicants."""
        return sum(item.literals for item in self.implicants)

    def as_expression(self, conditions: tuple[str, ...]) -> str:
        """Render the cover in standard QCA notation, e.g. ``A*~B + C``."""
        return " + ".join(item.as_expression(conditions) for item in self.implicants)


def prime_implicants(on_set: set[int], dont_cares: set[int], width: int) -> tuple[Implicant, ...]:
    """Generate all prime implicants exactly using classical QMC.

    Internally a cube is a pair of integers — a mask of the fixed positions and
    the values at those positions — rather than a tuple of optional bits.
    Benchmarking showed this phase dominating on remainder-heavy problems,
    which is exactly the parsimonious case, and integer masks make combining
    a few machine operations instead of a tuple walk. The algorithm is
    unchanged; only the representation is.

    Parameters
    ----------
    on_set : set of int
        Minterms that must be covered.
    dont_cares : set of int
        Minterms usable but not required.
    width : int
        Number of conditions.

    Returns
    -------
    tuple of Implicant
        Primes covering at least one required minterm, ordered by literal
        count then bit pattern.

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

    full_mask = (1 << width) - 1
    # A cube is (mask, value): `mask` marks the fixed positions, `value` holds
    # the bits there. A minterm m is covered when `m & mask == value`.
    current: set[tuple[int, int]] = {(full_mask, minterm) for minterm in universe}
    primes: set[tuple[int, int]] = set()

    while current:
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for mask, value in current:
            grouped[(mask, value.bit_count())].append(value)

        used: set[tuple[int, int]] = set()
        next_round: set[tuple[int, int]] = set()
        for (mask, ones), values in grouped.items():
            neighbours = grouped.get((mask, ones + 1))
            if not neighbours:
                continue
            for left in values:
                for right in neighbours:
                    difference = left ^ right
                    # Exactly one differing fixed position, and both cubes fix
                    # it, is the classical combination rule.
                    if difference & (difference - 1) or not difference & mask:
                        continue
                    used.add((mask, left))
                    used.add((mask, right))
                    reduced = mask & ~difference
                    next_round.add((reduced, left & reduced))

        primes.update(cube for cube in current if cube not in used)
        current = next_round

    def covers(cube: tuple[int, int], minterm: int) -> bool:
        mask, value = cube
        return minterm & mask == value

    # A prime built only from don't-cares cannot cover any required minterm.
    useful = [cube for cube in primes if any(covers(cube, m) for m in on_set)]
    return tuple(
        sorted(
            (_to_implicant(cube, width, on_set | dont_cares) for cube in useful),
            key=lambda item: (
                item.literals,
                tuple(2 if bit is None else bit for bit in item.pattern),
            ),
        )
    )


def _to_implicant(cube: tuple[int, int], width: int, universe: set[int]) -> Implicant:
    """Convert an internal (mask, value) cube to the public implicant type."""
    mask, value = cube
    pattern = tuple(
        None if not mask >> shift & 1 else value >> shift & 1 for shift in reversed(range(width))
    )
    origins = frozenset(minterm for minterm in universe if minterm & mask == value)
    return Implicant(pattern, origins)


def exact_minimum_covers(
    primes: tuple[Implicant, ...],
    on_set: set[int],
    *,
    max_solutions: int = 256,
) -> tuple[BooleanSolution, ...]:
    """Solve the prime-implicant chart exactly by branch-and-bound.

    Optimisation is lexicographic: first minimise the number of implicants,
    then the total number of literals. All tied minimal covers are returned up
    to ``max_solutions``.

    Three exactness-preserving reductions keep the search tractable:

    1. **Essential primes.** A minterm covered by exactly one prime forces that
       prime into every cover, so essentials are selected up front rather than
       rediscovered on every branch.
    2. **Independent-set lower bound.** Uncovered minterms whose candidate
       primes are pairwise disjoint each require a distinct further prime, which
       bounds the cost of any completion from below.
    3. **State memoisation.** Reaching the same set of uncovered minterms at a
       strictly worse cost can never yield a better or tied cover, because the
       completions available from a state depend only on that state.

    Parameters
    ----------
    primes : tuple of Implicant
        Candidate prime implicants, as produced by :func:`prime_implicants`.
    on_set : set of int
        Minterms that must be covered.
    max_solutions : int, default 256
        Upper bound on the number of tied minimal covers returned.

    Returns
    -------
    tuple of BooleanSolution
        Every returned cover has identical, provably minimal cost.

    Raises
    ------
    RuntimeError
        If some minterm of ``on_set`` is covered by no supplied prime.
    """
    if not on_set:
        return (BooleanSolution(()),)
    covered = [frozenset(m for m in on_set if prime.covers(m)) for prime in primes]
    literals = [prime.literals for prime in primes]
    choices = solve_minimum_cover(covered, literals, on_set, max_solutions=max_solutions)
    return tuple(BooleanSolution(tuple(primes[i] for i in indices)) for indices in choices)


def solve_minimum_cover(
    covered: Sequence[frozenset[int]],
    literals: Sequence[int],
    on_set: set[int],
    *,
    max_solutions: int = 256,
) -> tuple[tuple[int, ...], ...]:
    """Solve a covering problem exactly, independently of what is being covered.

    The exactness guarantee lives here, so every minimiser in the package —
    binary and multi-value alike — shares one verified implementation rather
    than repeating the search.

    Parameters
    ----------
    covered : sequence of frozenset of int
        For each candidate, the elements of ``on_set`` it covers.
    literals : sequence of int
        Cost of each candidate, used as the secondary objective.
    on_set : set of int
        Elements that must be covered.
    max_solutions : int, default 256
        Upper bound on the number of tied minimum covers returned.

    Returns
    -------
    tuple of tuple of int
        Candidate indices, one tuple per tied minimum cover.

    Raises
    ------
    RuntimeError
        If some element is covered by no candidate.
    """
    cover_map = {m: tuple(i for i, reach in enumerate(covered) if m in reach) for m in on_set}
    if any(not options for options in cover_map.values()):
        raise RuntimeError("Prime-implicant chart cannot cover every positive row.")

    # A minterm with a single candidate forces that candidate into every cover.
    essential = frozenset(options[0] for options in cover_map.values() if len(options) == 1)
    start_uncovered = (
        frozenset(on_set).difference(*(covered[i] for i in essential))
        if essential
        else frozenset(on_set)
    )

    def lower_bound(uncovered: frozenset[int]) -> int:
        """Return a lower bound on the number of further candidates required."""
        blocked: set[int] = set()
        bound = 0
        for minterm in sorted(uncovered, key=lambda m: len(cover_map[m])):
            options = cover_map[minterm]
            if blocked.isdisjoint(options):
                bound += 1
                blocked.update(options)
        return bound

    best_cost: tuple[int, int] | None = None
    best: set[tuple[int, ...]] = set()
    seen: dict[frozenset[int], tuple[int, int]] = {}

    def search(chosen: frozenset[int], uncovered: frozenset[int]) -> None:
        nonlocal best_cost
        current_cost = (len(chosen), sum(literals[i] for i in chosen))
        if not uncovered:
            if best_cost is None or current_cost < best_cost:
                best_cost = current_cost
                best.clear()
            if current_cost == best_cost and len(best) < max_solutions:
                best.add(tuple(sorted(chosen)))
            return
        if best_cost is not None:
            if current_cost >= best_cost:
                return
            # Any completion needs at least `lower_bound` further candidates,
            # and candidates never reduce the cost.
            if (current_cost[0] + lower_bound(uncovered), current_cost[1]) > best_cost:
                return
        previous = seen.get(uncovered)
        if previous is not None and current_cost > previous:
            return
        if previous is None or current_cost < previous:
            seen[uncovered] = current_cost

        # Branch on the element with the fewest candidates: every cover must
        # contain one of them, so this is a complete and narrow branching rule.
        target = min(uncovered, key=lambda m: len(cover_map[m]))
        ordered = sorted(
            cover_map[target],
            key=lambda i: (-len(covered[i] & uncovered), literals[i], i),
        )
        for index in ordered:
            search(chosen | {index}, uncovered - covered[index])

    search(essential, start_uncovered)
    return tuple(sorted(best))


def minimize(
    on_set: set[int],
    *,
    dont_cares: set[int] | None = None,
    width: int,
    max_solutions: int = 256,
    complexity_guard: bool = True,
) -> tuple[BooleanSolution, ...]:
    """Return all exact minimum Boolean covers for the specified truth table.

    Parameters
    ----------
    on_set : set of int
        Minterms that must be covered.
    dont_cares : set of int, optional
        Logical remainders, usable but not required.
    width : int
        Number of conditions.
    max_solutions : int, default 256
        Upper bound on the number of tied minimum covers returned.
    complexity_guard : bool, default True
        Warn with :class:`~setqca.minimize.MinimizationComplexityWarning` when
        the chart looks likely to be slow. The result is exact either way; the
        warning arrives before the expensive phase rather than after it.

    Returns
    -------
    tuple of BooleanSolution
        Every cover of provably minimal cost.
    """
    dc = set() if dont_cares is None else set(dont_cares)
    required = set(on_set)
    primes = prime_implicants(required, dc, width)
    if complexity_guard:
        warn_if_complex(
            estimate_complexity(
                width=width,
                required=len(required),
                dont_cares=len(dc),
                primes=len(primes),
            )
        )
    return exact_minimum_covers(primes, required, max_solutions=max_solutions)
