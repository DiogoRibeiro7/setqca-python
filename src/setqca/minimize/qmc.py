"""Exact classical Quine-McCluskey minimisation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .implicant import Implicant, minterm_to_implicant


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
    """Generate all prime implicants exactly using classical QMC."""
    if on_set & dont_cares:
        raise ValueError("on_set and dont_cares must be disjoint.")
    universe = on_set | dont_cares
    if not universe:
        return ()

    current = {minterm_to_implicant(value, width) for value in universe}
    primes: set[Implicant] = set()

    while current:
        grouped: dict[int, list[Implicant]] = defaultdict(list)
        for implicant in current:
            grouped[sum(bit == 1 for bit in implicant.pattern)].append(implicant)

        used: set[Implicant] = set()
        next_round: dict[tuple[int | None, ...], Implicant] = {}
        for ones in sorted(grouped):
            for left in grouped[ones]:
                for right in grouped.get(ones + 1, []):
                    combined = left.combine(right)
                    if combined is None:
                        continue
                    used.add(left)
                    used.add(right)
                    previous = next_round.get(combined.pattern)
                    if previous is None:
                        next_round[combined.pattern] = combined
                    else:
                        next_round[combined.pattern] = Implicant(
                            combined.pattern, previous.origins | combined.origins
                        )

        primes.update(item for item in current if item not in used)
        current = set(next_round.values())

    # A prime built only from don't-cares cannot cover any required minterm.
    useful = [item for item in primes if any(item.covers(m) for m in on_set)]
    return tuple(
        sorted(
            useful,
            key=lambda item: (
                item.literals,
                tuple(2 if bit is None else bit for bit in item.pattern),
            ),
        )
    )


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
) -> tuple[BooleanSolution, ...]:
    """Return all exact minimum Boolean covers for the specified truth table."""
    dc = set() if dont_cares is None else set(dont_cares)
    primes = prime_implicants(set(on_set), dc, width)
    return exact_minimum_covers(primes, set(on_set), max_solutions=max_solutions)
