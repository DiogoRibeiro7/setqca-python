"""Tests for the exactness-preserving reductions in the chart solver.

The solver applies essential-prime selection, an independent-set lower bound and
state memoisation. Each is only sound if it never changes the answer, so these
tests pin the answer rather than the mechanism.
"""

from __future__ import annotations

from itertools import combinations

import pytest

from setqca.minimize import minimize, prime_implicants
from setqca.minimize.implicant import Implicant
from setqca.minimize.qmc import exact_minimum_covers


def _cost(solution) -> tuple[int, int]:
    return (len(solution.implicants), solution.literal_count)


def test_an_essential_prime_appears_in_every_returned_cover() -> None:
    """Minterm 0 is reachable only through ~A*~B*~C, so every cover contains it."""
    on_set = {0, 3, 5, 6}
    solutions = minimize(on_set, width=3)
    for solution in solutions:
        assert any(implicant.covers(0) for implicant in solution.implicants)


def test_a_chart_of_only_essentials_yields_exactly_one_cover() -> None:
    solutions = minimize({0, 7}, width=3)
    assert len(solutions) == 1
    assert _cost(solutions[0]) == (2, 6)


# The textbook cyclic chart: every minterm of f = Sm(0,1,2,5,6,7) is covered by
# exactly two primes and every prime covers exactly two minterms, so no prime is
# essential and the minimum cover is genuinely ambiguous.
CYCLIC_ON_SET = {0, 1, 2, 5, 6, 7}


def test_cyclic_chart_has_no_essentials_and_returns_all_tied_covers() -> None:
    primes = prime_implicants(CYCLIC_ON_SET, set(), width=3)
    coverage_counts = [sum(1 for prime in primes if prime.covers(m)) for m in sorted(CYCLIC_ON_SET)]
    assert coverage_counts == [2, 2, 2, 2, 2, 2], "no minterm may have a single candidate"

    solutions = minimize(CYCLIC_ON_SET, width=3)
    costs = {_cost(solution) for solution in solutions}
    assert costs == {(3, 6)}, "the minimum is three primes of two literals each"
    assert len(solutions) == 2, "a cyclic chart is genuinely ambiguous"

    for solution in solutions:
        covered = {m for m in range(8) for i in solution.implicants if i.covers(m)}
        assert covered >= CYCLIC_ON_SET


def test_max_solutions_bounds_the_returned_ties() -> None:
    capped = minimize(CYCLIC_ON_SET, width=3, max_solutions=1)
    full = minimize(CYCLIC_ON_SET, width=3, max_solutions=256)
    assert len(capped) == 1
    assert len(full) == 2
    assert _cost(capped[0]) == _cost(full[0]), "capping must not degrade the cost"


@pytest.mark.parametrize("width", [3, 4])
def test_solver_matches_exhaustive_search_over_all_small_tables(width: int) -> None:
    """Exhaustively verify minimality on every table of a given small width."""
    universe = list(range(2**width))
    for size in (1, 2, width):
        for on_tuple in combinations(universe, size):
            on_set = set(on_tuple)
            solutions = minimize(on_set, width=width)
            obtained = _cost(solutions[0])

            # Brute force: enumerate every cube legal for this table.
            legal: list[tuple[frozenset[int], int]] = []
            primes = prime_implicants(on_set, set(), width)
            for implicant in primes:
                covered = frozenset(m for m in universe if implicant.covers(m))
                legal.append((covered, implicant.literals))

            expected: tuple[int, int] | None = None
            for count in range(1, len(legal) + 1):
                candidates = [
                    (count, sum(literals for _, literals in choice))
                    for choice in combinations(legal, count)
                    if on_set <= set().union(*(covered for covered, _ in choice))
                ]
                if candidates:
                    expected = min(candidates)
                    break

            assert obtained == expected, f"width={width} on_set={sorted(on_set)}"


def test_reductions_preserve_the_cover_of_a_hand_checked_chart() -> None:
    """A four-variable chart whose minimum is known by hand."""
    # f = ~A~B~C~D + ~A~B~CD + ~A~BC~D + ~A~BCD  ->  ~A*~B
    on_set = {0, 1, 2, 3}
    solutions = minimize(on_set, width=4)
    assert len(solutions) == 1
    assert solutions[0].as_expression(("A", "B", "C", "D")) == "~A*~B"


def test_essential_selection_survives_a_chart_with_don_t_cares() -> None:
    on_set = {6, 7}
    solutions = minimize(on_set, dont_cares={4, 5}, width=3)
    assert len(solutions) == 1
    assert solutions[0].as_expression(("A", "B", "C")) == "A"


def test_memoisation_prunes_a_costlier_route_to_the_same_remainder() -> None:
    """Two primes covering the same rows must not both be expanded.

    ``wide`` and ``narrow`` cover exactly the same positive rows at different
    literal costs, so whichever is tried second reaches an already-seen set of
    uncovered rows at a strictly worse cost and is abandoned. The minimum is
    unaffected, which is the property that makes the pruning sound.
    """
    on_set = {0, 1, 5, 7}
    wide = Implicant((0, None, None), frozenset({0, 1}))  # 1 literal
    narrow = Implicant((0, 0, None), frozenset({0, 1}))  # 2 literals, same rows
    pair = Implicant((1, None, 1), frozenset({5, 7}))  # 2 literals
    single_five = Implicant((1, 0, 1), frozenset({5}))  # 3 literals
    single_seven = Implicant((1, 1, 1), frozenset({7}))  # 3 literals

    solutions = exact_minimum_covers((wide, narrow, pair, single_five, single_seven), on_set)

    assert len(solutions) == 1
    assert _cost(solutions[0]) == (2, 3)
    assert set(solutions[0].implicants) == {wide, pair}


def test_exact_minimum_covers_accepts_a_prefiltered_prime_set() -> None:
    """The chart solver is usable independently of prime generation."""
    on_set = {6, 7}
    primes = prime_implicants(on_set, set(), width=3)
    solutions = exact_minimum_covers(primes, on_set)
    assert solutions[0].as_expression(("A", "B", "C")) == "A*B"
