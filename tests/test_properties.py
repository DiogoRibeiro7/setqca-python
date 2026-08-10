"""Property-based tests for the set-theoretic and Boolean cores.

These check mathematical invariants that must hold for every admissible input,
rather than behaviour on hand-picked examples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from setqca import Condition, calibrate_direct, minimize, necessity, sufficiency
from setqca.minimize import prime_implicants

memberships = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=40,
)


@st.composite
def membership_pairs(draw: st.DrawFn) -> tuple[list[float], list[float]]:
    """Draw two membership vectors of equal length."""
    size = draw(st.integers(min_value=1, max_value=40))
    unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    return (
        draw(st.lists(unit, min_size=size, max_size=size)),
        draw(st.lists(unit, min_size=size, max_size=size)),
    )


@given(membership_pairs())
def test_fit_parameters_are_bounded_probabilities(pair: tuple[list[float], list[float]]) -> None:
    x, y = pair
    suf = sufficiency(x, y)
    nec = necessity(x, y)
    for value in (suf.consistency, suf.coverage, suf.pri, nec.consistency, nec.coverage, nec.ron):
        assert 0.0 <= value <= 1.0


@given(membership_pairs())
def test_sufficiency_and_necessity_are_duals(pair: tuple[list[float], list[float]]) -> None:
    """Necessity of X for Y is sufficiency of Y for X with roles exchanged."""
    x, y = pair
    assert necessity(x, y).consistency == pytest.approx(sufficiency(y, x).consistency)
    assert necessity(x, y).coverage == pytest.approx(sufficiency(y, x).coverage)


@given(membership_pairs())
def test_pri_never_exceeds_consistency(pair: tuple[list[float], list[float]]) -> None:
    x, y = pair
    fit = sufficiency(x, y)
    assert fit.pri <= fit.consistency + 1e-12


@given(x=memberships)
def test_perfect_subset_relation_is_perfectly_consistent(x: list[float]) -> None:
    """X <= Y elementwise implies sufficiency consistency of exactly one."""
    assume(sum(x) > 0.0)
    y = [min(1.0, value + (1.0 - value) / 2.0) for value in x]
    assert sufficiency(x, y).consistency == pytest.approx(1.0)


@given(membership_pairs())
def test_de_morgan_holds_for_the_fuzzy_operators(pair: tuple[list[float], list[float]]) -> None:
    a, b = pair
    data = pd.DataFrame({"A": a, "B": b})
    left = (~(Condition("A") & Condition("B"))).evaluate(data)
    right = (~Condition("A") | ~Condition("B")).evaluate(data)
    assert left == pytest.approx(right)


@given(membership_pairs())
def test_intersection_is_bounded_by_its_operands(pair: tuple[list[float], list[float]]) -> None:
    a, b = pair
    data = pd.DataFrame({"A": a, "B": b})
    intersection = (Condition("A") & Condition("B")).evaluate(data)
    union = (Condition("A") | Condition("B")).evaluate(data)
    assert np.all(intersection <= union + 1e-12)
    assert np.all(intersection <= np.asarray(a) + 1e-12)
    assert np.all(union >= np.asarray(b) - 1e-12)


@given(
    values=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50,
    )
)
def test_direct_calibration_is_monotone_and_bounded(values: list[float]) -> None:
    calibrated = calibrate_direct(values, full_out=0.0, crossover=50.0, full_in=100.0)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    order = np.argsort(np.asarray(values), kind="stable")
    assert np.all(np.diff(calibrated[order]) >= -1e-9)


@settings(max_examples=50, deadline=None)
@given(
    width=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_minimal_covers_are_valid_and_tied_on_cost(width: int, seed: int) -> None:
    """Every returned cover must cover the on-set, avoid the off-set, and tie on cost."""
    rng = np.random.default_rng(seed)
    universe = set(range(2**width))
    on_set = {m for m in universe if rng.random() < 0.5}
    assume(bool(on_set))
    dont_cares = {m for m in universe - on_set if rng.random() < 0.3}
    off_set = universe - on_set - dont_cares

    solutions = minimize(on_set, dont_cares=dont_cares, width=width)
    assert solutions
    costs = {(len(s.implicants), s.literal_count) for s in solutions}
    assert len(costs) == 1, "all returned solutions must share the minimal cost"

    for solution in solutions:
        covered = {m for m in universe for i in solution.implicants if i.covers(m)}
        assert on_set <= covered, "solution fails to cover a positive row"
        assert not covered & off_set, "solution covers a negative row"


@settings(max_examples=30, deadline=None)
@given(
    width=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_prime_implicants_are_mutually_irredundant(width: int, seed: int) -> None:
    """No prime implicant may be strictly contained in another."""
    rng = np.random.default_rng(seed)
    universe = set(range(2**width))
    on_set = {m for m in universe if rng.random() < 0.5}
    assume(bool(on_set))
    primes = prime_implicants(on_set, set(), width)

    for left in primes:
        for right in primes:
            if left is right:
                continue
            covered_left = {m for m in universe if left.covers(m)}
            covered_right = {m for m in universe if right.covers(m)}
            assert not covered_left < covered_right, "a prime implicant is not prime"
