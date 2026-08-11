"""Tests for the minimisation complexity guard."""

from __future__ import annotations

import warnings

import pytest

from setqca.minimize import complexity, minimize
from setqca.minimize.complexity import (
    HIGH_PRIMES,
    MODERATE_PRIMES,
    ComplexityEstimate,
    MinimizationComplexityWarning,
    estimate_complexity,
    warn_if_complex,
)


def _estimate(primes: int) -> ComplexityEstimate:
    return estimate_complexity(width=8, required=40, dont_cares=10, primes=primes)


class TestEstimate:
    def test_counts_are_reported_back(self) -> None:
        estimate = estimate_complexity(width=6, required=12, dont_cares=30, primes=9)
        assert estimate.width == 6
        assert estimate.required == 12
        assert estimate.dont_cares == 30
        assert estimate.primes == 9
        assert estimate.universe == 64
        assert estimate.chart_cells == 9 * 12

    @pytest.mark.parametrize(
        ("primes", "level"),
        [
            (1, "low"),
            (MODERATE_PRIMES, "low"),
            (MODERATE_PRIMES + 1, "moderate"),
            (HIGH_PRIMES, "moderate"),
            (HIGH_PRIMES + 1, "high"),
        ],
    )
    def test_the_band_follows_the_prime_count(self, primes: int, level: str) -> None:
        assert _estimate(primes).level == level

    def test_only_the_high_band_warrants_a_warning(self) -> None:
        assert not _estimate(MODERATE_PRIMES).should_warn
        assert not _estimate(HIGH_PRIMES).should_warn
        assert _estimate(HIGH_PRIMES + 1).should_warn

    def test_the_message_says_the_result_is_still_exact(self) -> None:
        """The guard must not read as a threat to correctness."""
        message = _estimate(HIGH_PRIMES + 1).message
        assert "still be exact" in message
        assert "fewer conditions" in message

    def test_the_estimate_renders_readably(self) -> None:
        assert "level=high" in str(_estimate(HIGH_PRIMES + 1))


class TestWarning:
    def test_an_easy_problem_is_silent(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert warn_if_complex(_estimate(4)) is False

    def test_a_hard_problem_warns(self) -> None:
        with pytest.warns(MinimizationComplexityWarning, match="worst-case exponential"):
            assert warn_if_complex(_estimate(HIGH_PRIMES + 1)) is True

    def test_the_warning_is_a_user_warning(self) -> None:
        """So that ordinary warning filters reach it."""
        assert issubclass(MinimizationComplexityWarning, UserWarning)


class TestIntegration:
    def test_ordinary_problems_do_not_warn(self) -> None:
        """The suite runs with warnings as errors, so this is enforced throughout."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            minimize({6, 7}, dont_cares={4, 5}, width=3)

    def test_the_guard_fires_before_the_expensive_phase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lowering the threshold proves the wiring without paying the cost.

        A chart genuinely large enough to trip the real threshold is, by
        construction, one that takes a long time to solve — so the test lowers
        the bar instead of building such a chart.
        """
        monkeypatch.setattr(complexity, "HIGH_PRIMES", 0)
        with pytest.warns(MinimizationComplexityWarning, match="still be exact"):
            minimize({6, 7}, width=3)

    def test_the_guard_can_be_switched_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(complexity, "HIGH_PRIMES", 0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            minimize({6, 7}, width=3, complexity_guard=False)

    def test_warning_does_not_change_the_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        quiet = minimize({6, 7}, dont_cares={4, 5}, width=3, complexity_guard=False)
        monkeypatch.setattr(complexity, "HIGH_PRIMES", 0)
        with pytest.warns(MinimizationComplexityWarning):
            loud = minimize({6, 7}, dont_cares={4, 5}, width=3)
        assert [s.implicants for s in loud] == [s.implicants for s in quiet]

    def test_the_warning_is_catchable_from_the_top_level_namespace(self) -> None:
        """Callers need the class to filter on, so it is exported alongside the API."""
        import setqca

        assert setqca.MinimizationComplexityWarning is MinimizationComplexityWarning
        assert "MinimizationComplexityWarning" in setqca.__all__
