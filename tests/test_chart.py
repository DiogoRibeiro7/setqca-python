"""Tests for the prime-implicant chart and its diagnostics."""

from __future__ import annotations

import pytest

from setqca.minimize import (
    build_chart,
    minimize,
    minimize_chart,
)

ABC = ("A", "B", "C")

# The textbook cyclic chart: no essential primes, two tied minimum covers.
CYCLIC = {0, 1, 2, 5, 6, 7}


class TestChartStructure:
    def test_a_chart_lists_every_prime_and_what_it_covers(self) -> None:
        chart = build_chart({6, 7}, width=3)
        assert chart.on_set == {6, 7}
        assert len(chart.primes) == 1
        assert chart.primes[0].covered == {6, 7}
        assert chart.primes[0].as_expression(ABC) == "A*B"

    def test_chart_indices_are_stable_and_zero_based(self) -> None:
        chart = build_chart(CYCLIC, width=3)
        assert [prime.index for prime in chart.primes] == list(range(len(chart.primes)))

    def test_candidates_for_a_row_are_the_primes_covering_it(self) -> None:
        chart = build_chart(CYCLIC, width=3)
        for minterm in CYCLIC:
            candidates = chart.candidates_for(minterm)
            assert candidates
            assert all(prime.covers(minterm) for prime in candidates)

    def test_the_chart_exports_as_a_boolean_table(self) -> None:
        chart = build_chart({6, 7}, width=3)
        frame = chart.to_frame()
        assert list(frame.index) == [6, 7]
        assert list(frame.columns) == ["P0"]
        assert frame["P0"].all()


class TestEssentialPrimes:
    def test_a_row_with_one_candidate_forces_its_prime(self) -> None:
        # Row 0 is reachable only through ~A*~B*~C.
        chart = build_chart({0, 3, 5, 6}, width=3)
        essential = chart.essential
        assert essential
        assert any(prime.covers(0) for prime in essential)

    def test_every_minimum_cover_contains_every_essential_prime(self) -> None:
        result = minimize_chart({0, 3, 5, 6}, width=3)
        essential = {prime.index for prime in result.chart.essential}
        for cover in result.covers:
            assert essential <= {prime.index for prime in cover.primes}

    def test_a_cyclic_chart_has_no_essential_primes(self) -> None:
        chart = build_chart(CYCLIC, width=3)
        assert chart.essential == ()

    def test_a_cover_explains_which_terms_were_forced(self) -> None:
        result = minimize_chart({6, 7}, dont_cares={4, 5}, width=3)
        explanation = result.covers[0].explain(ABC)
        assert "essential" in explanation


class TestDominatedPrimes:
    def test_a_prime_covering_strictly_fewer_rows_is_dominated(self) -> None:
        """~B covers only row 0; ~A covers rows 0 and 1 for the same one literal."""
        chart = build_chart({0, 1}, dont_cares={2}, width=2)
        rendered = {prime.as_expression(("A", "B")): prime for prime in chart.primes}
        assert rendered["~A"].covered == {0, 1}
        assert rendered["~B"].covered == {0}
        assert [prime.as_expression(("A", "B")) for prime in chart.dominated] == ["~B"]

    def test_primes_that_are_merely_tied_are_interchangeable_not_dominated(self) -> None:
        """Both primes cover exactly row 0 at two literals, so each heads a cover.

        Calling one dominated would be a tie-break, and this package reports
        every tied minimum rather than choosing between them.
        """
        chart = build_chart({0}, dont_cares={1, 2}, width=3)
        assert len(chart.primes) == 2
        assert all(prime.covered == {0} for prime in chart.primes)
        assert chart.dominated == ()

    def test_dominated_primes_never_appear_in_any_minimum_cover(self) -> None:
        for on_set, dont_cares, width in [
            ({0, 1}, {2}, 2),
            ({0}, {1, 2}, 3),
            ({0, 7}, {1, 2, 4}, 3),
            (CYCLIC, None, 3),
        ]:
            result = minimize_chart(on_set, dont_cares=dont_cares, width=width)
            dominated = {prime.index for prime in result.chart.dominated}
            for cover in result.covers:
                chosen = {prime.index for prime in cover.primes}
                assert not dominated & chosen, (
                    f"a dominated prime appeared in a minimum cover for {sorted(on_set)}"
                )

    def test_every_tied_alternative_actually_heads_a_cover(self) -> None:
        """If two primes are interchangeable, both must show up across the ties."""
        result = minimize_chart({0}, dont_cares={1, 2}, width=3)
        appearing = {prime.index for cover in result.covers for prime in cover.primes}
        assert appearing == {prime.index for prime in result.chart.primes}

    def test_a_cyclic_chart_has_no_domination(self) -> None:
        assert build_chart(CYCLIC, width=3).dominated == ()


class TestAmbiguity:
    def test_a_unique_minimum_is_reported_as_unambiguous(self) -> None:
        result = minimize_chart({6, 7}, width=3)
        assert not result.ambiguous
        assert len(result.covers) == 1

    def test_a_cyclic_chart_is_reported_as_ambiguous(self) -> None:
        result = minimize_chart(CYCLIC, width=3)
        assert result.ambiguous
        assert len(result.covers) == 2
        assert len({cover.cost for cover in result.covers}) == 1

    def test_truncation_is_reported_rather_than_hidden(self) -> None:
        result = minimize_chart(CYCLIC, width=3, max_solutions=1)
        assert len(result.covers) == 1
        assert result.truncated is True

    def test_an_untruncated_result_says_so(self) -> None:
        result = minimize_chart(CYCLIC, width=3, max_solutions=256)
        assert result.truncated is False


class TestRemaindersAndFailure:
    def test_remainders_enlarge_the_primes_without_being_required(self) -> None:
        chart = build_chart({6, 7}, dont_cares={4, 5}, width=3)
        assert chart.on_set == {6, 7}, "don't-cares are not required rows"
        assert any(prime.as_expression(ABC) == "A" for prime in chart.primes)

    def test_an_uncoverable_configuration_is_detected(self) -> None:
        chart = build_chart(set(), width=3)
        assert chart.uncoverable == frozenset()

    def test_an_uncoverable_chart_cannot_be_solved(self) -> None:
        from setqca.minimize import exact_minimum_covers

        with pytest.raises(RuntimeError, match="cannot cover every positive row"):
            exact_minimum_covers((), {1})


class TestExplanations:
    def test_a_forced_row_is_explained_as_forcing(self) -> None:
        chart = build_chart({0, 3, 5, 6}, width=3)
        assert "essential" in chart.explain(0, ABC)

    def test_an_interchangeable_row_lists_its_alternatives(self) -> None:
        chart = build_chart(CYCLIC, width=3)
        message = chart.explain(0, ABC)
        assert "can be covered by any of" in message

    def test_a_row_outside_the_on_set_is_reported_as_such(self) -> None:
        chart = build_chart({6, 7}, width=3)
        assert "not required" in chart.explain(0, ABC)

    def test_an_uncovered_row_is_reported_as_unsolvable(self) -> None:
        chart = build_chart({6, 7}, width=3)
        stripped = type(chart)(on_set=frozenset({6, 7, 0}), primes=chart.primes)
        assert "cannot be solved" in stripped.explain(0, ABC)
        assert stripped.uncoverable == frozenset({0})

    def test_the_summary_reports_the_shape_of_the_problem(self) -> None:
        summary = minimize_chart(CYCLIC, width=3).summary(ABC)
        assert "Configurations to cover: 6" in summary
        assert "Essential primes: 0" in summary
        assert "Minimum covers: 2" in summary


class TestAgreementWithTheFlatApi:
    """The chart must not change the answer, only explain it."""

    @pytest.mark.parametrize(
        ("on_set", "dont_cares"),
        [
            ({6, 7}, None),
            ({6, 7}, {4, 5}),
            (CYCLIC, None),
            ({0, 3, 5, 6}, None),
            ({1, 3, 5, 7}, {0}),
        ],
    )
    def test_covers_match_the_plain_minimiser(
        self, on_set: set[int], dont_cares: set[int] | None
    ) -> None:
        flat = minimize(on_set, dont_cares=dont_cares, width=3)
        result = minimize_chart(on_set, dont_cares=dont_cares, width=3)

        assert {solution.implicants for solution in flat} == {
            cover.as_boolean_solution().implicants for cover in result.covers
        }
        assert result.cost == (len(flat[0].implicants), flat[0].literal_count)

    def test_the_result_converts_back_to_plain_solutions(self) -> None:
        result = minimize_chart(CYCLIC, width=3)
        solutions = result.as_boolean_solutions()
        assert len(solutions) == len(result.covers)
        assert {s.implicants for s in solutions} == {
            s.implicants for s in minimize(CYCLIC, width=3)
        }
