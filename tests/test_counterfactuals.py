"""Tests for directional expectations and counterfactual classification."""

from __future__ import annotations

import pandas as pd
import pytest

from setqca import FSQCA
from setqca.counterfactuals import (
    DirectionalExpectation,
    coerce_expectations,
    is_easy_counterfactual,
)


class TestDirectionalExpectation:
    @pytest.mark.parametrize("value", ["+", "1", "present", "positive", 1, "PRESENT"])
    def test_positive_spellings(self, value: str | int) -> None:
        assert DirectionalExpectation.coerce(value) is DirectionalExpectation.POSITIVE

    @pytest.mark.parametrize("value", ["-", "absent", "negative", 0])
    def test_negative_spellings(self, value: str | int) -> None:
        assert DirectionalExpectation.coerce(value) is DirectionalExpectation.NEGATIVE

    @pytest.mark.parametrize("value", ["0", "", "unspecified"])
    def test_unspecified_spellings(self, value: str) -> None:
        assert DirectionalExpectation.coerce(value) is DirectionalExpectation.UNSPECIFIED

    def test_the_enum_passes_through(self) -> None:
        assert (
            DirectionalExpectation.coerce(DirectionalExpectation.POSITIVE)
            is DirectionalExpectation.POSITIVE
        )

    @pytest.mark.parametrize("value", ["up", "yes", 7, -1])
    def test_unknown_values_are_rejected(self, value: str | int) -> None:
        with pytest.raises(ValueError, match="directional expectation"):
            DirectionalExpectation.coerce(value)

    @pytest.mark.parametrize("value", [True, False])
    def test_booleans_are_rejected_as_ambiguous(self, value: bool) -> None:
        """`True` could plausibly mean present or "has an expectation"."""
        with pytest.raises(ValueError, match="Ambiguous"):
            DirectionalExpectation.coerce(value)

    def test_contributing_state(self) -> None:
        assert DirectionalExpectation.POSITIVE.contributing_state == 1
        assert DirectionalExpectation.NEGATIVE.contributing_state == 0
        assert DirectionalExpectation.UNSPECIFIED.contributing_state is None

    def test_expectations_must_name_known_conditions(self) -> None:
        with pytest.raises(KeyError, match="unknown condition"):
            coerce_expectations({"Z": "+"}, ("A", "B"))


class TestEasyCounterfactuals:
    conditions = ("A", "B")

    def test_a_remainder_reached_by_adding_a_helpful_condition_is_easy(self) -> None:
        # Observed sufficient: ~A*B (minterm 1). Remainder A*B (minterm 3) adds A,
        # which is expected to contribute.
        expectations = {"A": DirectionalExpectation.POSITIVE}
        assert is_easy_counterfactual(3, frozenset({1}), expectations, self.conditions)

    def test_a_remainder_reached_by_removing_a_helpful_condition_is_difficult(self) -> None:
        # Observed sufficient: A*B (3). Remainder ~A*B (1) removes A.
        expectations = {"A": DirectionalExpectation.POSITIVE}
        assert not is_easy_counterfactual(1, frozenset({3}), expectations, self.conditions)

    def test_a_negative_expectation_reverses_the_direction(self) -> None:
        expectations = {"A": DirectionalExpectation.NEGATIVE}
        assert is_easy_counterfactual(1, frozenset({3}), expectations, self.conditions)
        assert not is_easy_counterfactual(3, frozenset({1}), expectations, self.conditions)

    def test_a_difference_on_an_unspecified_condition_cannot_be_justified(self) -> None:
        """Without a stated expectation there is no ground to call the leap easy."""
        expectations = {"A": DirectionalExpectation.UNSPECIFIED}
        assert not is_easy_counterfactual(3, frozenset({1}), expectations, self.conditions)

    def test_a_condition_with_no_expectation_at_all_behaves_as_unspecified(self) -> None:
        assert not is_easy_counterfactual(3, frozenset({1}), {}, self.conditions)

    def test_every_differing_condition_must_be_justified(self) -> None:
        # Remainder 3 (A*B) vs observed 0 (~A*~B) differs on both conditions;
        # only A has a supporting expectation, so the leap is not easy.
        expectations = {"A": DirectionalExpectation.POSITIVE}
        assert not is_easy_counterfactual(3, frozenset({0}), expectations, self.conditions)
        both = {
            "A": DirectionalExpectation.POSITIVE,
            "B": DirectionalExpectation.POSITIVE,
        }
        assert is_easy_counterfactual(3, frozenset({0}), both, self.conditions)


class TestIntermediateSolutions:
    """The intermediate solution must sit between the other two families."""

    frame = pd.DataFrame(
        {
            "A": [0.9, 0.9, 0.1, 0.1],
            "B": [0.9, 0.1, 0.9, 0.1],
            "Y": [0.9, 0.85, 0.2, 0.1],
        }
    )

    def _fit(self, **expectations: str) -> object:
        model = FSQCA(consistency=0.8, directional_expectations=dict(expectations))
        return model.fit(self.frame, outcome="Y", conditions=["A", "B"])

    def test_intermediate_is_no_more_parsimonious_than_the_parsimonious_solution(self) -> None:
        result = self._fit(A="+", B="+")
        assert result.intermediate is not None
        intermediate = min(s.boolean.literal_count for s in result.intermediate)
        parsimonious = min(s.boolean.literal_count for s in result.parsimonious)
        conservative = min(s.boolean.literal_count for s in result.conservative)
        assert parsimonious <= intermediate <= conservative

    def test_difficult_counterfactuals_are_refused(self) -> None:
        result = self._fit(A="+", B="+")
        analysis = result.counterfactuals
        assert analysis is not None
        assert analysis.easy | analysis.difficult == analysis.simplifying_assumptions
        assert not analysis.easy & analysis.difficult

    def test_the_analysis_reports_the_expectations_it_used(self) -> None:
        result = self._fit(A="+", B="-")
        analysis = result.counterfactuals
        assert analysis is not None
        assert analysis.expectations["A"] is DirectionalExpectation.POSITIVE
        assert analysis.expectations["B"] is DirectionalExpectation.NEGATIVE
        assert "A+" in str(analysis)

    def test_no_expectations_means_no_intermediate_solution(self) -> None:
        result = FSQCA(consistency=0.8).fit(self.frame, outcome="Y", conditions=["A", "B"])
        assert result.intermediate is None
        assert result.counterfactuals is None

    def test_simplifying_assumptions_are_remainders_only(self) -> None:
        result = self._fit(A="+", B="+")
        analysis = result.counterfactuals
        assert analysis is not None
        assert analysis.simplifying_assumptions <= result.truth_table.remainder_minterms
