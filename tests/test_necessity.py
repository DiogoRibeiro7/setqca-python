"""Tests for systematic necessity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from setqca import necessity_analysis
from setqca.analysis.necessity import NecessityAnalysis, NecessityCandidate

# B is present in every case, so it is a superset of anything and will show
# perfect necessity consistency while explaining nothing.
DATA = pd.DataFrame(
    {
        "A": [0.9, 0.8, 0.2, 0.1],
        "B": [1.0, 1.0, 1.0, 1.0],
        "C": [0.1, 0.2, 0.8, 0.9],
        "Y": [0.8, 0.7, 0.2, 0.1],
    }
)


def _by_name(analysis: NecessityAnalysis, expression: str) -> NecessityCandidate:
    return next(item for item in analysis.candidates if item.expression == expression)


class TestScreening:
    def test_every_condition_is_screened_in_both_directions(self) -> None:
        analysis = necessity_analysis(DATA, outcome="Y", conditions=["A", "C"])
        assert {item.expression for item in analysis.candidates} == {"A", "~A", "C", "~C"}

    def test_absence_can_be_switched_off(self) -> None:
        analysis = necessity_analysis(
            DATA, outcome="Y", conditions=["A", "C"], include_absence=False
        )
        assert {item.expression for item in analysis.candidates} == {"A", "C"}

    def test_a_superset_of_the_outcome_is_consistent(self) -> None:
        analysis = necessity_analysis(DATA, outcome="Y", conditions=["A"])
        assert _by_name(analysis, "A").fit.consistency == pytest.approx(1.0)

    def test_fit_values_match_the_standalone_metric(self) -> None:
        from setqca.metrics import necessity

        analysis = necessity_analysis(DATA, outcome="Y", conditions=["A"])
        direct = necessity(DATA["A"], DATA["Y"])
        candidate = _by_name(analysis, "A")
        assert candidate.fit.consistency == pytest.approx(direct.consistency)
        assert candidate.fit.coverage == pytest.approx(direct.coverage)
        assert candidate.fit.ron == pytest.approx(direct.ron)

    def test_crisp_data_is_the_binary_special_case(self) -> None:
        crisp = pd.DataFrame({"A": [1, 1, 0, 0], "Y": [1, 0, 0, 0]})
        analysis = necessity_analysis(crisp, outcome="Y", conditions=["A"])
        assert _by_name(analysis, "A").fit.consistency == pytest.approx(1.0)


class TestTrivialNecessity:
    def test_a_constant_condition_is_flagged_trivial_not_necessary(self) -> None:
        """B is present everywhere: perfect consistency, zero relevance."""
        analysis = necessity_analysis(DATA, outcome="Y", conditions=["A", "B"])
        constant = _by_name(analysis, "B")

        assert constant.fit.consistency == pytest.approx(1.0)
        assert constant.fit.ron == pytest.approx(0.0)
        assert constant.prevalence == pytest.approx(1.0)
        assert constant.consistent is True
        assert constant.relevant is False
        assert constant.trivial is True
        assert constant.necessary is False

    def test_trivial_candidates_are_excluded_from_the_necessary_list(self) -> None:
        analysis = necessity_analysis(DATA, outcome="Y", conditions=["A", "B"])
        assert "B" not in {item.expression for item in analysis.necessary}
        assert "B" in {item.expression for item in analysis.trivial}

    def test_a_relevant_superset_is_reported_as_necessary(self) -> None:
        analysis = necessity_analysis(DATA, outcome="Y", conditions=["A"])
        assert "A" in {item.expression for item in analysis.necessary}

    def test_the_relevance_threshold_is_adjustable(self) -> None:
        strict = necessity_analysis(DATA, outcome="Y", conditions=["A"], relevance_threshold=1.0)
        assert strict.necessary == ()
        assert "A" in {item.expression for item in strict.trivial}

    def test_the_report_separates_necessary_from_trivial(self) -> None:
        text = str(necessity_analysis(DATA, outcome="Y", conditions=["A", "B"]))
        assert "Necessary:" in text
        assert "trivial" in text

    def test_the_report_says_so_when_nothing_is_necessary(self) -> None:
        # C is anti-correlated with Y, and its absence is excluded here, so no
        # candidate reaches the consistency threshold.
        text = str(necessity_analysis(DATA, outcome="Y", conditions=["C"], include_absence=False))
        assert "Necessary: none" in text

    def test_the_absence_of_a_condition_can_be_the_necessary_one(self) -> None:
        """C runs against the outcome, so ~C is the superset, not C."""
        analysis = necessity_analysis(DATA, outcome="Y", conditions=["C"])
        assert _by_name(analysis, "C").fit.consistency < 0.9
        assert _by_name(analysis, "~C").fit.consistency == pytest.approx(1.0)
        assert "~C" in {item.expression for item in analysis.necessary}


class TestDisjunctions:
    def test_unions_are_not_screened_unless_requested(self) -> None:
        analysis = necessity_analysis(DATA, outcome="Y", conditions=["A", "C"])
        assert not any("+" in item.expression for item in analysis.candidates)

    def test_pairs_are_screened_when_asked(self) -> None:
        analysis = necessity_analysis(
            DATA, outcome="Y", conditions=["A", "C"], max_disjunction_size=2
        )
        assert any("+" in item.expression for item in analysis.candidates)

    def test_a_union_is_never_less_consistent_than_its_parts(self) -> None:
        """min(max(A,B), Y) >= min(A, Y), so unions can only help."""
        analysis = necessity_analysis(
            DATA,
            outcome="Y",
            conditions=["A", "C"],
            max_disjunction_size=2,
            include_absence=False,
        )
        union = _by_name(analysis, "A+C")
        for part in ("A", "C"):
            assert union.fit.consistency >= _by_name(analysis, part).fit.consistency - 1e-12

    def test_a_union_can_be_necessary_when_no_part_is(self) -> None:
        """The SUIN case: neither half covers the outcome, but together they do."""
        frame = pd.DataFrame(
            {
                "A": [0.9, 0.1, 0.9, 0.1],
                "B": [0.1, 0.9, 0.1, 0.9],
                "Y": [0.8, 0.8, 0.1, 0.1],
            }
        )
        analysis = necessity_analysis(
            frame,
            outcome="Y",
            conditions=["A", "B"],
            max_disjunction_size=2,
            include_absence=False,
        )
        assert _by_name(analysis, "A").fit.consistency < 0.9
        assert _by_name(analysis, "B").fit.consistency < 0.9
        assert _by_name(analysis, "A+B").fit.consistency == pytest.approx(1.0)

    def test_conjunctions_are_never_screened(self) -> None:
        """A conjunction cannot beat its own parts, so testing it is pointless."""
        analysis = necessity_analysis(
            DATA, outcome="Y", conditions=["A", "C"], max_disjunction_size=2
        )
        assert not any("*" in item.expression for item in analysis.candidates)

    def test_the_documented_inequality_actually_holds(self) -> None:
        """consistency(A*B) <= min over parts, which is why conjunctions are excluded."""
        from setqca.metrics import necessity

        a, b, y = DATA["A"].to_numpy(), DATA["C"].to_numpy(), DATA["Y"].to_numpy()
        conjunction = necessity(np.minimum(a, b), y).consistency
        assert conjunction <= necessity(a, y).consistency + 1e-12
        assert conjunction <= necessity(b, y).consistency + 1e-12


class TestExport:
    def test_the_frame_has_the_documented_columns(self) -> None:
        frame = necessity_analysis(DATA, outcome="Y", conditions=["A", "B"]).to_frame()
        assert list(frame.columns) == [
            "condition",
            "consistency",
            "coverage",
            "RoN",
            "prevalence",
            "necessary",
            "trivial",
        ]

    def test_the_frame_is_sorted_by_consistency(self) -> None:
        frame = necessity_analysis(DATA, outcome="Y", conditions=["A", "B", "C"]).to_frame()
        assert frame["consistency"].is_monotonic_decreasing

    def test_the_frame_has_one_row_per_candidate(self) -> None:
        analysis = necessity_analysis(DATA, outcome="Y", conditions=["A", "B"])
        assert len(analysis.to_frame()) == len(analysis.candidates)


class TestGuards:
    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_consistency_threshold_must_be_a_proportion(self, value: float) -> None:
        with pytest.raises(ValueError, match="consistency_threshold"):
            necessity_analysis(DATA, outcome="Y", conditions=["A"], consistency_threshold=value)

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_relevance_threshold_must_be_a_proportion(self, value: float) -> None:
        with pytest.raises(ValueError, match="relevance_threshold"):
            necessity_analysis(DATA, outcome="Y", conditions=["A"], relevance_threshold=value)

    def test_disjunction_size_must_be_at_least_one(self) -> None:
        with pytest.raises(ValueError, match="max_disjunction_size"):
            necessity_analysis(DATA, outcome="Y", conditions=["A"], max_disjunction_size=0)

    def test_at_least_one_condition_is_required(self) -> None:
        with pytest.raises(ValueError, match="At least one condition"):
            necessity_analysis(DATA, outcome="Y", conditions=[])

    def test_unknown_columns_are_rejected(self) -> None:
        with pytest.raises(KeyError, match="Missing columns"):
            necessity_analysis(DATA, outcome="Y", conditions=["Z"])

    def test_uncalibrated_data_is_rejected(self) -> None:
        frame = pd.DataFrame({"A": [1.5, 0.2], "Y": [0.9, 0.1]})
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            necessity_analysis(frame, outcome="Y", conditions=["A"])
