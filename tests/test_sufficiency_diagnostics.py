"""Tests for case-level sufficiency diagnostics.

The dataset below is built so every case's role is known by hand before the
code runs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from setqca import CaseRole, sufficiency_diagnostics
from setqca.analysis.sufficiency import classify_case

# X is the term membership, Y the outcome. Roles, in order:
#   t1  X=0.9 Y=0.9  in both, X <= Y            -> typical
#   t2  X=0.9 Y=0.7  in both, X > Y             -> deviant consistency (degree)
#   t3  X=0.8 Y=0.2  in term, outcome absent    -> deviant consistency (kind)
#   t4  X=0.2 Y=0.8  outcome without the term   -> deviant coverage
#   t5  X=0.1 Y=0.1  outside both               -> individually irrelevant
KNOWN = pd.DataFrame(
    {
        "X": [0.9, 0.9, 0.8, 0.2, 0.1],
        "Y": [0.9, 0.7, 0.2, 0.8, 0.1],
    },
    index=["t1", "t2", "t3", "t4", "t5"],
)


class TestCaseTypology:
    @pytest.mark.parametrize(
        ("term", "outcome", "expected"),
        [
            (0.9, 0.9, CaseRole.TYPICAL),
            (0.6, 1.0, CaseRole.TYPICAL),
            (1.0, 1.0, CaseRole.TYPICAL),
            (0.9, 0.7, CaseRole.DEVIANT_CONSISTENCY_IN_DEGREE),
            (0.8, 0.2, CaseRole.DEVIANT_CONSISTENCY_IN_KIND),
            (0.2, 0.8, CaseRole.DEVIANT_COVERAGE),
            (0.1, 0.1, CaseRole.INDIVIDUALLY_IRRELEVANT),
        ],
    )
    def test_classification_matches_the_typology(
        self, term: float, outcome: float, expected: CaseRole
    ) -> None:
        assert classify_case(term, outcome) is expected

    def test_equal_memberships_above_the_crossover_are_typical(self) -> None:
        """X <= Y is the consistency condition, so equality is consistent."""
        assert classify_case(0.8, 0.8) is CaseRole.TYPICAL

    def test_the_crossover_itself_is_outside_both_sets(self) -> None:
        """Membership of exactly 0.5 is not 'more in than out'."""
        assert classify_case(0.5, 0.5) is CaseRole.INDIVIDUALLY_IRRELEVANT
        assert classify_case(0.5, 0.9) is CaseRole.DEVIANT_COVERAGE

    def test_only_consistency_deviance_counts_against_the_claim(self) -> None:
        assert CaseRole.DEVIANT_CONSISTENCY_IN_KIND.contradicts_sufficiency
        assert CaseRole.DEVIANT_CONSISTENCY_IN_DEGREE.contradicts_sufficiency
        assert not CaseRole.DEVIANT_COVERAGE.contradicts_sufficiency
        assert not CaseRole.TYPICAL.contradicts_sufficiency
        assert not CaseRole.INDIVIDUALLY_IRRELEVANT.contradicts_sufficiency


class TestKnownRoles:
    def test_every_case_lands_in_its_hand_computed_role(self) -> None:
        result = sufficiency_diagnostics(KNOWN, outcome="Y", terms=["X"])
        roles = {item.case: item.role for item in result.terms[0].cases}
        assert roles == {
            "t1": CaseRole.TYPICAL,
            "t2": CaseRole.DEVIANT_CONSISTENCY_IN_DEGREE,
            "t3": CaseRole.DEVIANT_CONSISTENCY_IN_KIND,
            "t4": CaseRole.DEVIANT_COVERAGE,
            "t5": CaseRole.INDIVIDUALLY_IRRELEVANT,
        }

    def test_the_named_accessors_agree_with_the_roles(self) -> None:
        term = sufficiency_diagnostics(KNOWN, outcome="Y", terms=["X"]).terms[0]
        assert term.typical == ("t1",)
        assert term.contradictory == ("t3",)
        assert term.deviant_coverage == ("t4",)
        assert set(term.deviant_consistency) == {"t2", "t3"}

    def test_frequency_counts_cases_in_the_term(self) -> None:
        term = sufficiency_diagnostics(KNOWN, outcome="Y", terms=["X"]).terms[0]
        assert term.frequency == 3  # t1, t2, t3

    def test_case_labels_come_from_the_index_by_default(self) -> None:
        term = sufficiency_diagnostics(KNOWN, outcome="Y", terms=["X"]).terms[0]
        assert [item.case for item in term.cases] == ["t1", "t2", "t3", "t4", "t5"]

    def test_a_case_id_column_can_be_used_instead(self) -> None:
        frame = KNOWN.reset_index(names="country")
        term = sufficiency_diagnostics(frame, outcome="Y", terms=["X"], case_id="country").terms[0]
        assert term.typical == ("t1",)


class TestUniqueCoverage:
    frame = pd.DataFrame(
        {
            "A": [0.9, 0.1, 0.9],
            "B": [0.1, 0.9, 0.9],
            "Y": [0.9, 0.9, 0.9],
        },
        index=["a-only", "b-only", "both"],
    )

    def test_a_single_term_uniquely_covers_everything_it_covers(self) -> None:
        result = sufficiency_diagnostics(self.frame, outcome="Y", terms=["A"])
        term = result.terms[0]
        assert term.unique_coverage == pytest.approx(term.fit.coverage)

    def test_overlap_is_removed_from_unique_coverage(self) -> None:
        result = sufficiency_diagnostics(self.frame, outcome="Y", terms=["A", "B"])
        for term in result.terms:
            assert term.unique_coverage < term.fit.coverage

    def test_a_duplicated_term_has_no_unique_coverage(self) -> None:
        """Two identical terms each explain nothing the other does not."""
        result = sufficiency_diagnostics(self.frame, outcome="Y", terms=["A", "A"])
        for term in result.terms:
            assert term.unique_coverage == pytest.approx(0.0)
            assert term.redundant is True

    def test_a_case_covered_by_two_terms_is_uniquely_covered_by_neither(self) -> None:
        result = sufficiency_diagnostics(self.frame, outcome="Y", terms=["A", "B"])
        by_term = {term.expression: term.uniquely_covered for term in result.terms}
        assert by_term["A"] == ("a-only",)
        assert by_term["B"] == ("b-only",)

    def test_an_empty_outcome_yields_zero_rather_than_dividing_by_zero(self) -> None:
        frame = pd.DataFrame({"A": [0.9, 0.8], "Y": [0.0, 0.0]})
        result = sufficiency_diagnostics(frame, outcome="Y", terms=["A"])
        assert result.terms[0].unique_coverage == 0.0


class TestSolutionLevel:
    def test_solution_fit_uses_the_union_of_the_terms(self) -> None:
        from setqca import sufficiency

        frame = TestUniqueCoverage.frame
        result = sufficiency_diagnostics(frame, outcome="Y", terms=["A", "B"])
        union = frame[["A", "B"]].max(axis=1)
        assert result.fit.consistency == pytest.approx(sufficiency(union, frame["Y"]).consistency)

    def test_redundant_terms_are_reported(self) -> None:
        result = sufficiency_diagnostics(TestUniqueCoverage.frame, outcome="Y", terms=["A", "A"])
        assert len(result.redundant_terms) == 2

    def test_terms_may_be_given_as_expressions(self) -> None:
        from setqca import Condition

        result = sufficiency_diagnostics(KNOWN, outcome="Y", terms=[Condition("X")])
        assert result.terms[0].expression == "X"

    def test_term_strings_are_parsed(self) -> None:
        frame = pd.DataFrame({"A": [0.9, 0.1], "B": [0.8, 0.2], "Y": [0.9, 0.1]})
        result = sufficiency_diagnostics(frame, outcome="Y", terms=["A*~B"])
        assert result.terms[0].expression == "A*~B"


class TestExport:
    def test_the_term_frame_has_a_column_per_role(self) -> None:
        frame = sufficiency_diagnostics(KNOWN, outcome="Y", terms=["X"]).to_frame()
        for role in CaseRole:
            assert role.value in frame.columns
        assert frame.loc[0, "n"] == 3

    def test_the_case_frame_has_one_row_per_case_per_term(self) -> None:
        result = sufficiency_diagnostics(TestUniqueCoverage.frame, outcome="Y", terms=["A", "B"])
        frame = result.cases_frame()
        assert len(frame) == 2 * 3
        assert list(frame.columns) == [
            "term",
            "case",
            "term_membership",
            "outcome_membership",
            "role",
            "uniquely_covered",
        ]

    def test_the_report_names_the_awkward_cases(self) -> None:
        text = str(sufficiency_diagnostics(KNOWN, outcome="Y", terms=["X"]))
        assert "typical: t1" in text
        assert "contradictory: t3" in text
        assert "unexplained outcomes: t4" in text

    def test_the_report_omits_sections_with_nothing_to_say(self) -> None:
        """A term with no awkward cases gets no awkward-case lines."""
        clean = pd.DataFrame({"X": [0.9, 0.1], "Y": [0.95, 0.05]}, index=["a", "b"])
        text = str(sufficiency_diagnostics(clean, outcome="Y", terms=["X"]))
        assert "typical: a" in text
        assert "contradictory" not in text
        assert "unexplained outcomes" not in text
        assert "redundant" not in text

    def test_the_report_names_terms_with_no_typical_cases(self) -> None:
        """A term supported by nothing still appears, without a typical line."""
        awkward = pd.DataFrame({"X": [0.9, 0.8], "Y": [0.2, 0.1]}, index=["a", "b"])
        text = str(sufficiency_diagnostics(awkward, outcome="Y", terms=["X"]))
        assert "typical:" not in text
        assert "contradictory: a, b" in text

    def test_the_report_flags_redundancy(self) -> None:
        text = str(sufficiency_diagnostics(TestUniqueCoverage.frame, outcome="Y", terms=["A", "A"]))
        assert "redundant" in text


class TestGuards:
    def test_at_least_one_term_is_required(self) -> None:
        with pytest.raises(ValueError, match="At least one term"):
            sufficiency_diagnostics(KNOWN, outcome="Y", terms=[])

    def test_an_unknown_outcome_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="Missing columns"):
            sufficiency_diagnostics(KNOWN, outcome="Z", terms=["X"])

    def test_an_unknown_case_column_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="Missing columns"):
            sufficiency_diagnostics(KNOWN, outcome="Y", terms=["X"], case_id="nope")

    def test_uncalibrated_data_is_rejected(self) -> None:
        frame = pd.DataFrame({"X": [0.5, 0.5], "Y": [1.5, 0.1]})
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            sufficiency_diagnostics(frame, outcome="Y", terms=["X"])
