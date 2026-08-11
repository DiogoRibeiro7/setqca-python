"""Tests for the structured result objects and their exports."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from setqca import CSQCA, FSQCA, QCAResult


@pytest.fixture
def fitted(crisp_data: pd.DataFrame) -> QCAResult:
    return CSQCA().fit(crisp_data, outcome="Y", conditions=["A", "B"], case_id="case")


def test_method_name_reflects_the_estimator(fitted: QCAResult, fuzzy_data: pd.DataFrame) -> None:
    assert fitted.method == "Crisp-set Qualitative Comparative Analysis"
    fuzzy = FSQCA(consistency=0.8).fit(fuzzy_data, outcome="Y", conditions=["A", "B"])
    assert fuzzy.method == "Fuzzy-set Qualitative Comparative Analysis"


def test_summary_frame_reports_one_row_per_minimal_solution(fitted: QCAResult) -> None:
    frame = fitted.summary_frame("parsimonious")
    assert list(frame.columns) == [
        "solution",
        "consistency",
        "coverage",
        "PRI",
        "n_implicants",
        "n_literals",
    ]
    assert len(frame) == len(fitted.parsimonious)
    assert frame.loc[0, "solution"] == "A"


def test_summary_frame_of_an_uncomputed_family_is_empty(fitted: QCAResult) -> None:
    assert fitted.intermediate is None
    assert fitted.summary_frame("intermediate").empty


def test_solutions_rejects_an_unknown_family(fitted: QCAResult) -> None:
    with pytest.raises(ValueError, match="Unknown solution kind"):
        fitted.solutions("parsimonius")


def test_str_reports_both_solution_families(fitted: QCAResult) -> None:
    text = str(fitted)
    assert "Crisp-set Qualitative Comparative Analysis" in text
    assert "Outcome: Y" in text
    assert "Conditions: A, B" in text
    assert "Conservative solution(s):" in text
    assert "Parsimonious solution(s):" in text
    assert "cons=1.000" in text


def test_intermediate_solutions_are_no_longer_experimental(
    fuzzy_data: pd.DataFrame,
) -> None:
    result = FSQCA(consistency=0.8, directional_expectations={"A": "+", "B": "0"}).fit(
        fuzzy_data, outcome="Y", conditions=["A", "B"]
    )
    assert result.intermediate_experimental is False
    assert result.intermediate is not None
    assert "Intermediate solution(s):" in str(result)
    assert "experimental" not in str(result)


def test_str_reports_the_counterfactual_split(fuzzy_data: pd.DataFrame) -> None:
    result = FSQCA(consistency=0.8, directional_expectations={"A": "+", "B": "+"}).fit(
        fuzzy_data, outcome="Y", conditions=["A", "B"]
    )
    assert "Counterfactuals:" in str(result)
    assert "easy admitted" in str(result)


def test_the_experimental_label_is_retained_for_backward_compatibility(
    fuzzy_data: pd.DataFrame,
) -> None:
    """The field survives from 0.1.0 but is now always False for fitted results.

    Setting it by hand still drives the rendering, so a caller reconstructing an
    older result keeps the old output.
    """
    result = FSQCA(consistency=0.8, directional_expectations={"A": "+"}).fit(
        fuzzy_data, outcome="Y", conditions=["A", "B"]
    )
    assert result.intermediate_experimental is False
    legacy = replace(result, intermediate_experimental=True)
    assert "Intermediate solution(s) [experimental]:" in str(legacy)


def test_term_level_fits_are_reported_per_implicant(fuzzy_data: pd.DataFrame) -> None:
    result = FSQCA(consistency=0.7).fit(fuzzy_data, outcome="Y", conditions=["A", "B"])
    for solution in result.conservative:
        assert len(solution.term_fits) == len(solution.boolean.implicants)
        for fit in solution.term_fits:
            assert 0.0 <= fit.consistency <= 1.0


def test_solution_membership_is_the_union_of_its_terms(fuzzy_data: pd.DataFrame) -> None:
    """A disjunctive solution covers at least as much of the outcome as each term."""
    result = FSQCA(consistency=0.7).fit(fuzzy_data, outcome="Y", conditions=["A", "B"])
    for solution in result.conservative:
        for term in solution.term_fits:
            assert solution.fit.coverage >= term.coverage - 1e-12
