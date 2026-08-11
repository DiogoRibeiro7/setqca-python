"""Tests for the textual rendering of set expressions and truth tables."""

from __future__ import annotations

import pandas as pd
import pytest

from setqca import Condition, build_truth_table


def test_set_expressions_render_in_standard_qca_notation() -> None:
    A, B, C = Condition("A"), Condition("B"), Condition("C")
    assert str(A) == "A"
    assert str(~A) == "~A"
    assert str(A & B) == "A*B"
    assert str(A | B) == "A+B"
    assert str((A & B) | ~C) == "A*B+~C"


def test_truth_table_frame_reports_one_row_per_corner(crisp_data: pd.DataFrame) -> None:
    table = build_truth_table(
        crisp_data,
        outcome="Y",
        conditions=["A", "B"],
        inclusion_cutoff=1.0,
        case_id="case",
    )
    frame = table.to_frame()

    assert len(frame) == 4, "a complete truth table has 2**k rows"
    assert list(frame.columns) == [
        "A",
        "B",
        "minterm",
        "n",
        "consistency",
        "PRI",
        "OUT",
        "cases",
        "excluded_because",
    ]
    assert frame["minterm"].tolist() == [0, 1, 2, 3]
    assert frame.loc[frame["minterm"] == 3, "cases"].item() == "c1"
    assert frame.loc[frame["minterm"] == 1, "OUT"].item() == "R"


def test_rows_know_whether_they_were_observed(crisp_data: pd.DataFrame) -> None:
    table = build_truth_table(crisp_data, outcome="Y", conditions=["A", "B"], inclusion_cutoff=1.0)
    observed = {row.minterm for row in table.rows if row.observed}
    assert observed == {0, 2, 3}
    assert table.remainder_minterms == {1}


def test_case_labels_fall_back_to_the_frame_index() -> None:
    data = pd.DataFrame({"A": [0.9, 0.1], "Y": [0.9, 0.1]}, index=["alpha", "beta"])
    table = build_truth_table(data, outcome="Y", conditions=["A"])
    labels = {row.minterm: row.cases for row in table.rows}
    assert labels[1] == ("alpha",)
    assert labels[0] == ("beta",)


def test_contradictory_rows_are_coded_between_the_two_cutoffs() -> None:
    # The A=1 corner has consistency 0.6: below inclusion, above exclusion.
    data = pd.DataFrame({"A": [0.9, 0.9], "Y": [0.9, 0.2]})
    table = build_truth_table(
        data,
        outcome="Y",
        conditions=["A"],
        inclusion_cutoff=0.8,
        exclusion_cutoff=0.4,
    )
    row = next(row for row in table.rows if row.minterm == 1)
    assert row.consistency == pytest.approx(1.1 / 1.8)
    assert row.outcome == "C"
    assert table.contradictory_minterms == {1}
