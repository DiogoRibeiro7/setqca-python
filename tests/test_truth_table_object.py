"""Tests for the truth table as a reusable analytical object."""

from __future__ import annotations

import pandas as pd
import pytest

from setqca import build_truth_table
from setqca.truth_table import TruthTable

# Two clearly sufficient corners, one clearly negative, one unobserved.
DATA = pd.DataFrame(
    {
        "A": [0.9, 0.9, 0.8, 0.1, 0.1],
        "B": [0.9, 0.8, 0.9, 0.1, 0.2],
        "Y": [0.95, 0.9, 0.85, 0.1, 0.15],
    },
    index=["c1", "c2", "c3", "c4", "c5"],
)


def _table(**kwargs: object) -> TruthTable:
    return build_truth_table(DATA, outcome="Y", conditions=["A", "B"], **kwargs)  # type: ignore[arg-type]


class TestRowAccessors:
    def test_the_four_codes_partition_the_table(self) -> None:
        table = _table()
        groups = (
            table.positive_rows()
            + table.negative_rows()
            + table.contradictions()
            + table.remainders()
        )
        assert len(groups) == len(table.rows)
        assert {row.minterm for row in groups} == {row.minterm for row in table.rows}

    def test_accessors_agree_with_the_minterm_properties(self) -> None:
        table = _table()
        assert {row.minterm for row in table.positive_rows()} == table.positive_minterms
        assert {row.minterm for row in table.negative_rows()} == table.negative_minterms
        assert {row.minterm for row in table.contradictions()} == table.contradictory_minterms
        assert {row.minterm for row in table.remainders()} == table.remainder_minterms

    def test_rows_come_back_in_minterm_order(self) -> None:
        table = _table()
        minterms = [row.minterm for row in table.rows]
        assert minterms == sorted(minterms)
        for accessor in (table.positive_rows, table.negative_rows, table.remainders):
            values = [row.minterm for row in accessor()]
            assert values == sorted(values)

    def test_rows_carry_their_cases(self) -> None:
        table = _table(case_id=None)
        positive = table.positive_rows()
        assert positive
        assert any(row.cases for row in positive)


class TestExclusionReasons:
    def test_a_sufficient_row_has_no_reason(self) -> None:
        for row in _table().positive_rows():
            assert row.exclusion_reason is None

    def test_a_remainder_names_the_frequency_cutoff(self) -> None:
        table = _table(frequency_cutoff=2)
        remainders = table.remainders()
        assert remainders
        assert all("frequency" in (row.exclusion_reason or "") for row in remainders)

    def test_a_low_consistency_row_names_consistency(self) -> None:
        row = next(row for row in _table().negative_rows() if row.frequency > 0)
        assert "consistency" in (row.exclusion_reason or "")

    def test_a_row_failing_only_pri_names_pri_not_consistency(self) -> None:
        """The outcome code alone cannot distinguish these two exclusions."""
        # This corner is consistent but its PRI is poor, so only the PRI cutoff
        # keeps it out.
        frame = pd.DataFrame(
            {"A": [0.9, 0.9], "B": [0.9, 0.9], "Y": [0.6, 0.55]},
            index=["c1", "c2"],
        )
        table = build_truth_table(
            frame, outcome="Y", conditions=["A", "B"], inclusion_cutoff=0.5, pri_cutoff=0.9
        )
        row = next(row for row in table.rows if row.minterm == 3)

        assert row.consistency >= 0.5, "consistency passes"
        assert row.pri < 0.9, "PRI does not"
        assert "PRI" in (row.exclusion_reason or "")
        assert "consistency" not in (row.exclusion_reason or "")

    def test_threshold_exclusions_are_separated_from_evidence_exclusions(self) -> None:
        table = _table(frequency_cutoff=2)
        excluded = table.excluded_rows()
        assert excluded
        for row in excluded:
            assert row.excluded_by_threshold
        # A row with genuinely poor consistency is excluded by the data, and is
        # therefore not something a different threshold would rescue.
        poor = [row for row in table.negative_rows() if row.frequency >= 2]
        for row in poor:
            assert not row.excluded_by_threshold

    def test_the_reason_appears_in_the_frame(self) -> None:
        frame = _table(frequency_cutoff=2).to_frame()
        assert "excluded_because" in frame.columns
        assert frame["excluded_because"].str.contains("frequency").any()


class TestSummary:
    def test_the_summary_counts_every_group(self) -> None:
        text = _table(frequency_cutoff=2).summary()
        assert "configurations of 2 conditions" in text
        assert "remainders:" in text
        assert "excluded by a threshold:" in text


class TestSerialisation:
    def test_a_table_round_trips_through_json(self) -> None:
        table = _table(frequency_cutoff=2, inclusion_cutoff=0.75, pri_cutoff=0.1)
        restored = TruthTable.from_json(table.to_json())

        assert restored.conditions == table.conditions
        assert restored.outcome_name == table.outcome_name
        assert restored.inclusion_cutoff == table.inclusion_cutoff
        assert restored.exclusion_cutoff == table.exclusion_cutoff
        assert restored.pri_cutoff == table.pri_cutoff
        assert restored.frequency_cutoff == table.frequency_cutoff
        assert restored.rows == table.rows

    def test_the_exclusion_reasons_survive(self) -> None:
        table = _table(frequency_cutoff=2)
        restored = TruthTable.from_json(table.to_json())
        assert [row.exclusion_reason for row in restored.rows] == [
            row.exclusion_reason for row in table.rows
        ]

    def test_the_dictionary_form_is_json_compatible(self) -> None:
        import json

        payload = _table().to_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_a_missing_key_is_reported(self) -> None:
        with pytest.raises(KeyError):
            TruthTable.from_dict({"conditions": ["A"]})


class TestMinimisationFromTheTable:
    def test_a_stored_table_minimises_without_the_original_data(self) -> None:
        table = _table()
        restored = TruthTable.from_json(table.to_json())

        solutions = restored.minimize()
        assert solutions
        assert solutions[0].as_expression(restored.conditions)

    def test_it_agrees_with_the_estimator(self) -> None:
        from setqca import FSQCA

        table = _table()
        result = FSQCA(consistency=0.8).fit(DATA, outcome="Y", conditions=["A", "B"])
        direct = table.minimize()

        assert {solution.implicants for solution in direct} == {
            solution.boolean.implicants for solution in result.conservative
        }

    def test_remainders_can_be_admitted_for_the_parsimonious_result(self) -> None:
        from setqca import FSQCA

        table = _table()
        result = FSQCA(consistency=0.8).fit(DATA, outcome="Y", conditions=["A", "B"])
        parsimonious = table.minimize(include_remainders=True)

        assert {solution.implicants for solution in parsimonious} == {
            solution.boolean.implicants for solution in result.parsimonious
        }

    def test_a_table_with_no_sufficient_row_refuses_to_minimise(self) -> None:
        table = _table(inclusion_cutoff=1.0, frequency_cutoff=99)
        with pytest.raises(ValueError, match="No truth-table row is sufficient"):
            table.minimize()

    def test_max_solutions_is_respected(self) -> None:
        table = _table()
        assert len(table.minimize(max_solutions=1)) == 1
