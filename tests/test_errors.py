"""Tests for the error contracts of the public API.

Every guard that rejects invalid input is exercised here so that the promised
failure modes cannot regress into silent approximation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from setqca import CSQCA, FSQCA, Condition, DirectCalibration, build_truth_table
from setqca.minimize import exact_minimum_covers, minimize, prime_implicants
from setqca.minimize.implicant import Implicant, minterm_to_implicant
from setqca.sets import Intersection, Union


class TestCalibrationGuards:
    @pytest.mark.parametrize(
        ("full_out", "crossover", "full_in"),
        [(10, 10, 30), (10, 40, 30), (30, 20, 30), (10, 5, 30)],
    )
    def test_anchors_must_be_strictly_ordered(
        self, full_out: float, crossover: float, full_in: float
    ) -> None:
        with pytest.raises(ValueError, match="strictly ordered"):
            DirectCalibration(full_out=full_out, crossover=crossover, full_in=full_in)

    @pytest.mark.parametrize("idm", [0.5, 0.4, 1.0, 1.5])
    def test_idm_must_lie_strictly_between_half_and_one(self, idm: float) -> None:
        with pytest.raises(ValueError, match="idm"):
            DirectCalibration(full_out=10, crossover=20, full_in=30, idm=idm)

    @pytest.mark.parametrize(("below", "above"), [(0.0, 1.0), (1.0, -1.0)])
    def test_piecewise_exponents_must_be_positive(self, below: float, above: float) -> None:
        with pytest.raises(ValueError, match="positive"):
            DirectCalibration(full_out=10, crossover=20, full_in=30, below=below, above=above)


class TestBooleanEngineGuards:
    def test_on_set_and_dont_cares_must_be_disjoint(self) -> None:
        with pytest.raises(ValueError, match="disjoint"):
            prime_implicants({1, 2}, {2}, width=3)

    def test_empty_problem_yields_no_prime_implicants(self) -> None:
        assert prime_implicants(set(), set(), width=3) == ()

    def test_empty_on_set_yields_the_empty_cover(self) -> None:
        solutions = minimize(set(), width=3)
        assert len(solutions) == 1
        assert solutions[0].implicants == ()
        assert solutions[0].literal_count == 0

    def test_an_uncoverable_chart_is_reported_rather_than_silently_truncated(self) -> None:
        with pytest.raises(RuntimeError, match="cannot cover every positive row"):
            exact_minimum_covers((), {1})

    def test_minterm_must_fall_inside_the_truth_table_domain(self) -> None:
        with pytest.raises(ValueError, match="outside the truth-table domain"):
            minterm_to_implicant(8, width=3)
        with pytest.raises(ValueError, match="outside the truth-table domain"):
            minterm_to_implicant(-1, width=3)

    def test_implicants_of_different_widths_do_not_combine(self) -> None:
        assert minterm_to_implicant(1, 3).combine(minterm_to_implicant(1, 4)) is None

    def test_implicants_differing_in_more_than_one_literal_do_not_combine(self) -> None:
        assert minterm_to_implicant(0, 3).combine(minterm_to_implicant(3, 3)) is None

    def test_a_dont_care_literal_blocks_combination(self) -> None:
        left = Implicant((1, None, 0), frozenset({4}))
        right = Implicant((1, 1, 1), frozenset({7}))
        assert left.combine(right) is None

    def test_rendering_requires_matching_condition_count(self) -> None:
        with pytest.raises(ValueError, match="condition count"):
            minterm_to_implicant(3, width=3).as_expression(("A", "B"))

    def test_the_tautology_renders_as_one(self) -> None:
        assert Implicant((None, None), frozenset({0})).as_expression(("A", "B")) == "1"


class TestSetExpressionGuards:
    def test_missing_condition_column_is_reported(self) -> None:
        with pytest.raises(KeyError, match="Missing condition column"):
            Condition("Z").evaluate(pd.DataFrame({"A": [0.5]}))

    def test_empty_intersection_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one operand"):
            Intersection(()).evaluate(pd.DataFrame({"A": [0.5]}))

    def test_empty_union_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one operand"):
            Union(()).evaluate(pd.DataFrame({"A": [0.5]}))


class TestTruthTableGuards:
    def test_data_must_be_a_dataframe(self) -> None:
        with pytest.raises(TypeError, match="pandas DataFrame"):
            build_truth_table({"A": [1]}, outcome="Y", conditions=["A"])  # type: ignore[arg-type]

    def test_at_least_one_condition_is_required(self, fuzzy_data: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="At least one condition"):
            build_truth_table(fuzzy_data, outcome="Y", conditions=[])

    @pytest.mark.parametrize("cutoff", [-0.1, 1.1])
    def test_inclusion_cutoff_must_be_a_proportion(
        self, fuzzy_data: pd.DataFrame, cutoff: float
    ) -> None:
        with pytest.raises(ValueError, match="inclusion_cutoff"):
            build_truth_table(fuzzy_data, outcome="Y", conditions=["A"], inclusion_cutoff=cutoff)

    def test_exclusion_cutoff_may_not_exceed_the_inclusion_cutoff(
        self, fuzzy_data: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="exclusion_cutoff"):
            build_truth_table(
                fuzzy_data,
                outcome="Y",
                conditions=["A"],
                inclusion_cutoff=0.8,
                exclusion_cutoff=0.9,
            )

    def test_pri_cutoff_must_be_a_proportion(self, fuzzy_data: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="pri_cutoff"):
            build_truth_table(fuzzy_data, outcome="Y", conditions=["A"], pri_cutoff=1.5)

    def test_frequency_cutoff_must_be_positive(self, fuzzy_data: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="frequency_cutoff"):
            build_truth_table(fuzzy_data, outcome="Y", conditions=["A"], frequency_cutoff=0)

    def test_conditions_must_be_calibrated(self) -> None:
        data = pd.DataFrame({"A": [1.5, 0.2], "Y": [0.9, 0.1]})
        with pytest.raises(ValueError, match=r"memberships in \[0, 1\]"):
            build_truth_table(data, outcome="Y", conditions=["A"])

    def test_crossover_cases_are_admitted_when_explicitly_allowed(self) -> None:
        data = pd.DataFrame({"A": [0.5, 0.9], "Y": [0.8, 0.9]})
        table = build_truth_table(data, outcome="Y", conditions=["A"], allow_crossover_cases=True)
        assert sum(row.frequency for row in table.rows) == 2


class TestEstimatorGuards:
    def test_max_solutions_must_be_positive(self, fuzzy_data: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="max_solutions"):
            FSQCA(max_solutions=0).fit(fuzzy_data, outcome="Y", conditions=["A", "B"])

    def test_directional_expectations_must_use_known_symbols(
        self, fuzzy_data: pd.DataFrame
    ) -> None:
        model = FSQCA(directional_expectations={"A": "up"})  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="must be"):
            model.fit(fuzzy_data, outcome="Y", conditions=["A", "B"])

    def test_directional_expectations_must_reference_known_conditions(
        self, fuzzy_data: pd.DataFrame
    ) -> None:
        model = FSQCA(consistency=0.8, directional_expectations={"Z": "+"})
        with pytest.raises(KeyError, match="unknown condition"):
            model.fit(fuzzy_data, outcome="Y", conditions=["A", "B"])

    def test_no_sufficient_row_is_reported_rather_than_returning_an_empty_solution(self) -> None:
        # No corner of the property space reaches the default 0.8 inclusion cutoff.
        data = pd.DataFrame({"A": [0.9, 0.8], "B": [0.7, 0.6], "Y": [0.2, 0.1]})
        with pytest.raises(ValueError, match="No truth-table row is sufficient"):
            FSQCA().fit(data, outcome="Y", conditions=["A", "B"])

    def test_csqca_rejects_a_fuzzy_condition(self, fuzzy_data: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not crisp"):
            CSQCA().fit(fuzzy_data, outcome="Y", conditions=["A", "B"])

    def test_csqca_rejects_a_fuzzy_outcome(self) -> None:
        data = pd.DataFrame({"A": [1, 0], "Y": [0.9, 0.1]})
        with pytest.raises(ValueError, match="'Y' is not crisp"):
            CSQCA().fit(data, outcome="Y", conditions=["A"])
