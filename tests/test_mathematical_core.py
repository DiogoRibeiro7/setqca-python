"""Hand-computable tests for every exported mathematical quantity.

Each test states the arithmetic it expects in a comment, so the assertion can
be checked by hand against the definition in ``docs/mathematical_validation.md``
without running the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from setqca import (
    Condition,
    build_truth_table,
    calibrate_crisp,
    calibrate_direct,
    minimize,
    necessity,
    sufficiency,
)

TOL = 1e-12


# ---------------------------------------------------------------------------
# Membership representation
# ---------------------------------------------------------------------------


class TestMembership:
    def test_the_closed_unit_interval_is_admissible(self) -> None:
        fit = sufficiency([0.0, 1.0], [0.0, 1.0])
        assert fit.consistency == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [-1e-9, 1.0 + 1e-9])
    def test_membership_outside_the_unit_interval_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            sufficiency([0.5, bad], [0.5, 0.5])

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_missing_and_infinite_values_are_rejected_not_imputed(self, bad: float) -> None:
        with pytest.raises(ValueError, match="NaN or infinite"):
            sufficiency([0.5, bad], [0.5, 0.5])

    def test_crisp_membership_is_the_binary_special_case(self) -> None:
        assert calibrate_crisp([1, 5, 9], [4]).tolist() == [0, 1, 1]


# ---------------------------------------------------------------------------
# Fuzzy operators
# ---------------------------------------------------------------------------


class TestFuzzyOperators:
    frame = pd.DataFrame({"A": [0.0, 0.3, 0.7, 1.0], "B": [1.0, 0.6, 0.2, 0.0]})

    def test_negation_is_one_minus_membership(self) -> None:
        # ~A = 1 - A
        assert (~Condition("A")).evaluate(self.frame) == pytest.approx([1.0, 0.7, 0.3, 0.0])

    def test_intersection_is_the_minimum_t_norm(self) -> None:
        # A*B = min(A, B)
        result = (Condition("A") & Condition("B")).evaluate(self.frame)
        assert result == pytest.approx([0.0, 0.3, 0.2, 0.0])

    def test_union_is_the_maximum_s_norm(self) -> None:
        # A+B = max(A, B)
        result = (Condition("A") | Condition("B")).evaluate(self.frame)
        assert result == pytest.approx([1.0, 0.6, 0.7, 1.0])

    def test_double_negation_is_the_identity(self) -> None:
        assert (~~Condition("A")).evaluate(self.frame) == pytest.approx([0.0, 0.3, 0.7, 1.0])

    def test_operators_are_idempotent(self) -> None:
        a = Condition("A")
        assert (a & a).evaluate(self.frame) == pytest.approx([0.0, 0.3, 0.7, 1.0])
        assert (a | a).evaluate(self.frame) == pytest.approx([0.0, 0.3, 0.7, 1.0])

    def test_operators_are_commutative(self) -> None:
        a, b = Condition("A"), Condition("B")
        assert (a & b).evaluate(self.frame) == pytest.approx((b & a).evaluate(self.frame))
        assert (a | b).evaluate(self.frame) == pytest.approx((b | a).evaluate(self.frame))

    def test_boundary_values_behave_as_crisp_logic(self) -> None:
        crisp = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0], "B": [1.0, 0.0, 1.0, 0.0]})
        assert (Condition("A") & Condition("B")).evaluate(crisp) == pytest.approx(
            [1.0, 0.0, 0.0, 0.0]
        )
        assert (Condition("A") | Condition("B")).evaluate(crisp) == pytest.approx(
            [1.0, 1.0, 1.0, 0.0]
        )


# ---------------------------------------------------------------------------
# Sufficiency
# ---------------------------------------------------------------------------


class TestSufficiency:
    def test_consistency_of_a_perfect_subset_is_one(self) -> None:
        # X <= Y elementwise, so sum(min(X,Y)) == sum(X)
        assert sufficiency([0.2, 0.4], [0.5, 0.9]).consistency == pytest.approx(1.0)

    def test_consistency_is_hand_computable(self) -> None:
        # min(X,Y) = [0.3, 0.2]; sum = 0.5; sum(X) = 0.9+0.2 = 1.1
        fit = sufficiency([0.9, 0.2], [0.3, 0.8])
        assert fit.consistency == pytest.approx(0.5 / 1.1, abs=TOL)

    def test_raw_coverage_is_hand_computable(self) -> None:
        # min(X,Y) = [0.3, 0.2]; sum = 0.5; sum(Y) = 0.3+0.8 = 1.1
        fit = sufficiency([0.9, 0.2], [0.3, 0.8])
        assert fit.coverage == pytest.approx(0.5 / 1.1, abs=TOL)

    def test_pri_is_hand_computable(self) -> None:
        # min(X,Y)      = [0.3, 0.2]      sum = 0.5
        # min(X,Y,1-Y)  = [0.3, 0.2]      sum = 0.5   (1-Y = [0.7, 0.2])
        # PRI = (0.5 - 0.5) / (1.1 - 0.5) = 0
        fit = sufficiency([0.9, 0.2], [0.3, 0.8])
        assert fit.pri == pytest.approx(0.0, abs=TOL)

    def test_pri_of_a_perfectly_consistent_crisp_relation_is_one(self) -> None:
        # min(X,Y,1-Y) = 0 everywhere when Y is crisp and X <= Y
        assert sufficiency([1, 1, 0], [1, 1, 0]).pri == pytest.approx(1.0)

    def test_an_empty_cause_yields_zero_rather_than_dividing_by_zero(self) -> None:
        fit = sufficiency([0.0, 0.0], [0.5, 0.5])
        assert (fit.consistency, fit.coverage, fit.pri) == (0.0, 0.0, 0.0)

    def test_an_empty_outcome_yields_zero_coverage(self) -> None:
        fit = sufficiency([0.5, 0.5], [0.0, 0.0])
        assert fit.coverage == 0.0

    def test_a_single_case_is_admissible(self) -> None:
        assert sufficiency([0.8], [0.9]).consistency == pytest.approx(1.0)

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            sufficiency([0.5, 0.5], [0.5])


# ---------------------------------------------------------------------------
# Necessity
# ---------------------------------------------------------------------------


class TestNecessity:
    def test_consistency_of_a_perfect_superset_is_one(self) -> None:
        # Y <= X elementwise, so sum(min(X,Y)) == sum(Y)
        assert necessity([0.9, 0.7], [0.5, 0.2]).consistency == pytest.approx(1.0)

    def test_consistency_is_hand_computable(self) -> None:
        # min(X,Y) = [0.3, 0.2]; sum = 0.5; sum(Y) = 1.1
        fit = necessity([0.9, 0.2], [0.3, 0.8])
        assert fit.consistency == pytest.approx(0.5 / 1.1, abs=TOL)

    def test_relevance_of_necessity_is_hand_computable(self) -> None:
        # RoN = sum(1-X) / sum(1 - min(X,Y))
        # 1-X = [0.1, 0.8] -> 0.9 ; 1-min(X,Y) = [0.7, 0.8] -> 1.5
        fit = necessity([0.9, 0.2], [0.3, 0.8])
        assert fit.ron == pytest.approx(0.9 / 1.5, abs=TOL)

    def test_a_constantly_present_condition_has_trivial_relevance(self) -> None:
        # X = 1 everywhere: necessity consistency is 1 but RoN collapses to 0.
        fit = necessity([1.0, 1.0, 1.0], [0.9, 0.4, 0.1])
        assert fit.consistency == pytest.approx(1.0)
        assert fit.ron == pytest.approx(0.0)

    def test_an_empty_outcome_yields_zero_rather_than_dividing_by_zero(self) -> None:
        fit = necessity([0.5, 0.5], [0.0, 0.0])
        assert fit.consistency == 0.0

    def test_ron_denominator_of_zero_yields_zero(self) -> None:
        # X = Y = 1 everywhere, so sum(1 - min(X,Y)) == 0.
        assert necessity([1.0, 1.0], [1.0, 1.0]).ron == 0.0

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            necessity([0.5, 0.5], [0.5])


# ---------------------------------------------------------------------------
# Subset relations
# ---------------------------------------------------------------------------


class TestSubsetRelations:
    def test_sufficiency_expresses_the_subset_relation_x_in_y(self) -> None:
        assert sufficiency([0.2, 0.3], [0.9, 0.9]).consistency == pytest.approx(1.0)
        assert sufficiency([0.9, 0.9], [0.2, 0.3]).consistency < 1.0

    def test_necessity_expresses_the_superset_relation_y_in_x(self) -> None:
        assert necessity([0.9, 0.9], [0.2, 0.3]).consistency == pytest.approx(1.0)
        assert necessity([0.2, 0.3], [0.9, 0.9]).consistency < 1.0

    def test_the_two_relations_are_duals(self) -> None:
        x, y = [0.9, 0.2, 0.5], [0.3, 0.8, 0.5]
        assert necessity(x, y).consistency == pytest.approx(sufficiency(y, x).consistency)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_the_three_anchors_map_to_their_defining_memberships(self) -> None:
        result = calibrate_direct([10, 20, 30], full_out=10, crossover=20, full_in=30)
        assert result == pytest.approx([0.05, 0.5, 0.95], abs=TOL)

    def test_calibration_output_stays_inside_the_unit_interval(self) -> None:
        result = calibrate_direct([-1e6, 0, 50, 1e6], full_out=10, crossover=20, full_in=30)
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_crisp_calibration_is_right_continuous_at_the_threshold(self) -> None:
        # findInterval semantics: a value equal to the threshold falls above it.
        assert calibrate_crisp([9.999, 10.0, 10.001], [10.0]).tolist() == [0, 1, 1]


# ---------------------------------------------------------------------------
# Truth tables and minimisation
# ---------------------------------------------------------------------------


class TestTruthTableAndMinimisation:
    def test_a_complete_table_enumerates_two_to_the_k_rows(self) -> None:
        frame = pd.DataFrame({"A": [0.9, 0.1], "B": [0.9, 0.1], "C": [0.9, 0.1], "Y": [0.9, 0.1]})
        table = build_truth_table(frame, outcome="Y", conditions=["A", "B", "C"])
        assert len(table.rows) == 8

    def test_row_classification_partitions_the_table(self) -> None:
        frame = pd.DataFrame({"A": [0.9, 0.9, 0.1], "B": [0.9, 0.1, 0.1], "Y": [0.9, 0.2, 0.1]})
        table = build_truth_table(
            frame, outcome="Y", conditions=["A", "B"], inclusion_cutoff=0.8, exclusion_cutoff=0.3
        )
        groups = (
            table.positive_minterms
            | table.negative_minterms
            | table.contradictory_minterms
            | table.remainder_minterms
        )
        assert groups == {0, 1, 2, 3}
        total = (
            len(table.positive_minterms)
            + len(table.negative_minterms)
            + len(table.contradictory_minterms)
            + len(table.remainder_minterms)
        )
        assert total == 4, "the four codes must partition the table, not overlap"

    def test_minimisation_covers_every_positive_row(self) -> None:
        on_set = {3, 5, 6, 7}
        solution = minimize(on_set, width=3)[0]
        covered = {m for m in range(8) for i in solution.implicants if i.covers(m)}
        assert on_set <= covered

    def test_minimisation_excludes_every_negative_row(self) -> None:
        on_set = {3, 5, 6, 7}
        solution = minimize(on_set, width=3)[0]
        covered = {m for m in range(8) for i in solution.implicants if i.covers(m)}
        assert not covered & ({0, 1, 2, 4} - set())

    def test_remainders_are_usable_only_when_offered_as_dont_cares(self) -> None:
        conservative = minimize({6, 7}, width=3)[0]
        parsimonious = minimize({6, 7}, dont_cares={4, 5}, width=3)[0]
        assert conservative.literal_count > parsimonious.literal_count
