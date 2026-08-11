"""Tests for multi-value QCA.

The central claim under test is that the multi-value engine minimises cubes
directly rather than reducing to Boolean indicators, and that its answers are
exact.
"""

from __future__ import annotations

from itertools import combinations, product

import pandas as pd
import pytest

from setqca.multivalue import (
    MVQCA,
    MultiValueCube,
    MultiValueDomain,
    build_multivalue_truth_table,
    minimize_multivalue,
    prime_cubes,
)

# Two conditions: a three-level regime type and a binary wealth indicator.
DOMAIN = MultiValueDomain(("regime", "wealth"), (3, 2))


class TestDomain:
    def test_the_property_space_is_the_product_of_the_levels(self) -> None:
        assert DOMAIN.size == 6
        assert len(list(DOMAIN.configurations())) == 6

    def test_indexing_is_mixed_radix_and_big_endian(self) -> None:
        assert DOMAIN.index_of((0, 0)) == 0
        assert DOMAIN.index_of((0, 1)) == 1
        assert DOMAIN.index_of((1, 0)) == 2
        assert DOMAIN.index_of((2, 1)) == 5

    def test_indices_round_trip(self) -> None:
        for configuration in DOMAIN.configurations():
            assert DOMAIN.values_of(DOMAIN.index_of(configuration)) == configuration

    def test_a_binary_domain_reproduces_the_minterm_encoding(self) -> None:
        """With two levels everywhere, mixed radix is the ordinary minterm."""
        binary = MultiValueDomain(("A", "B", "C"), (2, 2, 2))
        assert binary.index_of((1, 1, 0)) == 6
        assert binary.values_of(5) == (1, 0, 1)

    def test_a_value_outside_its_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside the range"):
            DOMAIN.index_of((3, 0))

    def test_an_index_outside_the_space_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside the property space"):
            DOMAIN.values_of(6)

    def test_the_wrong_number_of_values_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Expected 2 values"):
            DOMAIN.index_of((0,))

    @pytest.mark.parametrize(
        ("conditions", "levels", "message"),
        [
            ((), (), "At least one condition"),
            (("A",), (2, 3), "same length"),
            (("A", "A"), (2, 2), "unique"),
            (("A",), (1,), "at least two levels"),
        ],
    )
    def test_malformed_domains_are_rejected(
        self, conditions: tuple[str, ...], levels: tuple[int, ...], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            MultiValueDomain(conditions, levels)

    def test_a_domain_renders_in_multi_value_notation(self) -> None:
        assert str(DOMAIN) == "regime{0,1,2}, wealth{0,1}"

    def test_a_domain_can_be_built_from_a_mapping(self) -> None:
        assert MultiValueDomain.from_mapping({"regime": 3, "wealth": 2}) == DOMAIN


class TestCubes:
    def test_a_configuration_cube_covers_only_itself(self) -> None:
        cube = MultiValueCube.from_configuration((1, 0))
        assert cube.covers(DOMAIN.index_of((1, 0)), DOMAIN)
        assert not cube.covers(DOMAIN.index_of((2, 0)), DOMAIN)

    def test_merging_takes_the_union_at_the_one_differing_condition(self) -> None:
        left = MultiValueCube.from_configuration((0, 1))
        right = MultiValueCube.from_configuration((2, 1))
        merged = left.merge(right)
        assert merged is not None
        assert merged.pattern[0] == frozenset({0, 2})
        assert merged.pattern[1] == frozenset({1})

    def test_a_merged_cube_covers_exactly_the_union(self) -> None:
        """The property the merge rule rests on: no extra coverage appears."""
        left = MultiValueCube.from_configuration((0, 1))
        right = MultiValueCube.from_configuration((2, 1))
        merged = left.merge(right)
        assert merged is not None
        covered = {i for i in range(DOMAIN.size) if merged.covers(i, DOMAIN)}
        assert covered == {DOMAIN.index_of((0, 1)), DOMAIN.index_of((2, 1))}

    def test_cubes_differing_at_two_conditions_do_not_merge(self) -> None:
        left = MultiValueCube.from_configuration((0, 0))
        right = MultiValueCube.from_configuration((1, 1))
        assert left.merge(right) is None

    def test_identical_cubes_do_not_merge(self) -> None:
        cube = MultiValueCube.from_configuration((0, 0))
        assert cube.merge(cube) is None

    def test_a_condition_allowing_every_level_is_not_a_literal(self) -> None:
        cube = MultiValueCube((frozenset({0, 1, 2}), frozenset({1})))
        assert cube.literals(DOMAIN) == 1
        assert cube.as_expression(DOMAIN) == "wealth{1}"

    def test_a_cube_constraining_nothing_renders_as_one(self) -> None:
        cube = MultiValueCube((frozenset({0, 1, 2}), frozenset({0, 1})))
        assert cube.is_tautology(DOMAIN)
        assert cube.as_expression(DOMAIN) == "1"

    def test_containment_is_per_condition(self) -> None:
        wide = MultiValueCube((frozenset({0, 1}), frozenset({0, 1})))
        narrow = MultiValueCube((frozenset({0}), frozenset({1})))
        assert wide.contains(narrow)
        assert not narrow.contains(wide)


class TestMinimisation:
    def test_all_levels_of_a_condition_collapse_it(self) -> None:
        """Regime taking every level, with wealth fixed, eliminates regime."""
        on_set = {DOMAIN.index_of((level, 1)) for level in range(3)}
        solutions = minimize_multivalue(on_set, domain=DOMAIN)
        assert len(solutions) == 1
        assert solutions[0].as_expression(DOMAIN) == "wealth{1}"

    def test_a_partial_set_of_levels_survives_as_a_subset_literal(self) -> None:
        """Two of three levels cannot eliminate the condition, but do combine."""
        on_set = {DOMAIN.index_of((0, 1)), DOMAIN.index_of((2, 1))}
        solutions = minimize_multivalue(on_set, domain=DOMAIN)
        assert len(solutions) == 1
        assert solutions[0].as_expression(DOMAIN) == "regime{0,2}*wealth{1}"

    def test_a_single_configuration_is_its_own_solution(self) -> None:
        on_set = {DOMAIN.index_of((1, 0))}
        solutions = minimize_multivalue(on_set, domain=DOMAIN)
        assert solutions[0].as_expression(DOMAIN) == "regime{1}*wealth{0}"

    def test_an_empty_problem_yields_the_empty_cover(self) -> None:
        assert minimize_multivalue(set(), domain=DOMAIN)[0].cubes == ()

    def test_a_problem_with_nothing_at_all_yields_no_primes(self) -> None:
        assert prime_cubes(set(), set(), DOMAIN) == ()

    def test_rows_know_whether_they_were_observed(self) -> None:
        table = build_multivalue_truth_table(
            pd.DataFrame({"regime": [0, 1], "wealth": [0, 1], "Y": [0.1, 0.9]}),
            outcome="Y",
            conditions=["regime", "wealth"],
        )
        observed = [row for row in table.rows if row.observed]
        assert len(observed) == 2
        assert all(row.frequency > 0 for row in observed)

    def test_remainders_permit_a_simpler_cover(self) -> None:
        on_set = {DOMAIN.index_of((0, 1)), DOMAIN.index_of((1, 1))}
        remainder = {DOMAIN.index_of((2, 1))}
        conservative = minimize_multivalue(on_set, domain=DOMAIN)[0]
        parsimonious = minimize_multivalue(on_set, domain=DOMAIN, dont_cares=remainder)[0]

        assert parsimonious.literal_count(DOMAIN) < conservative.literal_count(DOMAIN)
        assert parsimonious.as_expression(DOMAIN) == "wealth{1}"

    def test_overlapping_sets_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="disjoint"):
            prime_cubes({0, 1}, {1}, DOMAIN)

    def test_every_returned_cover_is_valid_and_tied_on_cost(self) -> None:
        on_set = {DOMAIN.index_of((0, 0)), DOMAIN.index_of((1, 1)), DOMAIN.index_of((2, 0))}
        off_set = set(range(DOMAIN.size)) - on_set
        solutions = minimize_multivalue(on_set, domain=DOMAIN)

        costs = {(len(item.cubes), item.literal_count(DOMAIN)) for item in solutions}
        assert len(costs) == 1
        for solution in solutions:
            covered = {i for i in range(DOMAIN.size) if solution.covers(i, DOMAIN)}
            assert on_set <= covered
            assert not covered & off_set


class TestExactness:
    """Verified against exhaustive search, as the binary engine is."""

    @pytest.mark.parametrize("levels", [(2, 2), (3, 2), (2, 3), (3, 3)])
    def test_minimisation_matches_brute_force(self, levels: tuple[int, ...]) -> None:
        domain = MultiValueDomain(tuple("AB"[: len(levels)]), levels)
        universe = list(range(domain.size))

        # Every legal cube: a non-empty subset of levels per condition.
        options = [
            [
                frozenset(subset)
                for size in range(1, count + 1)
                for subset in combinations(range(count), size)
            ]
            for count in levels
        ]
        all_cubes = [MultiValueCube(tuple(pattern)) for pattern in product(*options)]

        for size in (1, 2, 3):
            for on_tuple in combinations(universe, size):
                on_set = set(on_tuple)
                off_set = set(universe) - on_set

                legal = [
                    cube
                    for cube in all_cubes
                    if not any(cube.covers(i, domain) for i in off_set)
                    and any(cube.covers(i, domain) for i in on_set)
                ]
                expected: tuple[int, int] | None = None
                for count in range(1, len(legal) + 1):
                    candidates = [
                        (count, sum(cube.literals(domain) for cube in choice))
                        for choice in combinations(legal, count)
                        if on_set
                        <= {i for i in universe for cube in choice if cube.covers(i, domain)}
                    ]
                    if candidates:
                        expected = min(candidates)
                        break

                solution = minimize_multivalue(on_set, domain=domain)[0]
                obtained = (len(solution.cubes), solution.literal_count(domain))
                assert obtained == expected, f"levels={levels} on_set={sorted(on_set)}"


class TestNotBooleanDummies:
    """The encoding shortcut this implementation deliberately avoids."""

    def test_a_three_level_condition_is_not_split_into_indicators(self) -> None:
        on_set = {DOMAIN.index_of((0, 1)), DOMAIN.index_of((2, 1))}
        expression = minimize_multivalue(on_set, domain=DOMAIN)[0].as_expression(DOMAIN)

        # A dummy encoding would express this through indicator variables such
        # as regime_0 and regime_2; the multi-value form keeps one literal.
        assert "regime{0,2}" in expression
        assert "regime_0" not in expression

    def test_no_cube_can_describe_an_impossible_case(self) -> None:
        """Every cube covers only real configurations, by construction.

        A Boolean dummy encoding admits points where two indicators for the
        same condition are both true, which correspond to no case at all.
        """
        on_set = set(range(DOMAIN.size))
        for cube in prime_cubes(on_set, set(), DOMAIN):
            covered = [i for i in range(DOMAIN.size) if cube.covers(i, DOMAIN)]
            for index in covered:
                values = DOMAIN.values_of(index)
                assert len(values) == DOMAIN.width
                for value, count in zip(values, DOMAIN.levels, strict=True):
                    assert 0 <= value < count


class TestTruthTable:
    data = pd.DataFrame(
        {
            "regime": [0, 0, 1, 1, 2, 2],
            "wealth": [0, 0, 1, 1, 1, 1],
            "Y": [0.1, 0.2, 0.9, 0.95, 0.85, 0.9],
        },
        index=["a", "b", "c", "d", "e", "f"],
    )

    def test_the_table_covers_the_whole_property_space(self) -> None:
        table = build_multivalue_truth_table(
            self.data, outcome="Y", conditions=["regime", "wealth"]
        )
        assert len(table.rows) == 6
        assert table.domain.levels == (3, 2)

    def test_unobserved_configurations_are_remainders(self) -> None:
        table = build_multivalue_truth_table(
            self.data, outcome="Y", conditions=["regime", "wealth"]
        )
        assert table.remainder_indices
        for row in table.rows_with("R"):
            assert row.frequency == 0
            assert "frequency" in (row.exclusion_reason or "")

    def test_consistency_is_the_share_of_the_outcome_in_the_configuration(self) -> None:
        table = build_multivalue_truth_table(
            self.data, outcome="Y", conditions=["regime", "wealth"]
        )
        row = next(row for row in table.rows if row.configuration == (1, 1))
        assert row.frequency == 2
        assert row.consistency == pytest.approx((0.9 + 0.95) / 2)

    def test_declared_levels_enlarge_the_property_space(self) -> None:
        """A level with no cases still exists, and becomes a remainder."""
        table = build_multivalue_truth_table(
            self.data,
            outcome="Y",
            conditions=["regime", "wealth"],
            levels={"regime": 4, "wealth": 2},
        )
        assert table.domain.levels == (4, 2)
        assert len(table.rows) == 8

    def test_declared_levels_may_not_contradict_the_data(self) -> None:
        with pytest.raises(ValueError, match="declares 2 levels"):
            build_multivalue_truth_table(
                self.data,
                outcome="Y",
                conditions=["regime", "wealth"],
                levels={"regime": 2, "wealth": 2},
            )

    def test_missing_declared_levels_are_reported(self) -> None:
        with pytest.raises(KeyError, match="Levels missing"):
            build_multivalue_truth_table(
                self.data,
                outcome="Y",
                conditions=["regime", "wealth"],
                levels={"regime": 3},
            )

    def test_fuzzy_conditions_are_rejected(self) -> None:
        frame = pd.DataFrame({"regime": [0.5, 1.0], "Y": [0.9, 0.1]})
        with pytest.raises(ValueError, match="categorical, not fuzzy"):
            build_multivalue_truth_table(frame, outcome="Y", conditions=["regime"])

    def test_negative_category_codes_are_rejected(self) -> None:
        frame = pd.DataFrame({"regime": [-1, 1], "Y": [0.9, 0.1]})
        with pytest.raises(ValueError, match="negative category code"):
            build_multivalue_truth_table(frame, outcome="Y", conditions=["regime"])

    def test_the_frame_export_names_the_conditions(self) -> None:
        frame = build_multivalue_truth_table(
            self.data, outcome="Y", conditions=["regime", "wealth"]
        ).to_frame()
        assert list(frame.columns)[:2] == ["regime", "wealth"]
        assert "OUT" in frame.columns

    def test_a_table_with_nothing_sufficient_refuses_to_minimise(self) -> None:
        table = build_multivalue_truth_table(
            self.data, outcome="Y", conditions=["regime", "wealth"], inclusion_cutoff=1.0
        )
        with pytest.raises(ValueError, match="No configuration is sufficient"):
            table.minimize()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"inclusion_cutoff": 1.5}, "inclusion_cutoff"),
            ({"frequency_cutoff": 0}, "frequency_cutoff"),
            ({"conditions": []}, "At least one condition"),
        ],
    )
    def test_guards(self, kwargs: dict[str, object], message: str) -> None:
        arguments: dict[str, object] = {
            "outcome": "Y",
            "conditions": ["regime", "wealth"],
        }
        arguments.update(kwargs)
        with pytest.raises(ValueError, match=message):
            build_multivalue_truth_table(self.data, **arguments)  # type: ignore[arg-type]

    def test_data_must_be_a_frame(self) -> None:
        with pytest.raises(TypeError, match="pandas DataFrame"):
            build_multivalue_truth_table({"a": [1]}, outcome="Y", conditions=["a"])  # type: ignore[arg-type]


class TestEstimator:
    data = TestTruthTable.data

    def test_the_estimator_returns_both_families(self) -> None:
        result = MVQCA(consistency=0.8).fit(self.data, outcome="Y", conditions=["regime", "wealth"])
        assert result.conservative
        assert result.parsimonious

    def test_the_parsimonious_solution_is_no_more_complex(self) -> None:
        result = MVQCA(consistency=0.8).fit(self.data, outcome="Y", conditions=["regime", "wealth"])
        assert min(item.literal_count(result.domain) for item in result.parsimonious) <= min(
            item.literal_count(result.domain) for item in result.conservative
        )

    def test_the_summary_frame_carries_fit(self) -> None:
        result = MVQCA(consistency=0.8).fit(self.data, outcome="Y", conditions=["regime", "wealth"])
        frame = result.summary_frame("parsimonious")
        assert list(frame.columns) == [
            "solution",
            "n_cubes",
            "n_literals",
            "consistency",
            "coverage",
        ]
        assert (frame["consistency"] >= 0.8).all()

    def test_an_unknown_family_is_rejected(self) -> None:
        result = MVQCA().fit(self.data, outcome="Y", conditions=["regime", "wealth"])
        with pytest.raises(ValueError, match="Unknown solution kind"):
            result.summary_frame("intermediate")

    def test_the_report_names_the_property_space(self) -> None:
        result = MVQCA().fit(self.data, outcome="Y", conditions=["regime", "wealth"])
        text = str(result)
        assert "Multi-value" in text
        assert "regime{0,1,2}" in text

    def test_case_labels_can_come_from_a_column(self) -> None:
        frame = self.data.reset_index(names="country")
        result = MVQCA().fit(frame, outcome="Y", conditions=["regime", "wealth"], case_id="country")
        assert any(row.cases for row in result.truth_table.rows)

    def test_declared_levels_flow_through_the_estimator(self) -> None:
        result = MVQCA(levels={"regime": 4, "wealth": 2}).fit(
            self.data, outcome="Y", conditions=["regime", "wealth"]
        )
        assert result.domain.levels == (4, 2)


class TestBinaryAgreement:
    """With two levels everywhere, mvQCA must reproduce csQCA."""

    def test_a_binary_problem_gives_the_same_minimal_cost(self) -> None:
        from setqca.minimize import minimize

        domain = MultiValueDomain(("A", "B", "C"), (2, 2, 2))
        for on_tuple in combinations(range(8), 3):
            on_set = set(on_tuple)
            binary = minimize(on_set, width=3)[0]
            multi = minimize_multivalue(on_set, domain=domain)[0]

            assert (len(multi.cubes), multi.literal_count(domain)) == (
                len(binary.implicants),
                binary.literal_count,
            )
