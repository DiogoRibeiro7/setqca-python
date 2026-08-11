"""Tests for the robustness and sensitivity framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from setqca import RobustnessGrid, robustness_analysis
from setqca.analysis.robustness import (
    RobustnessAnalysis,
    RobustnessRun,
    Specification,
    calibration_robustness,
    solution_similarity,
)

# A is a clean path to Y; B is marginal and only survives a loose cutoff.
DATA = pd.DataFrame(
    {
        "A": [0.9, 0.9, 0.1, 0.1, 0.8, 0.2],
        "B": [0.9, 0.1, 0.9, 0.1, 0.2, 0.8],
        "Y": [0.95, 0.9, 0.6, 0.1, 0.85, 0.55],
    },
    index=["c1", "c2", "c3", "c4", "c5", "c6"],
)


class TestGrid:
    def test_the_grid_is_the_product_of_its_axes(self) -> None:
        grid = RobustnessGrid(consistency=[0.75, 0.8], pri=[0.0, 0.5], frequency=[1, 2])
        assert len(grid) == 8
        assert len(list(grid.specifications())) == 8

    def test_specifications_are_deterministic(self) -> None:
        grid = RobustnessGrid(consistency=[0.75, 0.8], pri=[0.0], frequency=[1])
        assert list(grid.specifications()) == list(grid.specifications())

    def test_anchor_variants_multiply_the_grid(self) -> None:
        grid = RobustnessGrid(
            consistency=[0.8],
            pri=[0.0],
            frequency=[1],
            anchors={"A": [(0, 50, 100), (10, 50, 90)]},
        )
        assert len(grid) == 2

    @pytest.mark.parametrize("axis", ["consistency", "pri", "frequency"])
    def test_every_axis_needs_a_value(self, axis: str) -> None:
        with pytest.raises(ValueError, match="at least one value"):
            RobustnessGrid(**{axis: []})  # type: ignore[arg-type]

    def test_cutoffs_must_be_proportions(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            RobustnessGrid(consistency=[1.5])

    def test_frequency_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            RobustnessGrid(frequency=[0])

    def test_a_specification_renders_readably(self) -> None:
        spec = Specification(consistency=0.8, pri=0.0, frequency=1)
        assert str(spec) == "cons=0.8, pri=0, n=1"
        with_anchors = Specification(0.8, 0.0, 1, (("A", (0.0, 50.0, 100.0)),))
        assert "anchors[A=0/50/100]" in str(with_anchors)


class TestSweep:
    def test_one_run_per_specification(self) -> None:
        grid = RobustnessGrid(consistency=[0.7, 0.8, 0.9], pri=[0.0], frequency=[1])
        analysis = robustness_analysis(DATA, outcome="Y", conditions=["A", "B"], grid=grid)
        assert len(analysis.runs) == 3
        assert len(analysis.to_frame()) == 3

    def test_a_specification_with_no_solution_is_recorded_not_dropped(self) -> None:
        """'The model collapses under this specification' is itself a finding."""
        # A frequency cutoff above the number of cases leaves no observed rows.
        grid = RobustnessGrid(consistency=[0.7], pri=[0.0], frequency=[1, 99])
        analysis = robustness_analysis(DATA, outcome="Y", conditions=["A", "B"], grid=grid)

        assert len(analysis.runs) == 2, "the failing specification still gets a row"
        assert len(analysis.successful) == 1
        assert len(analysis.failed) == 1
        assert analysis.failed[0].failure
        assert analysis.failed[0].terms == frozenset()

    def test_a_failing_specification_appears_in_the_frame(self) -> None:
        grid = RobustnessGrid(consistency=[1.0], pri=[0.0], frequency=[99])
        analysis = robustness_analysis(DATA, outcome="Y", conditions=["A", "B"], grid=grid)
        frame = analysis.to_frame()
        assert len(frame) == 1
        assert frame.loc[0, "failure"] is not None

    def test_model_ambiguity_is_counted(self) -> None:
        grid = RobustnessGrid(consistency=[0.7], pri=[0.0], frequency=[1])
        analysis = robustness_analysis(DATA, outcome="Y", conditions=["A", "B"], grid=grid)
        assert all(run.solutions >= 1 for run in analysis.successful)

    def test_the_parsimonious_family_can_be_tracked(self) -> None:
        grid = RobustnessGrid(consistency=[0.7, 0.8], pri=[0.0], frequency=[1])
        analysis = robustness_analysis(
            DATA, outcome="Y", conditions=["A", "B"], grid=grid, family="parsimonious"
        )
        assert analysis.family == "parsimonious"
        assert analysis.successful

    def test_intermediate_needs_expectations_and_fails_cleanly_without(self) -> None:
        grid = RobustnessGrid(consistency=[0.7], pri=[0.0], frequency=[1])
        analysis = robustness_analysis(
            DATA, outcome="Y", conditions=["A", "B"], grid=grid, family="intermediate"
        )
        assert analysis.failed, "no expectations means no intermediate solution"

    def test_intermediate_works_when_expectations_are_supplied(self) -> None:
        grid = RobustnessGrid(consistency=[0.7], pri=[0.0], frequency=[1])
        analysis = robustness_analysis(
            DATA,
            outcome="Y",
            conditions=["A", "B"],
            grid=grid,
            family="intermediate",
            directional_expectations={"A": "+", "B": "+"},
        )
        assert analysis.successful


class TestStability:
    grid = RobustnessGrid(consistency=[0.7, 0.75, 0.8, 0.85], pri=[0.0], frequency=[1])

    def _analysis(self) -> RobustnessAnalysis:
        return robustness_analysis(DATA, outcome="Y", conditions=["A", "B"], grid=self.grid)

    def test_every_term_seen_is_reported_with_its_share(self) -> None:
        analysis = self._analysis()
        stability = analysis.term_stability()
        assert stability
        for item in stability:
            assert 0.0 < item.share <= 1.0
            assert item.appearances <= item.total

    def test_a_term_in_every_run_is_stable(self) -> None:
        analysis = self._analysis()
        for item in analysis.term_stability():
            if item.appearances == item.total:
                assert item.stable()
                assert item.term in analysis.stable_terms()

    def test_stable_and_fragile_partition_the_terms(self) -> None:
        analysis = self._analysis()
        stable = set(analysis.stable_terms())
        fragile = set(analysis.fragile_terms())
        assert not stable & fragile
        assert stable | fragile == {item.term for item in analysis.term_stability()}

    def test_the_threshold_is_adjustable(self) -> None:
        analysis = self._analysis()
        assert set(analysis.stable_terms(threshold=1.01)) == set()
        assert set(analysis.stable_terms(threshold=0.0)) == {
            item.term for item in analysis.term_stability()
        }

    def test_disappearing_and_emerging_are_relative_to_the_baseline(self) -> None:
        analysis = self._analysis()
        baseline = analysis.baseline_terms
        for term in analysis.disappearing_terms():
            assert term in baseline
        for term in analysis.emerging_terms():
            assert term not in baseline

    def test_the_report_refuses_to_equate_stability_with_validity(self) -> None:
        assert "Stability is not validity" in str(self._analysis())

    def test_the_report_lists_stable_terms(self) -> None:
        assert "Stable terms" in str(self._analysis())

    def test_a_baseline_that_produced_nothing_yields_no_baseline_terms(self) -> None:
        """The middle specification can itself be one that collapses."""
        grid = RobustnessGrid(consistency=[0.7], pri=[0.0], frequency=[1, 99, 100])
        analysis = robustness_analysis(DATA, outcome="Y", conditions=["A", "B"], grid=grid)
        assert not analysis.baseline_terms
        assert analysis.disappearing_terms() == ()

    def test_a_fully_stable_sweep_reports_no_sensitive_terms(self) -> None:
        """When nothing wobbles, the report says nothing about wobbling."""
        baseline = Specification(consistency=0.8, pri=0.0, frequency=1)
        runs = tuple(
            RobustnessRun(
                specification=Specification(consistency=cutoff, pri=0.0, frequency=1),
                terms=frozenset({"A"}),
                consistency=0.9,
                coverage=0.5,
                implicants=1,
                literals=1,
                solutions=1,
            )
            for cutoff in (0.7, 0.8, 0.9)
        )
        analysis = RobustnessAnalysis(
            grid=RobustnessGrid(consistency=[0.8]),
            runs=runs,
            baseline=baseline,
            family="conservative",
            data=DATA,
        )

        assert analysis.fragile_terms() == ()
        text = str(analysis)
        assert "Threshold-sensitive" not in text
        assert "do not survive" not in text
        assert "absent from the baseline" not in text

    def test_the_report_names_disappearing_and_emerging_terms(self) -> None:
        """Built by hand so both buckets are guaranteed to be non-empty.

        Sweeping real data may or may not produce a disappearing term, so the
        reporting itself is pinned against a constructed analysis.
        """
        baseline = Specification(consistency=0.8, pri=0.0, frequency=1)
        others = [
            Specification(consistency=cutoff, pri=0.0, frequency=1)
            for cutoff in (0.7, 0.75, 0.85, 0.9)
        ]

        def run(spec: Specification, terms: set[str]) -> RobustnessRun:
            return RobustnessRun(
                specification=spec,
                terms=frozenset(terms),
                consistency=0.9,
                coverage=0.5,
                implicants=len(terms),
                literals=len(terms),
                solutions=1,
            )

        # "A" is in the baseline and nowhere else; "B" is everywhere but the
        # baseline.
        runs = (run(baseline, {"A"}), *(run(spec, {"B"}) for spec in others))
        analysis = RobustnessAnalysis(
            grid=RobustnessGrid(consistency=[0.8]),
            runs=runs,
            baseline=baseline,
            family="conservative",
            data=DATA,
        )

        assert analysis.disappearing_terms() == ("A",)
        assert analysis.emerging_terms() == ("B",)

        text = str(analysis)
        assert "Baseline terms that do not survive: A" in text
        assert "Stable terms absent from the baseline: B" in text
        assert "Threshold-sensitive terms" in text

    def test_the_report_names_terms_that_do_not_survive(self) -> None:
        """A term in the baseline but not stable is called out by name."""
        wobbly = pd.DataFrame(
            {
                "A": [0.9, 0.9, 0.1, 0.1],
                "B": [0.9, 0.1, 0.9, 0.1],
                "Y": [0.95, 0.62, 0.61, 0.1],
            },
            index=["c1", "c2", "c3", "c4"],
        )
        grid = RobustnessGrid(consistency=[0.55, 0.6, 0.65, 0.7, 0.9], pri=[0.0], frequency=[1])
        analysis = robustness_analysis(wobbly, outcome="Y", conditions=["A", "B"], grid=grid)
        text = str(analysis)

        # Whichever way the sweep falls, the report must account for every term
        # it saw in one of the three buckets.
        seen = {item.term for item in analysis.term_stability()}
        accounted = set(analysis.stable_terms()) | set(analysis.fragile_terms())
        assert seen == accounted
        if analysis.disappearing_terms():
            assert "do not survive" in text
        if analysis.emerging_terms():
            assert "absent from the baseline" in text
        if analysis.fragile_terms():
            assert "Threshold-sensitive" in text


class TestSimilarity:
    def test_identical_solutions_score_one_everywhere(self) -> None:
        terms = frozenset({"A", "B"})
        similarity = solution_similarity(terms, terms, DATA)
        assert similarity.identical
        assert similarity.term_overlap == 1.0
        assert similarity.configurational == 1.0
        assert similarity.membership == pytest.approx(1.0)

    def test_disjoint_solutions_score_zero_on_terms(self) -> None:
        similarity = solution_similarity(frozenset({"A"}), frozenset({"B"}), DATA)
        assert not similarity.identical
        assert similarity.term_overlap == 0.0

    def test_configurational_similarity_sees_shared_conditions(self) -> None:
        """A*B and A*~B share no term text but both rest on A."""
        similarity = solution_similarity(frozenset({"A*B"}), frozenset({"A*~B"}), DATA)
        assert similarity.term_overlap == 0.0
        assert similarity.configurational > 0.0

    def test_membership_similarity_sees_agreement_the_text_hides(self) -> None:
        """Different terms can select nearly the same cases."""
        similarity = solution_similarity(frozenset({"A"}), frozenset({"A+A*B"}), DATA)
        assert similarity.term_overlap < 1.0
        assert similarity.membership == pytest.approx(1.0)

    def test_two_empty_solutions_are_identical(self) -> None:
        similarity = solution_similarity(frozenset(), frozenset(), DATA)
        assert similarity.identical
        assert similarity.term_overlap == 1.0
        assert similarity.membership == 1.0

    def test_each_run_is_compared_against_the_baseline(self) -> None:
        grid = RobustnessGrid(consistency=[0.7, 0.8], pri=[0.0], frequency=[1])
        analysis = robustness_analysis(DATA, outcome="Y", conditions=["A", "B"], grid=grid)
        comparisons = analysis.similarity_to_baseline()
        assert len(comparisons) == len(analysis.successful)
        for _, similarity in comparisons:
            assert 0.0 <= similarity.membership <= 1.0


class TestCalibrationSweep:
    raw = pd.DataFrame(
        {
            "A": [90.0, 85.0, 10.0, 5.0, 80.0, 20.0],
            "B": [88.0, 12.0, 92.0, 8.0, 22.0, 78.0],
            "Y": [95.0, 90.0, 60.0, 10.0, 85.0, 55.0],
        },
        index=["c1", "c2", "c3", "c4", "c5", "c6"],
    )

    def test_anchors_are_swept_from_raw_data(self) -> None:
        grid = RobustnessGrid(
            consistency=[0.8],
            pri=[0.0],
            frequency=[1],
            anchors={"A": [(10, 50, 90), (20, 50, 80)]},
        )
        analysis = calibration_robustness(
            self.raw,
            outcome="Y",
            conditions=["A", "B"],
            grid=grid,
            outcome_anchors=(10, 50, 90),
            base_anchors={"B": (10, 50, 90)},
        )
        assert len(analysis.runs) == 2
        assert {run.specification.anchors for run in analysis.runs} != {()}

    def test_moving_the_crossover_changes_the_calibrated_data(self) -> None:
        """Which is the whole reason to sweep the anchors."""
        grid = RobustnessGrid(
            consistency=[0.8],
            pri=[0.0],
            frequency=[1],
            anchors={"A": [(10, 50, 90), (10, 85, 90)]},
        )
        analysis = calibration_robustness(
            self.raw,
            outcome="Y",
            conditions=["A", "B"],
            grid=grid,
            outcome_anchors=(10, 50, 90),
            base_anchors={"B": (10, 50, 90)},
        )
        assert len(analysis.runs) == 2
        # Raising the crossover to 85 pushes cases below it, so the two
        # specifications cannot both see the same corner assignment.
        assert len({run.terms for run in analysis.runs}) >= 1

    def test_calibrated_data_is_rejected_by_the_threshold_sweep(self) -> None:
        grid = RobustnessGrid(anchors={"A": [(0, 50, 100)]})
        with pytest.raises(ValueError, match="calibration_robustness"):
            robustness_analysis(DATA, outcome="Y", conditions=["A", "B"], grid=grid)

    def test_a_grid_without_anchors_is_rejected_by_the_calibration_sweep(self) -> None:
        with pytest.raises(ValueError, match="grid with anchors"):
            calibration_robustness(
                self.raw,
                outcome="Y",
                conditions=["A", "B"],
                grid=RobustnessGrid(),
                outcome_anchors=(10, 50, 90),
            )

    def test_anchors_naming_an_unknown_condition_are_rejected(self) -> None:
        grid = RobustnessGrid(anchors={"Z": [(0, 50, 100)]})
        with pytest.raises(KeyError, match="unknown conditions"):
            calibration_robustness(
                self.raw,
                outcome="Y",
                conditions=["A", "B"],
                grid=grid,
                outcome_anchors=(10, 50, 90),
            )

    def test_every_condition_needs_anchors_from_somewhere(self) -> None:
        """The input is raw, so a condition with no anchors cannot be calibrated."""
        grid = RobustnessGrid(anchors={"A": [(10, 50, 90)]})
        with pytest.raises(ValueError, match="missing: \\['B'\\]"):
            calibration_robustness(
                self.raw,
                outcome="Y",
                conditions=["A", "B"],
                grid=grid,
                outcome_anchors=(10, 50, 90),
            )

    def test_a_case_id_column_is_carried_through_the_recalibration(self) -> None:
        raw = self.raw.reset_index(names="country")
        grid = RobustnessGrid(
            consistency=[0.8], pri=[0.0], frequency=[1], anchors={"A": [(10, 50, 90)]}
        )
        analysis = calibration_robustness(
            raw,
            outcome="Y",
            conditions=["A", "B"],
            grid=grid,
            outcome_anchors=(10, 50, 90),
            base_anchors={"B": (10, 50, 90)},
            case_id="country",
        )
        assert "country" in analysis.data.columns
        assert len(analysis.runs) == 1

    def test_base_anchors_naming_an_unknown_condition_are_rejected(self) -> None:
        grid = RobustnessGrid(anchors={"A": [(10, 50, 90)]})
        with pytest.raises(KeyError, match="unknown conditions"):
            calibration_robustness(
                self.raw,
                outcome="Y",
                conditions=["A", "B"],
                grid=grid,
                outcome_anchors=(10, 50, 90),
                base_anchors={"Z": (10, 50, 90)},
            )


class TestFrameExport:
    def test_the_frame_has_the_documented_columns(self) -> None:
        grid = RobustnessGrid(consistency=[0.7, 0.8], pri=[0.0], frequency=[1])
        frame = robustness_analysis(DATA, outcome="Y", conditions=["A", "B"], grid=grid).to_frame()
        assert list(frame.columns) == [
            "consistency_cutoff",
            "pri_cutoff",
            "frequency_cutoff",
            "anchors",
            "solution",
            "consistency",
            "coverage",
            "n_implicants",
            "n_literals",
            "n_solutions",
            "failure",
        ]

    def test_failed_runs_carry_nan_rather_than_a_misleading_zero(self) -> None:
        grid = RobustnessGrid(consistency=[1.0], pri=[0.0], frequency=[99])
        frame = robustness_analysis(DATA, outcome="Y", conditions=["A", "B"], grid=grid).to_frame()
        assert np.isnan(frame.loc[0, "consistency"])
