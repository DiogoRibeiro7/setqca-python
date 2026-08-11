"""Tests for calibration specifications, diagnostics and anchor helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from setqca.calibration import (
    CalibrationMethod,
    CalibrationSpec,
    calibrate,
    crisp_spec,
    diagnose_calibration,
    diagnose_frame,
    direct_spec,
    indirect_spec,
    suggest_anchors,
)

RAW = [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0]


class TestDirectSpec:
    def test_a_direct_spec_reproduces_the_function(self) -> None:
        from setqca import calibrate_direct

        spec = direct_spec("x", full_out=20, crossover=50, full_in=80)
        assert spec.apply(RAW) == pytest.approx(
            calibrate_direct(RAW, full_out=20, crossover=50, full_in=80)
        )

    def test_bad_anchors_fail_when_the_spec_is_written(self) -> None:
        """A spec is validated eagerly, not when it eventually meets data."""
        with pytest.raises(ValueError, match="strictly ordered"):
            direct_spec("x", full_out=50, crossover=50, full_in=80)

    def test_the_direct_method_requires_anchors(self) -> None:
        with pytest.raises(ValueError, match="requires anchors"):
            CalibrationSpec(condition="x", method=CalibrationMethod.DIRECT)

    def test_anchors_can_be_swapped_for_sensitivity_work(self) -> None:
        spec = direct_spec("x", full_out=20, crossover=50, full_in=80)
        moved = spec.with_anchors((10, 50, 90))
        assert moved.anchors == (10, 50, 90)
        assert spec.anchors == (20, 50, 80), "the original is untouched"


class TestCrispSpec:
    def test_one_threshold_gives_a_binary_set(self) -> None:
        spec = crisp_spec("x", thresholds=(50.0,))
        assert spec.apply(RAW).tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    def test_several_thresholds_are_rescaled_onto_the_unit_interval(self) -> None:
        """Downstream code expects memberships, not raw category indices."""
        spec = crisp_spec("x", thresholds=(25.0, 75.0))
        values = spec.apply(RAW)
        assert set(np.unique(values)) <= {0.0, 0.5, 1.0}
        assert values.max() == 1.0

    def test_the_crisp_method_requires_thresholds(self) -> None:
        with pytest.raises(ValueError, match="requires thresholds"):
            CalibrationSpec(condition="x", method=CalibrationMethod.CRISP)


class TestIndirectSpec:
    def test_points_are_interpolated_linearly(self) -> None:
        spec = indirect_spec("x", mapping=((0.0, 0.0), (50.0, 0.5), (100.0, 1.0)))
        assert spec.apply([0, 25, 50, 75, 100]) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])

    def test_values_beyond_the_ends_are_held_flat(self) -> None:
        spec = indirect_spec("x", mapping=((10.0, 0.0), (90.0, 1.0)))
        assert spec.apply([-100, 200]) == pytest.approx([0.0, 1.0])

    def test_a_plateau_the_direct_form_cannot_express(self) -> None:
        """Theory sometimes says 'no change across this range'."""
        spec = indirect_spec("x", mapping=((0.0, 0.0), (30.0, 0.5), (70.0, 0.5), (100.0, 1.0)))
        assert spec.apply([30, 50, 70]) == pytest.approx([0.5, 0.5, 0.5])

    def test_at_least_two_points_are_required(self) -> None:
        with pytest.raises(ValueError, match="at least two mapping points"):
            indirect_spec("x", mapping=((0.0, 0.0),))

    def test_raw_values_must_increase(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            indirect_spec("x", mapping=((10.0, 0.0), (10.0, 1.0)))

    def test_the_mapping_may_not_decrease(self) -> None:
        with pytest.raises(ValueError, match="non-decreasing"):
            indirect_spec("x", mapping=((0.0, 1.0), (10.0, 0.0)))

    def test_memberships_must_be_calibrated(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            indirect_spec("x", mapping=((0.0, 0.0), (10.0, 1.5)))


class TestIdentitySpec:
    def test_already_calibrated_values_pass_through(self) -> None:
        spec = CalibrationSpec(condition="x", method=CalibrationMethod.IDENTITY)
        assert spec.apply([0.0, 0.5, 1.0]).tolist() == [0.0, 0.5, 1.0]

    def test_uncalibrated_values_are_still_rejected(self) -> None:
        spec = CalibrationSpec(condition="x", method=CalibrationMethod.IDENTITY)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            spec.apply([0.0, 1.5])


class TestSerialisation:
    @pytest.mark.parametrize(
        ("spec", "sample"),
        [
            (direct_spec("x", full_out=20, crossover=50, full_in=80, note="theory"), RAW),
            (
                direct_spec("x", full_out=80, crossover=50, full_in=20, logistic=False, above=2.0),
                RAW,
            ),
            (crisp_spec("x", thresholds=(25.0, 75.0)), RAW),
            (indirect_spec("x", mapping=((0.0, 0.0), (50.0, 0.5), (100.0, 1.0))), RAW),
            # The identity method takes memberships, not raw measures.
            (CalibrationSpec(condition="x", method=CalibrationMethod.IDENTITY), [0.0, 0.4, 1.0]),
        ],
    )
    def test_specs_round_trip_through_json(
        self, spec: CalibrationSpec, sample: list[float]
    ) -> None:
        restored = CalibrationSpec.from_json(spec.to_json())
        assert restored == spec
        assert restored.apply(sample) == pytest.approx(spec.apply(sample))

    def test_the_note_survives_the_round_trip(self) -> None:
        """The reason for a choice is part of the specification."""
        spec = direct_spec("x", full_out=20, crossover=50, full_in=80, note="OECD threshold")
        assert CalibrationSpec.from_json(spec.to_json()).note == "OECD threshold"

    def test_an_unknown_method_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown calibration method"):
            CalibrationSpec.from_dict({"condition": "x", "method": "telepathy"})

    def test_a_payload_without_a_condition_is_rejected(self) -> None:
        with pytest.raises(KeyError):
            CalibrationSpec.from_dict({"method": "identity"})


class TestDiagnostics:
    def test_a_healthy_vector_reports_no_issues(self) -> None:
        values = [0.05, 0.2, 0.35, 0.65, 0.8, 0.95]
        diagnostics = diagnose_calibration(values)
        assert diagnostics.warnings == ()
        assert diagnostics.usable

    def test_exact_crossover_membership_makes_a_vector_unusable(self) -> None:
        diagnostics = diagnose_calibration([0.5, 0.9, 0.1])
        assert diagnostics.at_crossover == 1
        assert not diagnostics.usable
        assert any("exactly at 0.5" in warning for warning in diagnostics.warnings)

    def test_pile_up_at_the_crossover_is_reported(self) -> None:
        diagnostics = diagnose_calibration([0.48, 0.49, 0.51, 0.52, 0.9, 0.1])
        assert diagnostics.pile_up_share > 0.25
        assert any("of the crossover" in warning for warning in diagnostics.warnings)

    def test_compression_to_the_extremes_is_reported(self) -> None:
        diagnostics = diagnose_calibration([0.0, 0.01, 0.99, 1.0, 1.0, 0.0])
        assert diagnostics.compression_share > 0.9
        assert any("close to" in warning for warning in diagnostics.warnings)

    def test_low_variance_is_reported(self) -> None:
        diagnostics = diagnose_calibration([0.60, 0.61, 0.62, 0.60, 0.61])
        assert any("barely varies" in warning for warning in diagnostics.warnings)

    def test_a_condition_that_is_never_present_is_reported(self) -> None:
        diagnostics = diagnose_calibration([0.1, 0.2, 0.3, 0.05])
        assert diagnostics.above_crossover == 0
        assert any("never present" in warning for warning in diagnostics.warnings)

    def test_a_condition_that_is_always_present_is_reported(self) -> None:
        diagnostics = diagnose_calibration([0.9, 0.8, 0.95, 0.85])
        assert any("never absent" in warning for warning in diagnostics.warnings)

    def test_uncalibrated_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            diagnose_calibration([0.5, 1.5])

    def test_the_report_lists_warnings_or_says_there_are_none(self) -> None:
        assert "no issues found" in str(diagnose_calibration([0.05, 0.3, 0.7, 0.95]))
        assert "warning:" in str(diagnose_calibration([0.5, 0.9, 0.1]))

    def test_a_whole_frame_can_be_diagnosed(self) -> None:
        frame = pd.DataFrame({"A": [0.1, 0.9, 0.2], "B": [0.5, 0.5, 0.5]})
        summary = diagnose_frame(frame)
        assert list(summary["condition"]) == ["A", "B"]
        assert not summary.loc[1, "usable"]

    def test_specific_columns_can_be_selected(self) -> None:
        frame = pd.DataFrame({"A": [0.1, 0.9], "B": [0.2, 0.8], "Y": [0.3, 0.7]})
        assert list(diagnose_frame(frame, ["A", "B"])["condition"]) == ["A", "B"]


class TestAnchorSuggestions:
    def test_quantiles_are_reported_with_a_caveat(self) -> None:
        suggestion = suggest_anchors(list(range(101)))
        assert suggestion.anchors[0] < suggestion.anchors[1] < suggestion.anchors[2]
        assert "not a substitute" in suggestion.caveat
        assert "not a substitute" in str(suggestion)

    def test_the_quantiles_are_adjustable(self) -> None:
        suggestion = suggest_anchors(list(range(101)), quantiles=(0.1, 0.5, 0.9))
        assert suggestion.quantiles == (0.1, 0.5, 0.9)
        assert suggestion.values == pytest.approx((10.0, 50.0, 90.0))

    def test_quantiles_must_increase(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            suggest_anchors([1, 2, 3], quantiles=(0.5, 0.5, 0.9))

    def test_quantiles_must_be_probabilities(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            suggest_anchors([1, 2, 3], quantiles=(-0.1, 0.5, 0.9))

    def test_a_concentrated_variable_cannot_yield_distinct_anchors(self) -> None:
        with pytest.raises(ValueError, match="not distinct"):
            suggest_anchors([5.0] * 20)


class TestCalibrateEntryPoint:
    def test_values_spec_and_diagnostics_come_back_together(self) -> None:
        spec = direct_spec("innovation", full_out=20, crossover=50, full_in=80)
        result = calibrate(RAW, spec)
        assert result.spec is spec
        assert len(result.values) == len(RAW)
        assert result.diagnostics.n == len(RAW)

    def test_the_result_exports_as_a_named_frame(self) -> None:
        spec = direct_spec("innovation", full_out=20, crossover=50, full_in=80)
        frame = calibrate(RAW, spec).to_frame()
        assert list(frame.columns) == ["innovation"]

    def test_the_report_names_the_condition(self) -> None:
        spec = direct_spec("innovation", full_out=20, crossover=50, full_in=80)
        assert "Calibration of innovation" in str(calibrate(RAW, spec))

    def test_diagnostics_travel_with_a_bad_calibration(self) -> None:
        """A calibration that ruins the analysis still returns, and says why."""
        spec = crisp_spec("x", thresholds=(50.0,))
        result = calibrate([10.0, 20.0, 30.0], spec)
        assert result.diagnostics.warnings
        assert any("never present" in warning for warning in result.diagnostics.warnings)
