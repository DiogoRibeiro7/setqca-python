import numpy as np
import pytest

from setqca import DirectCalibration, calibrate_crisp, calibrate_direct


def test_direct_logistic_hits_three_anchors() -> None:
    values = np.array([10.0, 20.0, 30.0])
    calibrated = calibrate_direct(values, full_out=10, crossover=20, full_in=30)
    assert calibrated == pytest.approx([0.05, 0.5, 0.95], abs=1e-12)


def test_direct_logistic_supports_decreasing_sets() -> None:
    spec = DirectCalibration(full_out=30, crossover=20, full_in=10)
    calibrated = spec.transform([10, 20, 30])
    assert calibrated == pytest.approx([0.95, 0.5, 0.05], abs=1e-12)


def test_piecewise_calibration_hits_exact_endpoints() -> None:
    calibrated = calibrate_direct(
        [0, 25, 50, 75, 100], full_out=0, crossover=50, full_in=100, logistic=False
    )
    assert calibrated == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])


def test_piecewise_calibration_supports_decreasing_sets() -> None:
    calibrated = calibrate_direct(
        [0, 25, 50, 75, 100], full_out=100, crossover=50, full_in=0, logistic=False
    )
    assert calibrated == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])


def test_piecewise_exponents_shape_the_curve() -> None:
    """Raising the upper exponent accelerates the approach to full inclusion."""
    linear = calibrate_direct([75], full_out=0, crossover=50, full_in=100, logistic=False)
    curved = calibrate_direct(
        [75], full_out=0, crossover=50, full_in=100, logistic=False, above=2.0
    )
    assert linear[0] == pytest.approx(0.75)
    assert curved[0] == pytest.approx(0.875)
    assert 0.5 < linear[0] < curved[0] < 1.0


def test_piecewise_exponents_preserve_the_anchors() -> None:
    """Whatever the exponents, the three anchors still map to 0, 0.5 and 1."""
    calibrated = calibrate_direct(
        [0, 50, 100], full_out=0, crossover=50, full_in=100, logistic=False, below=3.0, above=2.0
    )
    assert calibrated == pytest.approx([0.0, 0.5, 1.0])


def test_logistic_calibration_is_stable_for_extreme_values() -> None:
    """Values far outside the anchors saturate instead of overflowing."""
    calibrated = calibrate_direct([-1e6, 1e6], full_out=10, crossover=20, full_in=30)
    assert calibrated == pytest.approx([0.0, 1.0])


def test_crisp_thresholds_match_find_interval_semantics() -> None:
    values = [0, 10, 20, 30]
    result = calibrate_crisp(values, [10, 20])
    assert result.tolist() == [0, 1, 2, 2]


def test_crisp_calibration_rejects_duplicate_thresholds() -> None:
    with pytest.raises(ValueError, match="unique"):
        calibrate_crisp([1, 2, 3], [2, 2])


def test_crisp_calibration_sorts_thresholds() -> None:
    assert calibrate_crisp([0, 15, 25], [20, 10]).tolist() == [0, 1, 2]
