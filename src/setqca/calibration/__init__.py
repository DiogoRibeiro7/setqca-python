"""Calibration: turning raw measures into set memberships.

Calibration is where substantive knowledge enters a QCA, and where a result is
most easily manufactured. The primitives are here, along with reproducible
specifications, diagnostics for the failures that spoil a truth table, and
quantile helpers that are explicitly *not* a calibration.

Examples
--------
>>> from setqca.calibration import calibrate, direct_spec
>>> spec = direct_spec("innovation", full_out=20, crossover=50, full_in=80)
>>> result = calibrate([10, 50, 90], spec)  # doctest: +SKIP
>>> result.diagnostics.warnings  # doctest: +SKIP
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._diagnostics import (
    AnchorSuggestion,
    CalibrationDiagnostics,
    CalibrationResult,
    diagnose_calibration,
    diagnose_frame,
    suggest_anchors,
)
from ._direct import DirectCalibration, calibrate_crisp, calibrate_direct
from ._spec import (
    CalibrationMethod,
    CalibrationSpec,
    crisp_spec,
    direct_spec,
    indirect_spec,
)

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    import numpy.typing as npt

__all__ = [
    "AnchorSuggestion",
    "CalibrationDiagnostics",
    "CalibrationMethod",
    "CalibrationResult",
    "CalibrationSpec",
    "DirectCalibration",
    "calibrate",
    "calibrate_crisp",
    "calibrate_direct",
    "crisp_spec",
    "diagnose_calibration",
    "diagnose_frame",
    "direct_spec",
    "indirect_spec",
    "suggest_anchors",
]


def calibrate(values: npt.ArrayLike, spec: CalibrationSpec) -> CalibrationResult:
    """Apply a specification and diagnose the result in one step.

    Parameters
    ----------
    values : array_like
        Raw values, or calibrated ones for the identity method.
    spec : CalibrationSpec
        The calibration to apply.

    Returns
    -------
    CalibrationResult
        The calibrated values, the specification that produced them, and
        diagnostics. Keeping the three together is what makes a calibration
        reproducible rather than a number that appeared once.
    """
    calibrated = spec.apply(values)
    return CalibrationResult(
        spec=spec,
        values=calibrated,
        diagnostics=diagnose_calibration(calibrated, name=spec.condition),
    )
