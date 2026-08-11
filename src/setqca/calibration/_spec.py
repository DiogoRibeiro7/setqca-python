"""Reproducible, serialisable calibration specifications.

Calibration is the step where substantive judgement enters an analysis, so it
is the step most worth recording. A :class:`CalibrationSpec` is a value: it can
be stored beside the results, shipped in a replication package, compared
against a colleague's, and applied unchanged to further cases.

Four methods are supported:

``direct``
    Three-anchor logistic or piecewise transformation. The standard approach.
``crisp``
    Threshold cuts into binary or ordered categories.
``indirect``
    An explicit monotone mapping from raw values to memberships. Use this when
    theory dictates a shape the direct transformation cannot express.
``identity``
    The values are already calibrated and are passed through, validated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from setqca._validation import as_float_array, validate_membership

from ._direct import DirectCalibration, calibrate_crisp

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    import numpy.typing as npt

    from setqca._validation import FloatArray


class CalibrationMethod(Enum):
    """How raw values become set memberships."""

    DIRECT = "direct"
    CRISP = "crisp"
    INDIRECT = "indirect"
    IDENTITY = "identity"


@dataclass(frozen=True, slots=True)
class CalibrationSpec:
    """A complete, replayable description of one condition's calibration.

    Parameters
    ----------
    condition : str
        Name of the condition this calibrates.
    method : CalibrationMethod
        Which transformation to apply.
    anchors : tuple of float, optional
        ``(full_out, crossover, full_in)`` for the direct method.
    idm : float, default 0.95
        Membership at the inclusion anchor, for the direct logistic form.
    logistic : bool, default True
        Use the logistic rather than piecewise direct transformation.
    below, above : float, default 1.0
        Piecewise shaping exponents.
    thresholds : tuple of float, optional
        Cut points for the crisp method.
    mapping : tuple of (float, float), optional
        ``(raw, membership)`` points for the indirect method. Interpolated
        linearly between points and held flat outside them.
    note : str, optional
        Free text recording *why* these choices were made. Carried through
        serialisation, because the reason is part of the specification.
    """

    condition: str
    method: CalibrationMethod = CalibrationMethod.DIRECT
    anchors: tuple[float, float, float] | None = None
    idm: float = 0.95
    logistic: bool = True
    below: float = 1.0
    above: float = 1.0
    thresholds: tuple[float, ...] | None = None
    mapping: tuple[tuple[float, float], ...] | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.method is CalibrationMethod.DIRECT:
            if self.anchors is None:
                raise ValueError("The direct method requires anchors.")
            # Validate eagerly so a bad spec fails when it is written, not when
            # it is applied to data much later.
            DirectCalibration(
                full_out=self.anchors[0],
                crossover=self.anchors[1],
                full_in=self.anchors[2],
                idm=self.idm,
                logistic=self.logistic,
                below=self.below,
                above=self.above,
            )
        elif self.method is CalibrationMethod.CRISP:
            if not self.thresholds:
                raise ValueError("The crisp method requires thresholds.")
        elif self.method is CalibrationMethod.INDIRECT:
            if self.mapping is None or len(self.mapping) < 2:
                raise ValueError("The indirect method requires at least two mapping points.")
            raws = [raw for raw, _ in self.mapping]
            memberships = [membership for _, membership in self.mapping]
            if any(np.diff(raws) <= 0):
                raise ValueError(
                    "Indirect mapping points must have strictly increasing raw values."
                )
            if any(np.diff(memberships) < 0):
                raise ValueError("Indirect mapping must be non-decreasing in membership.")
            if not all(0.0 <= membership <= 1.0 for membership in memberships):
                raise ValueError("Indirect mapping memberships must be in [0, 1].")

    def apply(self, values: npt.ArrayLike) -> FloatArray:
        """Calibrate raw values according to this specification."""
        if self.method is CalibrationMethod.DIRECT:
            assert self.anchors is not None
            return DirectCalibration(
                full_out=self.anchors[0],
                crossover=self.anchors[1],
                full_in=self.anchors[2],
                idm=self.idm,
                logistic=self.logistic,
                below=self.below,
                above=self.above,
            ).transform(values)
        if self.method is CalibrationMethod.CRISP:
            assert self.thresholds is not None
            categories = calibrate_crisp(values, self.thresholds)
            # A single threshold yields a binary set; several yield ordered
            # categories, rescaled onto [0, 1] so downstream code sees
            # memberships rather than raw category indices.
            top = len(self.thresholds)
            return (categories / top).astype(np.float64)
        if self.method is CalibrationMethod.INDIRECT:
            assert self.mapping is not None
            raw = as_float_array(values, name=self.condition)
            points = np.asarray([point[0] for point in self.mapping], dtype=np.float64)
            targets = np.asarray([point[1] for point in self.mapping], dtype=np.float64)
            interpolated: FloatArray = np.interp(raw, points, targets).astype(np.float64)
            return interpolated
        return validate_membership(values, name=self.condition)

    def with_anchors(self, anchors: tuple[float, float, float]) -> CalibrationSpec:
        """Return a copy with different anchors, for sensitivity work."""
        return replace(self, anchors=anchors)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for JSON or YAML."""
        payload: dict[str, Any] = {
            "condition": self.condition,
            "method": self.method.value,
            "note": self.note,
        }
        if self.anchors is not None:
            payload["anchors"] = list(self.anchors)
            payload["idm"] = self.idm
            payload["logistic"] = self.logistic
            payload["below"] = self.below
            payload["above"] = self.above
        if self.thresholds is not None:
            payload["thresholds"] = list(self.thresholds)
        if self.mapping is not None:
            payload["mapping"] = [list(point) for point in self.mapping]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationSpec:
        """Rebuild a specification from :meth:`to_dict` output.

        Raises
        ------
        KeyError
            If the payload lacks a condition or method.
        ValueError
            If the method is unknown, or the specification is invalid.
        """
        anchors = payload.get("anchors")
        thresholds = payload.get("thresholds")
        mapping = payload.get("mapping")
        try:
            method = CalibrationMethod(payload["method"])
        except ValueError as error:
            raise ValueError(f"Unknown calibration method {payload['method']!r}.") from error
        return cls(
            condition=payload["condition"],
            method=method,
            anchors=None if anchors is None else (anchors[0], anchors[1], anchors[2]),
            idm=payload.get("idm", 0.95),
            logistic=payload.get("logistic", True),
            below=payload.get("below", 1.0),
            above=payload.get("above", 1.0),
            thresholds=None if thresholds is None else tuple(thresholds),
            mapping=None if mapping is None else tuple((p[0], p[1]) for p in mapping),
            note=payload.get("note", ""),
        )

    def to_json(self, *, indent: int | None = None, sort_keys: bool = False) -> str:
        """Serialise to a JSON string.

        Parameters
        ----------
        indent : int, optional
            Passed to :func:`json.dumps` for readable output.
        sort_keys : bool, default False
            Sort keys, which makes stored specifications diff cleanly.
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)

    @classmethod
    def from_json(cls, text: str) -> CalibrationSpec:
        """Rebuild a specification from JSON."""
        return cls.from_dict(json.loads(text))


def direct_spec(
    condition: str,
    *,
    full_out: float,
    crossover: float,
    full_in: float,
    idm: float = 0.95,
    logistic: bool = True,
    below: float = 1.0,
    above: float = 1.0,
    note: str = "",
) -> CalibrationSpec:
    """Build a three-anchor direct calibration specification."""
    return CalibrationSpec(
        condition=condition,
        method=CalibrationMethod.DIRECT,
        anchors=(full_out, crossover, full_in),
        idm=idm,
        logistic=logistic,
        below=below,
        above=above,
        note=note,
    )


def crisp_spec(condition: str, *, thresholds: tuple[float, ...], note: str = "") -> CalibrationSpec:
    """Build a crisp threshold specification."""
    return CalibrationSpec(
        condition=condition,
        method=CalibrationMethod.CRISP,
        thresholds=thresholds,
        note=note,
    )


def indirect_spec(
    condition: str, *, mapping: tuple[tuple[float, float], ...], note: str = ""
) -> CalibrationSpec:
    """Build an explicit monotone mapping specification.

    Use this when theory dictates a shape the direct transformation cannot
    express — a plateau, a step, an asymmetric ramp. Points are interpolated
    linearly and held flat beyond the ends.
    """
    return CalibrationSpec(
        condition=condition,
        method=CalibrationMethod.INDIRECT,
        mapping=mapping,
        note=note,
    )
