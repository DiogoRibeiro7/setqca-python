"""Calibration of raw variables into crisp and fuzzy sets."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np
import numpy.typing as npt

from ._validation import FloatArray, as_float_array


def _logistic_cdf(z: FloatArray) -> FloatArray:
    """Return ``1 / (1 + exp(-z))`` without overflowing on extreme inputs.

    The naive expression overflows once ``|z|`` exceeds roughly 709. Evaluating
    each tail with the branch whose exponential decays keeps the result exact
    to within floating-point precision across the whole real line.
    """
    out = np.empty_like(z)
    positive = z >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    tail = np.exp(z[~positive])
    out[~positive] = tail / (1.0 + tail)
    return out


@dataclass(frozen=True, slots=True)
class DirectCalibration:
    """Three-anchor direct fuzzy-set calibration specification.

    The logistic implementation follows the standard three-anchor direct
    calibration parameterisation: the crossover maps to 0.5 and the inclusion
    and exclusion anchors map to ``idm`` and ``1-idm`` respectively.
    """

    full_out: float
    crossover: float
    full_in: float
    idm: float = 0.95
    logistic: bool = True
    below: float = 1.0
    above: float = 1.0

    def __post_init__(self) -> None:
        increasing = self.full_out < self.crossover < self.full_in
        decreasing = self.full_in < self.crossover < self.full_out
        if not (increasing or decreasing):
            raise ValueError("Anchors must be strictly ordered around the crossover.")
        if not 0.5 < self.idm < 1.0:
            raise ValueError("idm must be strictly between 0.5 and 1.")
        if self.below <= 0 or self.above <= 0:
            raise ValueError("below and above exponents must be positive.")

    def transform(self, values: npt.ArrayLike) -> FloatArray:
        """Calibrate raw values into fuzzy membership scores.

        Parameters
        ----------
        values : array_like
            Raw numeric values on the original measurement scale.

        Returns
        -------
        FloatArray
            Membership scores in ``[0, 1]``.
        """
        x = as_float_array(values, name="values")
        if self.logistic:
            return self._logistic(x)
        return self._piecewise(x)

    def _logistic(self, x: FloatArray) -> FloatArray:
        # Mirror decreasing calibration around increasing calibration. This is
        # algebraically equivalent to R QCA's direct logistic implementation.
        decreasing = self.full_out > self.full_in
        exclusion = min(self.full_out, self.full_in)
        inclusion = max(self.full_out, self.full_in)
        crossover = self.crossover
        log_odds = log(self.idm / (1.0 - self.idm))

        below_mask = x < crossover
        scale = np.where(below_mask, exclusion - crossover, inclusion - crossover)
        sign = np.where(below_mask, 1.0, -1.0)
        exponent = sign * (x - crossover) * log_odds / scale
        membership = _logistic_cdf(-exponent)
        if decreasing:
            membership = 1.0 - membership
        return membership.astype(np.float64)

    def _piecewise(self, x: FloatArray) -> FloatArray:
        decreasing = self.full_out > self.full_in
        if decreasing:
            inverse = DirectCalibration(
                full_out=self.full_in,
                crossover=self.crossover,
                full_in=self.full_out,
                idm=self.idm,
                logistic=False,
                below=self.below,
                above=self.above,
            )
            return 1.0 - inverse.transform(x)

        result = np.empty_like(x)
        low = x <= self.full_out
        low_mid = (x > self.full_out) & (x <= self.crossover)
        high_mid = (x > self.crossover) & (x <= self.full_in)
        high = x > self.full_in

        result[low] = 0.0
        result[low_mid] = (
            ((self.full_out - x[low_mid]) / (self.full_out - self.crossover)) ** self.below
        ) / 2.0
        result[high_mid] = (
            1.0
            - (((self.full_in - x[high_mid]) / (self.full_in - self.crossover)) ** self.above) / 2.0
        )
        result[high] = 1.0
        return result


def calibrate_direct(
    values: npt.ArrayLike,
    *,
    full_out: float,
    crossover: float,
    full_in: float,
    idm: float = 0.95,
    logistic: bool = True,
    below: float = 1.0,
    above: float = 1.0,
) -> FloatArray:
    """Calibrate raw values with three-anchor direct fuzzy calibration.

    Convenience wrapper around :class:`DirectCalibration` for one-shot use.

    Parameters
    ----------
    values : array_like
        Raw numeric values to calibrate.
    full_out, crossover, full_in : float
        The three qualitative anchors. ``full_out < crossover < full_in``
        defines an increasing set; the reverse order defines a decreasing set.
    idm : float, default 0.95
        Membership assigned to the inclusion anchor. Must lie in ``(0.5, 1)``.
    logistic : bool, default True
        Use the logistic transformation. When ``False``, a piecewise
        linear/power transformation with exact endpoints is used instead.
    below, above : float, default 1.0
        Positive exponents shaping the piecewise transformation on either side
        of the crossover. Ignored when ``logistic`` is ``True``.

    Returns
    -------
    FloatArray
        Fuzzy membership scores in ``[0, 1]``.

    Examples
    --------
    >>> calibrate_direct([10, 20, 30], full_out=10, crossover=20, full_in=30).round(2)
    array([0.05, 0.5 , 0.95])
    """
    return DirectCalibration(
        full_out=full_out,
        crossover=crossover,
        full_in=full_in,
        idm=idm,
        logistic=logistic,
        below=below,
        above=above,
    ).transform(values)


def calibrate_crisp(values: npt.ArrayLike, thresholds: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Calibrate a raw numeric variable into ordered crisp categories.

    With one threshold this produces a binary crisp set. Multiple thresholds
    produce integer categories ``0..k`` and form a future mvQCA-compatible API.
    """
    x = as_float_array(values, name="values")
    cuts = as_float_array(thresholds, name="thresholds")
    if np.any(np.diff(np.sort(cuts)) <= 0):
        raise ValueError("thresholds must be unique.")
    cuts = np.sort(cuts)
    return np.searchsorted(cuts, x, side="right").astype(np.int64)
