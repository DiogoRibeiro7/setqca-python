"""Set-theoretic parameters of fit for QCA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._validation import validate_membership


@dataclass(frozen=True, slots=True)
class SufficiencyFit:
    """Parameters of fit for a sufficiency relation X <= Y."""

    consistency: float
    coverage: float
    pri: float


@dataclass(frozen=True, slots=True)
class NecessityFit:
    """Parameters of fit for a necessity relation Y <= X."""

    consistency: float
    coverage: float
    ron: float


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def sufficiency(cause: npt.ArrayLike, outcome: npt.ArrayLike) -> SufficiencyFit:
    """Calculate fuzzy-set sufficiency consistency, coverage and PRI.

    PRI follows the implementation used by the R ``QCA`` package: inconsistency
    that simultaneously supports the outcome and its negation is removed from
    numerator and denominator.
    """
    x = validate_membership(cause, name="cause")
    y = validate_membership(outcome, name="outcome")
    if x.shape != y.shape:
        raise ValueError("cause and outcome must have equal length.")

    xy = np.minimum(x, y)
    sum_xy = float(xy.sum())
    sum_x = float(x.sum())
    sum_y = float(y.sum())
    contradictory = float(np.minimum(xy, 1.0 - y).sum())

    return SufficiencyFit(
        consistency=_safe_ratio(sum_xy, sum_x),
        coverage=_safe_ratio(sum_xy, sum_y),
        pri=_safe_ratio(sum_xy - contradictory, sum_x - contradictory),
    )


def necessity(cause: npt.ArrayLike, outcome: npt.ArrayLike) -> NecessityFit:
    """Calculate fuzzy-set necessity consistency, coverage and RoN."""
    x = validate_membership(cause, name="cause")
    y = validate_membership(outcome, name="outcome")
    if x.shape != y.shape:
        raise ValueError("cause and outcome must have equal length.")

    xy = np.minimum(x, y)
    sum_xy = float(xy.sum())
    sum_x = float(x.sum())
    sum_y = float(y.sum())
    ron_num = float((1.0 - x).sum())
    ron_den = float((1.0 - np.minimum(x, y)).sum())

    return NecessityFit(
        consistency=_safe_ratio(sum_xy, sum_y),
        coverage=_safe_ratio(sum_xy, sum_x),
        ron=_safe_ratio(ron_num, ron_den),
    )
