"""Warning before combinatorial explosion, rather than after.

Exact Boolean minimisation is worst-case exponential. That cost is the price of
an exact answer and is not negotiable here — no heuristic is substituted when a
problem gets hard, because a solution that is silently approximate is worse than
one that takes a long time.

What *is* negotiable is being told in advance. This module estimates how hard a
chart looks before the exponential phase begins, so a run that will not finish
says so rather than appearing to hang.

Where the thresholds come from
------------------------------

Measured with ``benchmarks/profile_phases.py`` on tables of seven conditions,
timing the exact cover phase alone:

===========  ==========  ===============
Primes       Positives   Cover solving
===========  ==========  ===============
10           11          <0.0001 s
26           33          0.0001 s
53           65          0.0024 s
78           91          0.60 s
===========  ==========  ===============

The count of prime implicants is the useful predictor: the cost climbs sharply
somewhere past sixty, and the climb is driven by how much the primes overlap
rather than by the number of conditions on its own. The thresholds below sit
either side of that transition. They are a warning, not a prediction — a chart
with many primes and little overlap solves instantly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

MODERATE_PRIMES = 32
HIGH_PRIMES = 64

Level = Literal["low", "moderate", "high"]


class MinimizationComplexityWarning(UserWarning):
    """Raised when a minimisation problem looks likely to be slow.

    The computation still runs, and still returns an exact answer. Silence it
    with :func:`warnings.simplefilter`, or turn the check off entirely with
    ``complexity_guard=False``.
    """


@dataclass(frozen=True, slots=True)
class ComplexityEstimate:
    """How hard a prime-implicant chart looks before it is solved."""

    width: int
    required: int
    dont_cares: int
    primes: int

    @property
    def universe(self) -> int:
        """Return the size of the property space."""
        size: int = 2**self.width
        return size

    @property
    def chart_cells(self) -> int:
        """Return the size of the chart, primes times rows to cover."""
        return self.primes * self.required

    @property
    def level(self) -> Level:
        """Return a coarse difficulty band."""
        if self.primes > HIGH_PRIMES:
            return "high"
        if self.primes > MODERATE_PRIMES:
            return "moderate"
        return "low"

    @property
    def should_warn(self) -> bool:
        """Return whether this problem warrants warning the caller."""
        return self.level == "high"

    @property
    def message(self) -> str:
        """Return a description of the problem's shape and what to do about it."""
        return (
            f"Exact minimisation of {self.required} configurations over "
            f"{self.width} conditions produced {self.primes} prime implicants "
            f"({self.chart_cells} chart cells). Solving the chart exactly is "
            "worst-case exponential and may take a long time. The result will "
            "still be exact. To reduce the cost, use fewer conditions, tighten "
            "the consistency cutoff so fewer configurations qualify, or lower "
            "max_solutions if the model is highly ambiguous."
        )

    def __str__(self) -> str:
        return (
            f"width={self.width}, required={self.required}, "
            f"dont_cares={self.dont_cares}, primes={self.primes}, "
            f"level={self.level}"
        )


def estimate_complexity(
    *, width: int, required: int, dont_cares: int, primes: int
) -> ComplexityEstimate:
    """Describe how hard a chart looks, from counts alone.

    Parameters
    ----------
    width : int
        Number of conditions.
    required : int
        Configurations that must be covered.
    dont_cares : int
        Configurations usable but not required.
    primes : int
        Number of prime implicants generated.

    Returns
    -------
    ComplexityEstimate
        The counts, a difficulty band, and an explanatory message.
    """
    return ComplexityEstimate(width=width, required=required, dont_cares=dont_cares, primes=primes)


def warn_if_complex(estimate: ComplexityEstimate, *, stacklevel: int = 3) -> bool:
    """Emit :class:`MinimizationComplexityWarning` when the chart looks hard.

    Parameters
    ----------
    estimate : ComplexityEstimate
        The estimate to judge.
    stacklevel : int, default 3
        Passed through to :func:`warnings.warn` so the warning points at the
        caller's code rather than at this module.

    Returns
    -------
    bool
        Whether a warning was emitted.
    """
    if not estimate.should_warn:
        return False
    warnings.warn(estimate.message, MinimizationComplexityWarning, stacklevel=stacklevel)
    return True
