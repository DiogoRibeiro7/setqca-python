"""Systematic analysis of necessary conditions.

A condition is **necessary** for an outcome when the outcome is a subset of the
condition: wherever the outcome is present, the condition is present too. This
is the mirror image of sufficiency, and the two answer different questions —
necessity asks what cannot be missing, sufficiency asks what is enough.

Necessity is easy to claim and easy to overclaim. A condition present in almost
every case is a superset of almost anything, so it will show near-perfect
necessity consistency while explaining nothing. That is **trivial necessity**,
and it is reported here rather than left for the reader to notice.

Which compounds are worth testing
---------------------------------

Only disjunctions. For the minimum/maximum operators:

- ``consistency(A*B) <= min(consistency(A), consistency(B))``, because
  ``min(A, B, Y) <= min(A, Y)``. A conjunction can therefore never be more
  necessary than its own components, and testing conjunctions adds nothing.
- ``consistency(A+B) >= max(consistency(A), consistency(B))``, because
  ``min(max(A, B), Y) >= min(A, Y)``. A union *can* be necessary when neither
  part is, which is the SUIN condition of the literature — a **s**ufficient part
  of an **i**nsufficient but **n**ecessary condition.

So conjunctions are excluded on mathematical grounds, not for lack of effort.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from setqca._validation import validate_columns, validate_membership
from setqca.metrics import NecessityFit, necessity

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from setqca._validation import FloatArray

DEFAULT_CONSISTENCY = 0.90
DEFAULT_RELEVANCE = 0.50


@dataclass(frozen=True, slots=True)
class NecessityCandidate:
    """One candidate necessary condition, with everything needed to judge it.

    Attributes
    ----------
    expression
        The candidate in standard QCA notation, e.g. ``"~DEV"`` or ``"DEV+URB"``.
    fit
        Consistency, coverage and relevance of necessity.
    prevalence
        Mean membership of the candidate across cases. A prevalence near 1 is
        what makes a necessity claim trivial.
    consistent
        Whether consistency reached the threshold.
    relevant
        Whether relevance of necessity reached its threshold.
    """

    expression: str
    fit: NecessityFit
    prevalence: float
    consistent: bool
    relevant: bool

    @property
    def necessary(self) -> bool:
        """Return whether the candidate is both consistent and non-trivial."""
        return self.consistent and self.relevant

    @property
    def trivial(self) -> bool:
        """Return whether the candidate is consistent but irrelevant.

        This is the dangerous combination: the numbers look like necessity, but
        the condition is so prevalent that the claim carries no information.
        """
        return self.consistent and not self.relevant


@dataclass(frozen=True, slots=True)
class NecessityAnalysis:
    """The result of screening candidates for necessity."""

    outcome: str
    consistency_threshold: float
    relevance_threshold: float
    candidates: tuple[NecessityCandidate, ...]

    @property
    def necessary(self) -> tuple[NecessityCandidate, ...]:
        """Return candidates that are consistent and non-trivial."""
        return tuple(item for item in self.candidates if item.necessary)

    @property
    def trivial(self) -> tuple[NecessityCandidate, ...]:
        """Return candidates that pass on consistency but fail on relevance."""
        return tuple(item for item in self.candidates if item.trivial)

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy table, sorted by consistency then relevance.

        Returns
        -------
        pandas.DataFrame
            Columns ``condition``, ``consistency``, ``coverage``, ``RoN``,
            ``prevalence``, ``necessary`` and ``trivial``.
        """
        frame = pd.DataFrame(
            {
                "condition": [item.expression for item in self.candidates],
                "consistency": [item.fit.consistency for item in self.candidates],
                "coverage": [item.fit.coverage for item in self.candidates],
                "RoN": [item.fit.ron for item in self.candidates],
                "prevalence": [item.prevalence for item in self.candidates],
                "necessary": [item.necessary for item in self.candidates],
                "trivial": [item.trivial for item in self.candidates],
            }
        )
        return frame.sort_values(
            ["consistency", "RoN"], ascending=False, kind="stable"
        ).reset_index(drop=True)

    def __str__(self) -> str:
        lines = [
            f"Necessity analysis for {self.outcome}",
            f"Thresholds: consistency >= {self.consistency_threshold}, "
            f"RoN >= {self.relevance_threshold}",
            "",
        ]
        if self.necessary:
            lines.append("Necessary:")
            lines.extend(
                f"  {item.expression} "
                f"[cons={item.fit.consistency:.3f}, cov={item.fit.coverage:.3f}, "
                f"RoN={item.fit.ron:.3f}]"
                for item in self.necessary
            )
        else:
            lines.append("Necessary: none")
        if self.trivial:
            lines.append("Consistent but trivial (prevalent enough to be uninformative):")
            lines.extend(
                f"  {item.expression} "
                f"[cons={item.fit.consistency:.3f}, RoN={item.fit.ron:.3f}, "
                f"prevalence={item.prevalence:.3f}]"
                for item in self.trivial
            )
        return "\n".join(lines)


def _candidate(
    expression: str,
    membership: FloatArray,
    outcome: FloatArray,
    consistency_threshold: float,
    relevance_threshold: float,
) -> NecessityCandidate:
    fit = necessity(membership, outcome)
    return NecessityCandidate(
        expression=expression,
        fit=fit,
        prevalence=float(np.mean(membership)),
        consistent=fit.consistency >= consistency_threshold,
        relevant=fit.ron >= relevance_threshold,
    )


def necessity_analysis(
    data: pd.DataFrame,
    *,
    outcome: str,
    conditions: list[str] | tuple[str, ...],
    consistency_threshold: float = DEFAULT_CONSISTENCY,
    relevance_threshold: float = DEFAULT_RELEVANCE,
    include_absence: bool = True,
    max_disjunction_size: int = 1,
) -> NecessityAnalysis:
    """Screen conditions, and optionally their disjunctions, for necessity.

    Parameters
    ----------
    data : pandas.DataFrame
        Calibrated memberships in ``[0, 1]``. Works for crisp and fuzzy alike,
        since crisp membership is the ``{0, 1}`` special case.
    outcome : str
        Name of the outcome column.
    conditions : list of str or tuple of str
        Condition columns to screen.
    consistency_threshold : float, default 0.90
        Minimum necessity consistency for a candidate to count as consistent.
    relevance_threshold : float, default 0.50
        Minimum relevance of necessity. Candidates that pass on consistency but
        fail here are reported as trivial rather than necessary.
    include_absence : bool, default True
        Also screen the negation of every condition. A condition's absence can
        be necessary when its presence is not.
    max_disjunction_size : int, default 1
        Largest union to test. ``1`` screens single conditions only; ``2`` adds
        every pair, and so on. Conjunctions are never tested — see the module
        docstring for why they cannot help.

    Returns
    -------
    NecessityAnalysis
        Every candidate screened, with those that are necessary and those that
        are merely trivial identified separately.

    Raises
    ------
    ValueError
        If a threshold is out of range, ``max_disjunction_size`` is below 1, or
        the data are not calibrated.
    KeyError
        If a named column is absent.

    Examples
    --------
    >>> analysis = necessity_analysis(  # doctest: +SKIP
    ...     data, outcome="SURV", conditions=["DEV", "URB", "LIT"]
    ... )
    >>> analysis.to_frame()  # doctest: +SKIP
    """
    if not 0.0 <= consistency_threshold <= 1.0:
        raise ValueError("consistency_threshold must be in [0, 1].")
    if not 0.0 <= relevance_threshold <= 1.0:
        raise ValueError("relevance_threshold must be in [0, 1].")
    if max_disjunction_size < 1:
        raise ValueError("max_disjunction_size must be at least 1.")

    names = validate_columns(data, conditions)
    if not names:
        raise ValueError("At least one condition is required.")
    validate_columns(data, [outcome])

    y = validate_membership(data[outcome].to_numpy(), name=outcome)

    # Literals first: each condition present, and optionally absent.
    literals: list[tuple[str, FloatArray]] = []
    for name in names:
        values = validate_membership(data[name].to_numpy(), name=name)
        literals.append((name, values))
        if include_absence:
            literals.append((f"~{name}", 1.0 - values))

    candidates = [
        _candidate(expression, membership, y, consistency_threshold, relevance_threshold)
        for expression, membership in literals
    ]

    # Unions of literals. A union can be necessary when no part of it is, which
    # is the only compound worth screening.
    for size in range(2, max_disjunction_size + 1):
        for chosen in combinations(literals, size):
            expression = "+".join(name for name, _ in chosen)
            membership = np.maximum.reduce([values for _, values in chosen])
            candidates.append(
                _candidate(expression, membership, y, consistency_threshold, relevance_threshold)
            )

    return NecessityAnalysis(
        outcome=outcome,
        consistency_threshold=consistency_threshold,
        relevance_threshold=relevance_threshold,
        candidates=tuple(candidates),
    )
