"""Structured QCA result objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._validation import FloatArray
from .metrics import SufficiencyFit, sufficiency

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .counterfactuals import CounterfactualAnalysis
    from .minimize.qmc import BooleanSolution
    from .truth_table import TruthTable

SolutionKind = str
"""Name of a solution family: ``"conservative"``, ``"parsimonious"`` or ``"intermediate"``."""

_SOLUTION_KINDS = ("conservative", "parsimonious", "intermediate")


@dataclass(frozen=True, slots=True)
class FittedSolution:
    """A Boolean solution together with its fuzzy empirical fit."""

    boolean: BooleanSolution
    fit: SufficiencyFit
    term_fits: tuple[SufficiencyFit, ...]

    def expression(self, conditions: tuple[str, ...]) -> str:
        """Render the solution in standard QCA notation, e.g. ``A*~B + C``."""
        return self.boolean.as_expression(conditions)


@dataclass(frozen=True, slots=True)
class QCAResult:
    """Complete fitted QCA result for one outcome."""

    method: str
    outcome: str
    conditions: tuple[str, ...]
    truth_table: TruthTable
    conservative: tuple[FittedSolution, ...]
    parsimonious: tuple[FittedSolution, ...]
    intermediate: tuple[FittedSolution, ...] | None
    intermediate_experimental: bool
    counterfactuals: CounterfactualAnalysis | None = None

    def solutions(self, kind: SolutionKind) -> tuple[FittedSolution, ...]:
        """Return the fitted solutions of one family.

        Parameters
        ----------
        kind : str
            One of ``"conservative"``, ``"parsimonious"`` or ``"intermediate"``.

        Returns
        -------
        tuple of FittedSolution
            Empty when the requested family was not computed.

        Raises
        ------
        ValueError
            If ``kind`` is not a recognised solution family.
        """
        if kind not in _SOLUTION_KINDS:
            raise ValueError(f"Unknown solution kind {kind!r}; expected one of {_SOLUTION_KINDS}.")
        values: tuple[FittedSolution, ...] | None = getattr(self, kind)
        return () if values is None else values

    def summary_frame(self, solution: SolutionKind = "conservative") -> pd.DataFrame:
        """Return one row per minimal solution of the requested family.

        Parameters
        ----------
        solution : str, default "conservative"
            Solution family to summarise.

        Returns
        -------
        pandas.DataFrame
            Columns ``solution``, ``consistency``, ``coverage``, ``PRI``,
            ``n_implicants`` and ``n_literals``. Empty when the family was not
            computed.
        """
        values = self.solutions(solution)
        return pd.DataFrame(
            {
                "solution": [item.expression(self.conditions) for item in values],
                "consistency": [item.fit.consistency for item in values],
                "coverage": [item.fit.coverage for item in values],
                "PRI": [item.fit.pri for item in values],
                "n_implicants": [len(item.boolean.implicants) for item in values],
                "n_literals": [item.boolean.literal_count for item in values],
            }
        )

    def _format_family(self, title: str, values: tuple[FittedSolution, ...]) -> list[str]:
        lines = [title]
        lines.extend(
            f"  {solution.expression(self.conditions)} "
            f"[cons={solution.fit.consistency:.3f}, cov={solution.fit.coverage:.3f}, "
            f"PRI={solution.fit.pri:.3f}]"
            for solution in values
        )
        return lines

    def __str__(self) -> str:
        lines = [
            self.method,
            f"Outcome: {self.outcome}",
            f"Conditions: {', '.join(self.conditions)}",
            f"Positive truth-table rows: {len(self.truth_table.positive_minterms)}",
            "",
        ]
        lines.extend(self._format_family("Conservative solution(s):", self.conservative))
        lines.extend(self._format_family("Parsimonious solution(s):", self.parsimonious))
        if self.intermediate is not None:
            title = "Intermediate solution(s)"
            if self.intermediate_experimental:
                title += " [experimental]"
            lines.extend(self._format_family(f"{title}:", self.intermediate))
        if self.counterfactuals is not None:
            analysis = self.counterfactuals
            lines.append(
                f"Counterfactuals: {len(analysis.easy)} easy admitted, "
                f"{len(analysis.difficult)} difficult refused, "
                f"of {len(analysis.simplifying_assumptions)} simplifying assumptions"
            )
        return "\n".join(lines)


def _oriented_membership(data: pd.DataFrame, condition: str, bit: int) -> FloatArray:
    """Return the condition membership, negated when the literal is absent."""
    values: FloatArray = data[condition].to_numpy(dtype=np.float64)
    return values if bit == 1 else 1.0 - values


def fit_boolean_solution(
    solution: BooleanSolution,
    *,
    data: pd.DataFrame,
    outcome: str,
    conditions: tuple[str, ...],
) -> FittedSolution:
    """Evaluate a Boolean solution as a fuzzy set over the original cases.

    Each prime implicant becomes a conjunction under the minimum t-norm and the
    solution as a whole becomes their disjunction under the maximum s-norm.

    Parameters
    ----------
    solution : BooleanSolution
        Minimal cover produced by the Boolean minimiser.
    data : pandas.DataFrame
        Calibrated case-level data.
    outcome : str
        Name of the outcome column.
    conditions : tuple of str
        Condition names in minterm order.

    Returns
    -------
    FittedSolution
        The solution with overall and term-level parameters of fit.
    """
    y = data[outcome].to_numpy(dtype=np.float64)
    n_cases = len(data)
    term_memberships: list[FloatArray] = []
    term_fits: list[SufficiencyFit] = []
    for implicant in solution.implicants:
        components: list[FloatArray] = [
            _oriented_membership(data, condition, bit)
            for condition, bit in zip(conditions, implicant.pattern, strict=True)
            if bit is not None
        ]
        # An implicant with no fixed literal is the tautology covering every case.
        membership = (
            np.ones(n_cases, dtype=np.float64) if not components else np.minimum.reduce(components)
        )
        term_memberships.append(membership)
        term_fits.append(sufficiency(membership, y))
    overall = (
        np.zeros(n_cases, dtype=np.float64)
        if not term_memberships
        else np.maximum.reduce(term_memberships)
    )
    return FittedSolution(solution, sufficiency(overall, y), tuple(term_fits))
