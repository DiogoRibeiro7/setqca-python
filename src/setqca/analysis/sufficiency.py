"""Case-level diagnostics for a sufficiency solution.

Parameters of fit summarise a solution in a few numbers. They do not say which
cases produced those numbers, and that is usually the question a researcher
actually has: which cases support this path, which contradict it, and which
outcomes does it fail to explain.

Case typology
-------------

For a term ``X`` and outcome ``Y``, with the crossover at 0.5
(Schneider and Rohlfing 2013):

- ``X > 0.5``, ``Y > 0.5``, ``X <= Y``: **typical**, supporting the claim.
- ``X > 0.5``, ``Y > 0.5``, ``X > Y``: **deviant for consistency in degree**,
  the right corner at the wrong magnitude.
- ``X > 0.5``, ``Y <= 0.5``: **deviant for consistency in kind**. The term holds
  and the outcome does not; this is the case-level contradiction.
- ``X <= 0.5``, ``Y > 0.5``: **deviant for coverage**, an outcome this term
  does not explain.
- ``X <= 0.5``, ``Y <= 0.5``: **individually irrelevant**, outside both sets.

Unique coverage
---------------

Raw coverage counts outcome membership a term accounts for. Unique coverage
counts only what *no other term* accounts for::

    covU_i = [ sum(min(Xi, Y)) - sum(min(Xi, max_over_others(Xj), Y)) ] / sum(Y)

A term with substantial raw coverage but near-zero unique coverage is
redundant in practice: drop it and the solution still explains the same cases.

References
----------
Schneider, C. Q. and Rohlfing, I. (2013). Combining QCA and process tracing in
set-theoretic multi-method research. *Sociological Methods & Research* 42(4),
559-597.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from setqca._validation import validate_columns, validate_membership
from setqca.expressions import evaluate_expression, parse_set_expression
from setqca.metrics import SufficiencyFit, sufficiency

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from setqca._validation import FloatArray
    from setqca.sets import SetExpression

CROSSOVER = 0.5


class CaseRole(Enum):
    """Where a case sits relative to one sufficiency claim."""

    TYPICAL = "typical"
    DEVIANT_CONSISTENCY_IN_DEGREE = "deviant consistency (degree)"
    DEVIANT_CONSISTENCY_IN_KIND = "deviant consistency (kind)"
    DEVIANT_COVERAGE = "deviant coverage"
    INDIVIDUALLY_IRRELEVANT = "individually irrelevant"

    @property
    def contradicts_sufficiency(self) -> bool:
        """Return whether this role counts against the sufficiency claim."""
        return self in {
            CaseRole.DEVIANT_CONSISTENCY_IN_DEGREE,
            CaseRole.DEVIANT_CONSISTENCY_IN_KIND,
        }


def classify_case(term_membership: float, outcome_membership: float) -> CaseRole:
    """Classify one case against one sufficiency claim.

    Parameters
    ----------
    term_membership : float
        Membership of the case in the term.
    outcome_membership : float
        Membership of the case in the outcome.

    Returns
    -------
    CaseRole
        The case's role, per the typology in the module docstring.
    """
    in_term = term_membership > CROSSOVER
    in_outcome = outcome_membership > CROSSOVER

    if in_term and in_outcome:
        if term_membership <= outcome_membership:
            return CaseRole.TYPICAL
        return CaseRole.DEVIANT_CONSISTENCY_IN_DEGREE
    if in_term:
        return CaseRole.DEVIANT_CONSISTENCY_IN_KIND
    if in_outcome:
        return CaseRole.DEVIANT_COVERAGE
    return CaseRole.INDIVIDUALLY_IRRELEVANT


@dataclass(frozen=True, slots=True)
class CaseDiagnostic:
    """One case, judged against one term."""

    case: str
    term_membership: float
    outcome_membership: float
    role: CaseRole
    uniquely_covered: bool


@dataclass(frozen=True, slots=True)
class TermDiagnostics:
    """One solution term, its fit, and every case's relation to it."""

    expression: str
    fit: SufficiencyFit
    unique_coverage: float
    frequency: int
    cases: tuple[CaseDiagnostic, ...]

    def by_role(self, role: CaseRole) -> tuple[str, ...]:
        """Return the labels of cases in one role."""
        return tuple(item.case for item in self.cases if item.role is role)

    @property
    def typical(self) -> tuple[str, ...]:
        """Return cases supporting the claim."""
        return self.by_role(CaseRole.TYPICAL)

    @property
    def deviant_consistency(self) -> tuple[str, ...]:
        """Return cases contradicting the claim, in kind or in degree."""
        return tuple(item.case for item in self.cases if item.role.contradicts_sufficiency)

    @property
    def contradictory(self) -> tuple[str, ...]:
        """Return cases where the term holds but the outcome does not."""
        return self.by_role(CaseRole.DEVIANT_CONSISTENCY_IN_KIND)

    @property
    def deviant_coverage(self) -> tuple[str, ...]:
        """Return outcome cases this term does not reach."""
        return self.by_role(CaseRole.DEVIANT_COVERAGE)

    @property
    def uniquely_covered(self) -> tuple[str, ...]:
        """Return cases this term covers that no other term does."""
        return tuple(item.case for item in self.cases if item.uniquely_covered)

    @property
    def redundant(self) -> bool:
        """Return whether the term adds no coverage another term does not already give."""
        return self.unique_coverage <= 0.0


@dataclass(frozen=True, slots=True)
class SolutionDiagnostics:
    """Diagnostics for a whole disjunctive solution."""

    outcome: str
    terms: tuple[TermDiagnostics, ...]
    fit: SufficiencyFit

    @property
    def redundant_terms(self) -> tuple[TermDiagnostics, ...]:
        """Return terms contributing no unique coverage."""
        return tuple(term for term in self.terms if term.redundant)

    def to_frame(self) -> pd.DataFrame:
        """Return one row per term, with fit and case counts.

        Returns
        -------
        pandas.DataFrame
            Columns ``term``, ``consistency``, ``PRI``, ``raw_coverage``,
            ``unique_coverage``, ``n``, and one count per case role.
        """
        return pd.DataFrame(
            {
                "term": [term.expression for term in self.terms],
                "consistency": [term.fit.consistency for term in self.terms],
                "PRI": [term.fit.pri for term in self.terms],
                "raw_coverage": [term.fit.coverage for term in self.terms],
                "unique_coverage": [term.unique_coverage for term in self.terms],
                "n": [term.frequency for term in self.terms],
                **{
                    role.value: [len(term.by_role(role)) for term in self.terms]
                    for role in CaseRole
                },
            }
        )

    def cases_frame(self) -> pd.DataFrame:
        """Return one row per case per term, for case-oriented work.

        Returns
        -------
        pandas.DataFrame
            Columns ``term``, ``case``, ``term_membership``,
            ``outcome_membership``, ``role`` and ``uniquely_covered``.
        """
        records = [
            {
                "term": term.expression,
                "case": item.case,
                "term_membership": item.term_membership,
                "outcome_membership": item.outcome_membership,
                "role": item.role.value,
                "uniquely_covered": item.uniquely_covered,
            }
            for term in self.terms
            for item in term.cases
        ]
        return pd.DataFrame.from_records(
            records,
            columns=[
                "term",
                "case",
                "term_membership",
                "outcome_membership",
                "role",
                "uniquely_covered",
            ],
        )

    def __str__(self) -> str:
        lines = [f"Sufficiency diagnostics for {self.outcome}", ""]
        for term in self.terms:
            lines.append(
                f"{term.expression} "
                f"[cons={term.fit.consistency:.3f}, cov={term.fit.coverage:.3f}, "
                f"uniq={term.unique_coverage:.3f}, n={term.frequency}]"
            )
            if term.typical:
                lines.append(f"  typical: {', '.join(term.typical)}")
            if term.contradictory:
                lines.append(f"  contradictory: {', '.join(term.contradictory)}")
            if term.deviant_coverage:
                lines.append(f"  unexplained outcomes: {', '.join(term.deviant_coverage)}")
            if term.redundant:
                lines.append("  redundant: adds no coverage beyond the other terms")
        return "\n".join(lines)


def _memberships(
    data: pd.DataFrame, terms: Sequence[str | SetExpression]
) -> list[tuple[str, FloatArray]]:
    resolved: list[tuple[str, FloatArray]] = []
    for term in terms:
        node = parse_set_expression(term) if isinstance(term, str) else term
        resolved.append((str(node), evaluate_expression(node, data)))
    return resolved


def _unique_coverage(
    membership: FloatArray, others: list[FloatArray], outcome: FloatArray
) -> float:
    """Return coverage of the outcome that no other term accounts for."""
    total = float(outcome.sum())
    if total == 0.0:
        return 0.0
    own = float(np.minimum(membership, outcome).sum())
    if not others:
        return own / total
    overlap_membership = np.maximum.reduce(others)
    shared = float(np.minimum(np.minimum(membership, overlap_membership), outcome).sum())
    return (own - shared) / total


def sufficiency_diagnostics(
    data: pd.DataFrame,
    *,
    outcome: str,
    terms: Sequence[str | SetExpression],
    case_id: str | None = None,
) -> SolutionDiagnostics:
    """Diagnose a disjunctive sufficiency solution case by case.

    Parameters
    ----------
    data : pandas.DataFrame
        Calibrated memberships in ``[0, 1]``.
    outcome : str
        Name of the outcome column.
    terms : sequence of str or SetExpression
        The solution's terms. Strings are parsed, so
        ``["DEV*URB", "LIT*~IND"]`` works directly.
    case_id : str, optional
        Column holding case labels. Defaults to the frame index, so no
        particular schema is assumed.

    Returns
    -------
    SolutionDiagnostics
        Per-term fit including unique coverage, and every case's role.

    Raises
    ------
    ValueError
        If no terms are given, or the data are not calibrated.
    KeyError
        If a named column is absent.

    Examples
    --------
    >>> diagnostics = sufficiency_diagnostics(  # doctest: +SKIP
    ...     data, outcome="SURV", terms=["DEV*URB*LIT*IND*STB"]
    ... )
    >>> diagnostics.terms[0].typical  # doctest: +SKIP
    ('BE', 'CZ', 'NL', 'UK')
    """
    if not terms:
        raise ValueError("At least one term is required.")
    validate_columns(data, [outcome])
    y = validate_membership(data[outcome].to_numpy(), name=outcome)

    if case_id is None:
        labels = [str(index) for index in data.index]
    else:
        validate_columns(data, [case_id])
        labels = [str(value) for value in data[case_id]]

    resolved = _memberships(data, terms)
    all_memberships = [membership for _, membership in resolved]

    diagnostics: list[TermDiagnostics] = []
    for position, (expression, membership) in enumerate(resolved):
        others = [other for index, other in enumerate(all_memberships) if index != position]
        covered_elsewhere = np.maximum.reduce(others) if others else np.zeros_like(membership)
        cases = tuple(
            CaseDiagnostic(
                case=label,
                term_membership=float(term_value),
                outcome_membership=float(outcome_value),
                role=classify_case(float(term_value), float(outcome_value)),
                uniquely_covered=bool(term_value > CROSSOVER and other_value <= CROSSOVER),
            )
            for label, term_value, outcome_value, other_value in zip(
                labels, membership, y, covered_elsewhere, strict=True
            )
        )
        diagnostics.append(
            TermDiagnostics(
                expression=expression,
                fit=sufficiency(membership, y),
                unique_coverage=_unique_coverage(membership, others, y),
                frequency=int(np.sum(membership > CROSSOVER)),
                cases=cases,
            )
        )

    overall = np.maximum.reduce(all_memberships)
    return SolutionDiagnostics(
        outcome=outcome,
        terms=tuple(diagnostics),
        fit=sufficiency(overall, y),
    )
