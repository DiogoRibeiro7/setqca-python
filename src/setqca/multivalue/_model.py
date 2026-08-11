"""Multi-value truth tables and the mvQCA estimator.

The public shape mirrors :mod:`setqca.models` so that moving between csQCA,
fsQCA and mvQCA is a change of estimator rather than a change of workflow.

Conditions are categorical and each case falls in exactly one configuration, so
configuration membership is crisp. The outcome may still be fuzzy: sufficiency
of a crisp configuration for a fuzzy outcome is well defined, and reduces to
the proportion of cases showing the outcome when the outcome is crisp too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from setqca._validation import validate_columns, validate_membership
from setqca.metrics import SufficiencyFit, sufficiency

from ._cube import MultiValueSolution, minimize_multivalue
from ._domain import MultiValueDomain

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from collections.abc import Mapping

MultiValueCode = Literal["1", "0", "C", "R"]


@dataclass(frozen=True, slots=True)
class MultiValueRow:
    """One configuration of the multi-value property space."""

    index: int
    configuration: tuple[int, ...]
    frequency: int
    consistency: float
    pri: float
    outcome: MultiValueCode
    cases: tuple[str, ...]
    exclusion_reason: str | None = None

    @property
    def observed(self) -> bool:
        """Return whether the configuration passed the frequency cutoff."""
        return self.outcome != "R"


@dataclass(frozen=True, slots=True)
class MultiValueTruthTable:
    """A complete multi-value truth table."""

    domain: MultiValueDomain
    outcome_name: str
    rows: tuple[MultiValueRow, ...]
    inclusion_cutoff: float
    frequency_cutoff: int

    @property
    def positive_indices(self) -> set[int]:
        """Return configurations coded sufficient."""
        return {row.index for row in self.rows if row.outcome == "1"}

    @property
    def remainder_indices(self) -> set[int]:
        """Return configurations with too few cases to judge."""
        return {row.index for row in self.rows if row.outcome == "R"}

    def rows_with(self, code: MultiValueCode) -> tuple[MultiValueRow, ...]:
        """Return rows carrying one outcome code."""
        return tuple(row for row in self.rows if row.outcome == code)

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy representation, one row per configuration."""
        records: list[dict[str, Any]] = []
        for row in self.rows:
            record: dict[str, Any] = dict(
                zip(self.domain.conditions, row.configuration, strict=True)
            )
            record.update(
                {
                    "index": row.index,
                    "n": row.frequency,
                    "consistency": row.consistency,
                    "PRI": row.pri,
                    "OUT": row.outcome,
                    "cases": ", ".join(row.cases),
                    "excluded_because": row.exclusion_reason or "",
                }
            )
            records.append(record)
        return pd.DataFrame.from_records(records)

    def minimize(
        self, *, include_remainders: bool = False, max_solutions: int = 256
    ) -> tuple[MultiValueSolution, ...]:
        """Minimise directly from the table.

        Raises
        ------
        ValueError
            If no configuration is coded sufficient.
        """
        on_set = self.positive_indices
        if not on_set:
            raise ValueError("No configuration is sufficient under the chosen thresholds.")
        return minimize_multivalue(
            on_set,
            domain=self.domain,
            dont_cares=self.remainder_indices if include_remainders else None,
            max_solutions=max_solutions,
        )


@dataclass(frozen=True, slots=True)
class MultiValueResult:
    """A fitted mvQCA analysis."""

    domain: MultiValueDomain
    outcome: str
    truth_table: MultiValueTruthTable
    conservative: tuple[MultiValueSolution, ...]
    parsimonious: tuple[MultiValueSolution, ...]
    fits: dict[str, SufficiencyFit] = field(default_factory=dict)

    def summary_frame(self, solution: str = "conservative") -> pd.DataFrame:
        """Return one row per minimal solution of a family."""
        if solution not in ("conservative", "parsimonious"):
            raise ValueError(
                f"Unknown solution kind {solution!r}; expected 'conservative' or 'parsimonious'."
            )
        solutions: tuple[MultiValueSolution, ...] = getattr(self, solution)
        expressions = [item.as_expression(self.domain) for item in solutions]
        return pd.DataFrame(
            {
                "solution": expressions,
                "n_cubes": [len(item.cubes) for item in solutions],
                "n_literals": [item.literal_count(self.domain) for item in solutions],
                "consistency": [
                    self.fits[expression].consistency if expression in self.fits else float("nan")
                    for expression in expressions
                ],
                "coverage": [
                    self.fits[expression].coverage if expression in self.fits else float("nan")
                    for expression in expressions
                ],
            }
        )

    def __str__(self) -> str:
        lines = [
            "Multi-value Qualitative Comparative Analysis",
            f"Outcome: {self.outcome}",
            f"Property space: {self.domain}",
            f"Sufficient configurations: {len(self.truth_table.positive_indices)}",
            "",
            "Conservative solution(s):",
        ]
        lines.extend(f"  {item.as_expression(self.domain)}" for item in self.conservative)
        lines.append("Parsimonious solution(s):")
        lines.extend(f"  {item.as_expression(self.domain)}" for item in self.parsimonious)
        return "\n".join(lines)


def _levels(data: pd.DataFrame, conditions: list[str]) -> tuple[int, ...]:
    counts: list[int] = []
    for name in conditions:
        values = data[name].to_numpy()
        if not np.all(np.equal(np.mod(values, 1), 0)):
            raise ValueError(
                f"Condition {name!r} must hold integer category codes; mvQCA "
                "conditions are categorical, not fuzzy."
            )
        integers = values.astype(np.int64)
        if integers.min() < 0:
            raise ValueError(f"Condition {name!r} has a negative category code.")
        counts.append(int(integers.max()) + 1)
    return tuple(counts)


def build_multivalue_truth_table(
    data: pd.DataFrame,
    *,
    outcome: str,
    conditions: list[str] | tuple[str, ...],
    levels: Mapping[str, int] | None = None,
    inclusion_cutoff: float = 0.8,
    frequency_cutoff: int = 1,
    case_id: str | None = None,
) -> MultiValueTruthTable:
    """Build a complete multi-value truth table.

    Parameters
    ----------
    data : pandas.DataFrame
        Condition columns holding integer category codes from ``0``, and an
        outcome column holding memberships in ``[0, 1]``.
    outcome : str
        Name of the outcome column.
    conditions : list of str or tuple of str
        Condition columns.
    levels : mapping of str to int, optional
        Number of categories per condition. Inferred from the data when
        omitted, which can understate a level that no case happens to take.
    inclusion_cutoff : float, default 0.8
        Minimum sufficiency consistency for a configuration to count.
    frequency_cutoff : int, default 1
        Minimum number of cases for a configuration to be observed.
    case_id : str, optional
        Column holding case labels. Defaults to the frame index.

    Returns
    -------
    MultiValueTruthTable
        One row per logically possible configuration.

    Raises
    ------
    ValueError
        If a condition is not categorical, a level count is too small, or a
        cutoff is out of range.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    names = validate_columns(data, conditions)
    if not names:
        raise ValueError("At least one condition is required.")
    validate_columns(data, [outcome])
    if not 0.0 <= inclusion_cutoff <= 1.0:
        raise ValueError("inclusion_cutoff must be in [0, 1].")
    if frequency_cutoff < 1:
        raise ValueError("frequency_cutoff must be at least 1.")

    counts = _levels(data, names)
    if levels is not None:
        missing = [name for name in names if name not in levels]
        if missing:
            raise KeyError(f"Levels missing for conditions: {missing}")
        declared = tuple(int(levels[name]) for name in names)
        for name, observed, stated in zip(names, counts, declared, strict=True):
            if stated < observed:
                raise ValueError(
                    f"Condition {name!r} declares {stated} levels but the data use {observed}."
                )
        counts = declared

    domain = MultiValueDomain(tuple(names), counts)
    y = validate_membership(data[outcome].to_numpy(), name=outcome)
    codes = data[names].to_numpy().astype(np.int64)

    if case_id is None:
        labels = np.asarray([str(index) for index in data.index], dtype=object)
    else:
        validate_columns(data, [case_id])
        labels = data[case_id].astype(str).to_numpy(dtype=object)

    rows: list[MultiValueRow] = []
    for configuration in domain.configurations():
        selector = np.all(codes == np.asarray(configuration), axis=1)
        n = int(selector.sum())
        # Each case belongs to exactly one configuration, so membership is crisp.
        membership = selector.astype(np.float64)
        fit = sufficiency(membership, y)

        reason: str | None = None
        if n < frequency_cutoff:
            code: MultiValueCode = "R"
            reason = f"frequency {n} below the cutoff of {frequency_cutoff}"
        elif fit.consistency >= inclusion_cutoff:
            code = "1"
        else:
            code = "0"
            reason = (
                f"consistency {fit.consistency:.3f} below the inclusion cutoff "
                f"of {inclusion_cutoff}"
            )

        rows.append(
            MultiValueRow(
                index=domain.index_of(configuration),
                configuration=configuration,
                frequency=n,
                consistency=fit.consistency,
                pri=fit.pri,
                outcome=code,
                cases=tuple(str(value) for value in labels[selector]),
                exclusion_reason=reason,
            )
        )

    return MultiValueTruthTable(
        domain=domain,
        outcome_name=outcome,
        rows=tuple(rows),
        inclusion_cutoff=inclusion_cutoff,
        frequency_cutoff=frequency_cutoff,
    )


@dataclass(slots=True)
class MVQCA:
    """Multi-value Qualitative Comparative Analysis estimator.

    Parameters
    ----------
    consistency : float, default 0.8
        Inclusion cutoff on sufficiency consistency.
    frequency : int, default 1
        Minimum number of cases for a configuration to be observed.
    max_solutions : int, default 256
        Upper bound on tied minimal covers.
    levels : mapping of str to int, optional
        Declared number of categories per condition. Supply this when a level
        is theoretically possible but happens to have no cases, since it
        changes the property space and therefore the remainders.
    """

    consistency: float = 0.8
    frequency: int = 1
    max_solutions: int = 256
    levels: Mapping[str, int] | None = None

    def fit(
        self,
        data: pd.DataFrame,
        *,
        outcome: str,
        conditions: list[str] | tuple[str, ...],
        case_id: str | None = None,
    ) -> MultiValueResult:
        """Fit mvQCA to categorical conditions and a calibrated outcome."""
        table = build_multivalue_truth_table(
            data,
            outcome=outcome,
            conditions=conditions,
            levels=self.levels,
            inclusion_cutoff=self.consistency,
            frequency_cutoff=self.frequency,
            case_id=case_id,
        )
        conservative = table.minimize(max_solutions=self.max_solutions)
        parsimonious = table.minimize(include_remainders=True, max_solutions=self.max_solutions)

        y = validate_membership(data[outcome].to_numpy(), name=outcome)
        codes = data[list(table.domain.conditions)].to_numpy().astype(np.int64)
        fits: dict[str, SufficiencyFit] = {}
        for solution in (*conservative, *parsimonious):
            expression = solution.as_expression(table.domain)
            if expression in fits:
                continue
            membership = np.asarray(
                [
                    1.0 if solution.covers(table.domain.index_of(tuple(row)), table.domain) else 0.0
                    for row in codes
                ],
                dtype=np.float64,
            )
            fits[expression] = sufficiency(membership, y)

        return MultiValueResult(
            domain=table.domain,
            outcome=outcome,
            truth_table=table,
            conservative=conservative,
            parsimonious=parsimonious,
            fits=fits,
        )
