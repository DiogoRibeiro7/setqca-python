"""High-level csQCA and fsQCA estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from .minimize.qmc import BooleanSolution, minimize
from .results import FittedSolution, QCAResult, fit_boolean_solution
from .truth_table import TruthTable, build_truth_table

Direction = Literal["+", "-", "0"]
"""Directional expectation: ``"+"`` present, ``"-"`` absent, ``"0"`` no expectation."""


@dataclass(slots=True)
class FSQCA:
    """Fuzzy-set Qualitative Comparative Analysis estimator.

    Parameters
    ----------
    consistency : float, default 0.8
        Inclusion cutoff on sufficiency consistency for truth-table rows.
    pri : float, default 0.0
        Minimum PRI for a row to be coded sufficient.
    frequency : int, default 1
        Minimum number of cases for a row to count as observed.
    exclusion_consistency : float, optional
        Consistency below which a row is coded ``"0"``. Rows between the two
        cutoffs are coded contradictory. Defaults to ``consistency``.
    max_solutions : int, default 256
        Upper bound on the number of tied minimal covers returned.
    directional_expectations : dict of str to Direction, optional
        Theoretical expectations enabling the experimental intermediate
        solution. Empty by default, which skips intermediate minimisation.

    Notes
    -----
    Conservative and parsimonious minimisation use an exact classical
    Quine-McCluskey engine. The current directional intermediate solution is
    explicitly experimental and uses only remainders that do not contradict
    the supplied directional expectations.
    """

    consistency: float = 0.8
    pri: float = 0.0
    frequency: int = 1
    exclusion_consistency: float | None = None
    max_solutions: int = 256
    directional_expectations: dict[str, Direction] = field(default_factory=dict)

    method_name: ClassVar[str] = "Fuzzy-set Qualitative Comparative Analysis"

    def fit(
        self,
        data: pd.DataFrame,
        *,
        outcome: str,
        conditions: list[str] | tuple[str, ...],
        case_id: str | None = None,
    ) -> QCAResult:
        """Fit fsQCA to already calibrated condition and outcome memberships.

        Parameters
        ----------
        data : pandas.DataFrame
            Calibrated memberships in ``[0, 1]``.
        outcome : str
            Name of the outcome column.
        conditions : list of str or tuple of str
            Names of the condition columns.
        case_id : str, optional
            Column holding case labels. Defaults to the frame index.

        Returns
        -------
        QCAResult
            Truth table plus conservative, parsimonious and — when directional
            expectations are supplied — intermediate solutions.

        Raises
        ------
        ValueError
            If no truth-table row is sufficient under the chosen thresholds.
        """
        self._validate_thresholds()
        truth_table = build_truth_table(
            data,
            outcome=outcome,
            conditions=conditions,
            inclusion_cutoff=self.consistency,
            exclusion_cutoff=self.exclusion_consistency,
            pri_cutoff=self.pri,
            frequency_cutoff=self.frequency,
            case_id=case_id,
        )
        return self._fit_from_truth_table(data, truth_table)

    def _validate_thresholds(self) -> None:
        if self.max_solutions < 1:
            raise ValueError("max_solutions must be at least 1.")
        for condition, direction in self.directional_expectations.items():
            if direction not in ("+", "-", "0"):
                raise ValueError(
                    f"Directional expectation for {condition!r} must be '+', '-' or '0'; "
                    f"got {direction!r}."
                )

    def _fit_from_truth_table(self, data: pd.DataFrame, truth_table: TruthTable) -> QCAResult:
        on_set = truth_table.positive_minterms
        if not on_set:
            raise ValueError("No truth-table row is sufficient under the chosen thresholds.")
        width = len(truth_table.conditions)
        conservative_raw = minimize(on_set, width=width, max_solutions=self.max_solutions)
        parsimonious_raw = minimize(
            on_set,
            dont_cares=truth_table.remainder_minterms,
            width=width,
            max_solutions=self.max_solutions,
        )
        intermediate_raw: tuple[BooleanSolution, ...] | None = None
        if self.directional_expectations:
            intermediate_raw = minimize(
                on_set,
                dont_cares=self._directionally_allowed_remainders(truth_table),
                width=width,
                max_solutions=self.max_solutions,
            )

        def fitted(raw: tuple[BooleanSolution, ...]) -> tuple[FittedSolution, ...]:
            return tuple(
                fit_boolean_solution(
                    solution,
                    data=data,
                    outcome=truth_table.outcome_name,
                    conditions=truth_table.conditions,
                )
                for solution in raw
            )

        return QCAResult(
            method=self.method_name,
            outcome=truth_table.outcome_name,
            conditions=truth_table.conditions,
            truth_table=truth_table,
            conservative=fitted(conservative_raw),
            parsimonious=fitted(parsimonious_raw),
            intermediate=None if intermediate_raw is None else fitted(intermediate_raw),
            intermediate_experimental=intermediate_raw is not None,
        )

    def _directionally_allowed_remainders(self, truth_table: TruthTable) -> set[int]:
        """Return remainders consistent with every directional expectation."""
        for condition in self.directional_expectations:
            if condition not in truth_table.conditions:
                raise KeyError(
                    f"Directional expectation references unknown condition {condition!r}."
                )
        expectations = [
            (truth_table.conditions.index(condition), 1 if direction == "+" else 0)
            for condition, direction in self.directional_expectations.items()
            if direction != "0"
        ]
        return {
            row.minterm
            for row in truth_table.rows
            if row.outcome == "R"
            and all(row.configuration[index] == state for index, state in expectations)
        }


@dataclass(slots=True)
class CSQCA(FSQCA):
    """Crisp-set QCA with strict 0/1 input validation.

    Identical to :class:`FSQCA` except that every condition and the outcome
    must already be calibrated to binary membership, and the default inclusion
    cutoff is perfect consistency.
    """

    consistency: float = 1.0

    method_name: ClassVar[str] = "Crisp-set Qualitative Comparative Analysis"

    def fit(
        self,
        data: pd.DataFrame,
        *,
        outcome: str,
        conditions: list[str] | tuple[str, ...],
        case_id: str | None = None,
    ) -> QCAResult:
        """Fit csQCA after verifying that every input column is crisp.

        Raises
        ------
        ValueError
            If any condition or the outcome contains values other than 0 or 1.
        """
        for column in (*conditions, outcome):
            values = data[column].to_numpy(dtype=np.float64)
            if not np.isin(values, (0.0, 1.0)).all():
                raise ValueError(f"CSQCA requires binary 0/1 calibration; {column!r} is not crisp.")
        # `dataclass(slots=True)` rebuilds the class object, which leaves the
        # implicit `__class__` cell of zero-argument `super()` pointing at the
        # discarded original. The explicit unbound call is the supported form.
        return FSQCA.fit(self, data, outcome=outcome, conditions=conditions, case_id=case_id)
