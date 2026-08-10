"""Truth-table construction for crisp-set and fuzzy-set QCA."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd

from ._validation import FloatArray, validate_columns, validate_membership
from .metrics import sufficiency

TruthCode = Literal["1", "0", "C", "R"]
"""Outcome code of a truth-table row.

``"1"`` sufficient, ``"0"`` not sufficient, ``"C"`` contradictory,
``"R"`` logical remainder.
"""


@dataclass(frozen=True, slots=True)
class TruthTableRow:
    """A single causal configuration and its empirical fit."""

    minterm: int
    configuration: tuple[int, ...]
    frequency: int
    consistency: float
    pri: float
    outcome: TruthCode
    cases: tuple[str, ...]

    @property
    def observed(self) -> bool:
        """Return whether the configuration passed the frequency cutoff."""
        return self.outcome != "R"


@dataclass(frozen=True, slots=True)
class TruthTable:
    """Immutable QCA truth table covering every logically possible corner."""

    conditions: tuple[str, ...]
    outcome_name: str
    rows: tuple[TruthTableRow, ...]
    inclusion_cutoff: float
    exclusion_cutoff: float
    pri_cutoff: float
    frequency_cutoff: int

    @property
    def positive_minterms(self) -> set[int]:
        """Return minterms of rows coded sufficient for the outcome."""
        return {row.minterm for row in self.rows if row.outcome == "1"}

    @property
    def negative_minterms(self) -> set[int]:
        """Return minterms of rows coded not sufficient for the outcome."""
        return {row.minterm for row in self.rows if row.outcome == "0"}

    @property
    def contradictory_minterms(self) -> set[int]:
        """Return minterms of rows falling between the exclusion and inclusion cutoffs."""
        return {row.minterm for row in self.rows if row.outcome == "C"}

    @property
    def remainder_minterms(self) -> set[int]:
        """Return minterms of logical remainders, i.e. rows below the frequency cutoff."""
        return {row.minterm for row in self.rows if row.outcome == "R"}

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy pandas representation of the truth table.

        Returns
        -------
        pandas.DataFrame
            One row per configuration, with the condition states followed by
            ``minterm``, ``n``, ``consistency``, ``PRI``, ``OUT`` and ``cases``.
        """
        records: list[dict[str, object]] = []
        for row in self.rows:
            record: dict[str, object] = dict(zip(self.conditions, row.configuration, strict=True))
            record.update(
                {
                    "minterm": row.minterm,
                    "n": row.frequency,
                    "consistency": row.consistency,
                    "PRI": row.pri,
                    "OUT": row.outcome,
                    "cases": ", ".join(row.cases),
                }
            )
            records.append(record)
        return pd.DataFrame.from_records(records)


def _configuration_membership(memberships: FloatArray, config: tuple[int, ...]) -> FloatArray:
    """Return membership in a corner of the property space.

    Parameters
    ----------
    memberships : FloatArray
        Two-dimensional ``(n_cases, n_conditions)`` array of calibrated scores.
    config : tuple of int
        Corner of the property space, one 0/1 state per condition.
    """
    states = np.asarray(config, dtype=np.float64)
    # Negated conditions contribute ``1 - x``; asserted conditions contribute ``x``.
    oriented = np.where(states == 1.0, memberships, 1.0 - memberships)
    return np.min(oriented, axis=1).astype(np.float64)


def _minterm(config: tuple[int, ...]) -> int:
    """Encode a configuration as a big-endian binary minterm index."""
    value = 0
    for bit in config:
        value = (value << 1) | bit
    return value


def build_truth_table(
    data: pd.DataFrame,
    *,
    outcome: str,
    conditions: list[str] | tuple[str, ...],
    inclusion_cutoff: float = 0.8,
    exclusion_cutoff: float | None = None,
    pri_cutoff: float = 0.0,
    frequency_cutoff: int = 1,
    case_id: str | None = None,
    allow_crossover_cases: bool = False,
) -> TruthTable:
    """Construct a complete binary truth table from calibrated data.

    Fuzzy cases are assigned to the crisp truth-table corner implied by scores
    above/below 0.5. Cases exactly at the crossover are rejected by default
    because their corner assignment is ambiguous.

    Parameters
    ----------
    data : pandas.DataFrame
        Calibrated condition and outcome memberships in ``[0, 1]``.
    outcome : str
        Name of the outcome column.
    conditions : list of str or tuple of str
        Names of the condition columns, in the order used for minterm coding.
    inclusion_cutoff : float, default 0.8
        Minimum sufficiency consistency for a row to be coded ``"1"``.
    exclusion_cutoff : float, optional
        Consistency below which a row is coded ``"0"``. Rows between the two
        cutoffs are coded ``"C"``. Defaults to ``inclusion_cutoff``, which
        disables the contradictory band.
    pri_cutoff : float, default 0.0
        Minimum PRI for a row to be coded ``"1"``.
    frequency_cutoff : int, default 1
        Minimum number of cases for a row to count as observed.
    case_id : str, optional
        Column holding case labels. Defaults to the frame index.
    allow_crossover_cases : bool, default False
        Permit membership scores of exactly 0.5.

    Returns
    -------
    TruthTable
        Complete table with one row per corner of the property space.

    Raises
    ------
    TypeError
        If ``data`` is not a :class:`pandas.DataFrame`.
    ValueError
        If any cutoff is out of range, memberships fall outside ``[0, 1]``, or
        a case sits exactly on the crossover while ``allow_crossover_cases``
        is ``False``.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    conds = validate_columns(data, conditions)
    if not conds:
        raise ValueError("At least one condition is required.")
    validate_columns(data, [outcome])
    if not 0.0 <= inclusion_cutoff <= 1.0:
        raise ValueError("inclusion_cutoff must be in [0, 1].")
    exclusion = inclusion_cutoff if exclusion_cutoff is None else exclusion_cutoff
    if not 0.0 <= exclusion <= inclusion_cutoff:
        raise ValueError("exclusion_cutoff must be in [0, inclusion_cutoff].")
    if not 0.0 <= pri_cutoff <= 1.0:
        raise ValueError("pri_cutoff must be in [0, 1].")
    if frequency_cutoff < 1:
        raise ValueError("frequency_cutoff must be at least 1.")

    y = validate_membership(data[outcome].to_numpy(), name=outcome)
    x = data[conds].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all() or np.any((x < 0.0) | (x > 1.0)):
        raise ValueError("All conditions must be calibrated memberships in [0, 1].")
    if not allow_crossover_cases and np.isclose(x, 0.5, atol=1e-12).any():
        raise ValueError(
            "At least one condition is exactly 0.5. Truth-table corner assignment is ambiguous; "
            "resolve crossover cases or set allow_crossover_cases=True."
        )

    assigned = (x >= 0.5).astype(np.int8)
    if case_id is None:
        case_names = np.asarray([str(idx) for idx in data.index], dtype=object)
    else:
        validate_columns(data, [case_id])
        case_names = data[case_id].astype(str).to_numpy(dtype=object)

    rows: list[TruthTableRow] = []
    for config_raw in product((0, 1), repeat=len(conds)):
        config = tuple(int(v) for v in config_raw)
        selector = np.all(assigned == np.asarray(config), axis=1)
        n = int(selector.sum())
        membership = _configuration_membership(x, config)
        fit = sufficiency(membership, y)
        if n < frequency_cutoff:
            code: TruthCode = "R"
        elif fit.consistency >= inclusion_cutoff and fit.pri >= pri_cutoff:
            code = "1"
        elif fit.consistency >= exclusion:
            code = "C"
        else:
            code = "0"
        rows.append(
            TruthTableRow(
                minterm=_minterm(config),
                configuration=config,
                frequency=n,
                consistency=fit.consistency,
                pri=fit.pri,
                outcome=code,
                cases=tuple(str(v) for v in case_names[selector]),
            )
        )

    return TruthTable(
        conditions=tuple(conds),
        outcome_name=outcome,
        rows=tuple(rows),
        inclusion_cutoff=inclusion_cutoff,
        exclusion_cutoff=exclusion,
        pri_cutoff=pri_cutoff,
        frequency_cutoff=frequency_cutoff,
    )
