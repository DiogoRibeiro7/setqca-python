"""Truth-table construction for crisp-set and fuzzy-set QCA."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ._validation import FloatArray, validate_columns, validate_membership
from .metrics import sufficiency

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .minimize.qmc import BooleanSolution

TruthCode = Literal["1", "0", "C", "R"]
"""Outcome code of a truth-table row.

``"1"`` sufficient, ``"0"`` not sufficient, ``"C"`` contradictory,
``"R"`` logical remainder.
"""


@dataclass(frozen=True, slots=True)
class TruthTableRow:
    """A single causal configuration and its empirical fit.

    Attributes
    ----------
    exclusion_reason
        Why the row is not coded sufficient, in words. ``None`` for rows coded
        ``"1"``. Recorded because the outcome code alone conflates distinct
        situations: a row can miss out for lack of cases, for low consistency,
        or for low PRI, and those call for different responses.
    """

    minterm: int
    configuration: tuple[int, ...]
    frequency: int
    consistency: float
    pri: float
    outcome: TruthCode
    cases: tuple[str, ...]
    exclusion_reason: str | None = None

    @property
    def observed(self) -> bool:
        """Return whether the configuration passed the frequency cutoff."""
        return self.outcome != "R"

    @property
    def excluded_by_threshold(self) -> bool:
        """Return whether a threshold, rather than the evidence, kept this row out.

        True for rows held back by the frequency or PRI cutoffs. A row with
        genuinely low consistency is excluded by the data, not by a choice.
        """
        return (
            self.outcome != "1"
            and self.exclusion_reason is not None
            and ("frequency" in self.exclusion_reason or "PRI" in self.exclusion_reason)
        )


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

    def rows_with(self, code: TruthCode) -> tuple[TruthTableRow, ...]:
        """Return the rows carrying one outcome code, in minterm order."""
        return tuple(row for row in self.rows if row.outcome == code)

    def positive_rows(self) -> tuple[TruthTableRow, ...]:
        """Return rows coded sufficient for the outcome."""
        return self.rows_with("1")

    def negative_rows(self) -> tuple[TruthTableRow, ...]:
        """Return rows coded not sufficient."""
        return self.rows_with("0")

    def contradictions(self) -> tuple[TruthTableRow, ...]:
        """Return rows falling between the exclusion and inclusion cutoffs."""
        return self.rows_with("C")

    def remainders(self) -> tuple[TruthTableRow, ...]:
        """Return logical remainders: rows below the frequency cutoff."""
        return self.rows_with("R")

    def excluded_rows(self) -> tuple[TruthTableRow, ...]:
        """Return rows a *threshold* kept out, rather than the evidence.

        These are the rows whose exclusion is a consequence of an analytical
        choice — the frequency or PRI cutoff — and therefore the rows to
        revisit when judging how much the result depends on those choices. A
        row with genuinely low consistency is excluded by the data and is not
        listed here.
        """
        return tuple(row for row in self.rows if row.excluded_by_threshold)

    def summary(self) -> str:
        """Return a short account of how the table came out."""
        return (
            f"{len(self.rows)} configurations of {len(self.conditions)} conditions "
            f"({self.outcome_name})\n"
            f"  sufficient:    {len(self.positive_rows())}\n"
            f"  not sufficient:{len(self.negative_rows())}\n"
            f"  contradictory: {len(self.contradictions())}\n"
            f"  remainders:    {len(self.remainders())}\n"
            f"  excluded by a threshold: {len(self.excluded_rows())}"
        )

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy pandas representation of the truth table.

        Returns
        -------
        pandas.DataFrame
            One row per configuration, with the condition states followed by
            ``minterm``, ``n``, ``consistency``, ``PRI``, ``OUT``, ``cases``
            and ``excluded_because``.
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
                    "excluded_because": row.exclusion_reason or "",
                }
            )
            records.append(record)
        return pd.DataFrame.from_records(records)

    def minimize(
        self, *, include_remainders: bool = False, max_solutions: int = 256
    ) -> tuple[BooleanSolution, ...]:
        """Minimise directly from the table, without the original data.

        A stored truth table carries everything Boolean minimisation needs, so
        a saved table can be re-minimised under different assumptions without
        recalibrating or rebuilding it.

        Parameters
        ----------
        include_remainders : bool, default False
            Treat logical remainders as don't-cares, giving the parsimonious
            solution rather than the conservative one.
        max_solutions : int, default 256
            Upper bound on tied minimal covers.

        Returns
        -------
        tuple of BooleanSolution
            Boolean covers only. Case-level parameters of fit need the original
            data and are produced by :meth:`~setqca.FSQCA.fit`.

        Raises
        ------
        ValueError
            If no row is coded sufficient.
        """
        from .minimize.qmc import minimize as _minimize

        on_set = self.positive_minterms
        if not on_set:
            raise ValueError("No truth-table row is sufficient under the chosen thresholds.")
        return _minimize(
            on_set,
            dont_cares=self.remainder_minterms if include_remainders else None,
            width=len(self.conditions),
            max_solutions=max_solutions,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary describing the whole table."""
        return {
            "conditions": list(self.conditions),
            "outcome": self.outcome_name,
            "inclusion_cutoff": self.inclusion_cutoff,
            "exclusion_cutoff": self.exclusion_cutoff,
            "pri_cutoff": self.pri_cutoff,
            "frequency_cutoff": self.frequency_cutoff,
            "rows": [
                {
                    "minterm": row.minterm,
                    "configuration": list(row.configuration),
                    "n": row.frequency,
                    "consistency": row.consistency,
                    "pri": row.pri,
                    "out": row.outcome,
                    "cases": list(row.cases),
                    "excluded_because": row.exclusion_reason,
                }
                for row in self.rows
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TruthTable:
        """Rebuild a table from :meth:`to_dict` output.

        Raises
        ------
        KeyError
            If a required key is missing.
        """
        rows = tuple(
            TruthTableRow(
                minterm=int(record["minterm"]),
                configuration=tuple(int(value) for value in record["configuration"]),
                frequency=int(record["n"]),
                consistency=float(record["consistency"]),
                pri=float(record["pri"]),
                outcome=record["out"],
                cases=tuple(str(case) for case in record["cases"]),
                exclusion_reason=record.get("excluded_because"),
            )
            for record in payload["rows"]
        )
        return cls(
            conditions=tuple(payload["conditions"]),
            outcome_name=payload["outcome"],
            rows=rows,
            inclusion_cutoff=float(payload["inclusion_cutoff"]),
            exclusion_cutoff=float(payload["exclusion_cutoff"]),
            pri_cutoff=float(payload["pri_cutoff"]),
            frequency_cutoff=int(payload["frequency_cutoff"]),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialise the table to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> TruthTable:
        """Rebuild a table from JSON."""
        return cls.from_dict(json.loads(text))


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
    # Bind through an annotated local: the element type numpy's stubs infer for a
    # reduction varies between releases, and returning it directly makes the
    # strict-mode result depend on which numpy happens to be installed.
    membership: FloatArray = np.min(oriented, axis=1).astype(np.float64)
    return membership


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
        reason: str | None = None
        if n < frequency_cutoff:
            code: TruthCode = "R"
            reason = f"frequency {n} below the cutoff of {frequency_cutoff}"
        elif fit.consistency >= inclusion_cutoff and fit.pri >= pri_cutoff:
            code = "1"
        else:
            # Consistency and PRI fail for different reasons and warrant
            # different responses, so both are named rather than collapsed
            # into the outcome code.
            failures = []
            if fit.consistency < inclusion_cutoff:
                failures.append(
                    f"consistency {fit.consistency:.3f} below the inclusion "
                    f"cutoff of {inclusion_cutoff}"
                )
            if fit.pri < pri_cutoff:
                failures.append(f"PRI {fit.pri:.3f} below the cutoff of {pri_cutoff}")
            reason = "; ".join(failures)
            code = "C" if fit.consistency >= exclusion else "0"
        rows.append(
            TruthTableRow(
                minterm=_minterm(config),
                configuration=config,
                frequency=n,
                consistency=fit.consistency,
                pri=fit.pri,
                outcome=code,
                cases=tuple(str(v) for v in case_names[selector]),
                exclusion_reason=reason,
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
