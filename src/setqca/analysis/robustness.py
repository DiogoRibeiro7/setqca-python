"""Sensitivity of a QCA result to the choices that produced it.

A QCA solution is conditional on decisions the data do not make for you: where
the calibration anchors sit, how consistent a row must be to count as
sufficient, how many cases a row needs. Reporting one solution from one set of
those choices hides how much of the result was the choice rather than the
evidence.

This module runs the analysis across a grid of those choices and reports which
paths survive.

What robustness is not
----------------------

A path that appears under every threshold is **stable**, not **true**. Stability
says the finding does not depend on one arbitrary cutoff. It says nothing about
whether the conditions are causally relevant, whether the calibration was
substantively sensible, or whether the case selection was sound. A thoroughly
mis-specified model can be perfectly stable.

Nothing here reports a verdict. The measures are descriptive, and the
interpretation is the researcher's.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from setqca.calibration import calibrate_direct
from setqca.expressions import evaluate_expression
from setqca.models import FSQCA

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from setqca._validation import FloatArray
    from setqca.models import Direction

DEFAULT_STABILITY = 0.80


@dataclass(frozen=True, slots=True)
class Specification:
    """One combination of analytical choices."""

    consistency: float
    pri: float
    frequency: int
    anchors: tuple[tuple[str, tuple[float, float, float]], ...] = ()

    def __str__(self) -> str:
        base = f"cons={self.consistency:g}, pri={self.pri:g}, n={self.frequency}"
        if not self.anchors:
            return base
        rendered = "; ".join(
            f"{name}={anchor[0]:g}/{anchor[1]:g}/{anchor[2]:g}" for name, anchor in self.anchors
        )
        return f"{base}, anchors[{rendered}]"


@dataclass(frozen=True, slots=True)
class RobustnessGrid:
    """The analytical choices to sweep.

    Parameters
    ----------
    consistency, pri : sequence of float
        Inclusion and PRI cutoffs to try.
    frequency : sequence of int
        Frequency cutoffs to try.
    anchors : mapping of str to sequence of (float, float, float)
        Alternative calibration anchors per condition, as
        ``(full_out, crossover, full_in)``. Only usable with raw data through
        :func:`calibration_robustness`.
    """

    consistency: Sequence[float] = (0.75, 0.80, 0.85)
    pri: Sequence[float] = (0.0,)
    frequency: Sequence[int] = (1,)
    anchors: Mapping[str, Sequence[tuple[float, float, float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.consistency or not self.pri or not self.frequency:
            raise ValueError("Every grid axis needs at least one value.")
        for value in (*self.consistency, *self.pri):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Consistency and PRI cutoffs must be in [0, 1].")
        if any(value < 1 for value in self.frequency):
            raise ValueError("Frequency cutoffs must be at least 1.")

    def specifications(self) -> Iterator[Specification]:
        """Yield every combination in the grid, in a deterministic order."""
        names = sorted(self.anchors)
        anchor_options: list[list[tuple[str, tuple[float, float, float]]]] = [
            [(name, tuple(anchor)) for anchor in self.anchors[name]]  # type: ignore[misc]
            for name in names
        ]
        combinations: Sequence[tuple[tuple[str, tuple[float, float, float]], ...]] = (
            list(product(*anchor_options)) if anchor_options else [()]
        )
        for consistency, pri, frequency, anchors in product(
            self.consistency, self.pri, self.frequency, combinations
        ):
            yield Specification(
                consistency=float(consistency),
                pri=float(pri),
                frequency=int(frequency),
                anchors=tuple(anchors),
            )

    def __len__(self) -> int:
        return sum(1 for _ in self.specifications())


@dataclass(frozen=True, slots=True)
class RobustnessRun:
    """The outcome of one specification.

    A specification that produces no solution is recorded rather than dropped:
    "the model collapses above 0.85" is itself a finding.
    """

    specification: Specification
    terms: frozenset[str]
    consistency: float
    coverage: float
    implicants: int
    literals: int
    solutions: int
    failure: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether a solution was produced."""
        return self.failure is None


@dataclass(frozen=True, slots=True)
class TermStability:
    """How often one term survived the sweep."""

    term: str
    appearances: int
    total: int
    in_baseline: bool

    @property
    def share(self) -> float:
        """Return the proportion of successful runs containing the term."""
        return self.appearances / self.total if self.total else 0.0

    def stable(self, threshold: float = DEFAULT_STABILITY) -> bool:
        """Return whether the term appears in at least ``threshold`` of runs."""
        return self.share >= threshold


@dataclass(frozen=True, slots=True)
class SolutionSimilarity:
    """Several ways two solutions can resemble each other.

    Attributes
    ----------
    identical
        The two term sets are equal.
    term_overlap
        Jaccard index over term sets: exact string agreement.
    configurational
        Jaccard index over the literals used, so solutions that differ in how
        terms are cut but use the same conditions still score highly.
    membership
        Fuzzy Jaccard over case membership in the solution,
        ``Σ min(a, b) / Σ max(a, b)``. Two solutions can differ textually and
        still select the same cases; this is what notices that.
    """

    identical: bool
    term_overlap: float
    configurational: float
    membership: float


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _literals(terms: frozenset[str]) -> frozenset[str]:
    return frozenset(literal for term in terms for literal in term.split("*"))


def _membership(terms: frozenset[str], data: pd.DataFrame) -> FloatArray:
    if not terms:
        return np.zeros(len(data), dtype=np.float64)
    combined: FloatArray = np.maximum.reduce(
        [evaluate_expression(term, data) for term in sorted(terms)]
    )
    return combined


def solution_similarity(
    left: frozenset[str], right: frozenset[str], data: pd.DataFrame
) -> SolutionSimilarity:
    """Compare two solutions on four scales, from strictest to loosest.

    Parameters
    ----------
    left, right : frozenset of str
        Solution terms, in standard QCA notation.
    data : pandas.DataFrame
        Calibrated data, used for the membership comparison.

    Returns
    -------
    SolutionSimilarity
        Exact identity, term overlap, configurational overlap and membership
        agreement.
    """
    left_membership = _membership(left, data)
    right_membership = _membership(right, data)
    union = float(np.maximum(left_membership, right_membership).sum())
    membership = (
        1.0 if union == 0.0 else float(np.minimum(left_membership, right_membership).sum()) / union
    )
    return SolutionSimilarity(
        identical=left == right,
        term_overlap=_jaccard(left, right),
        configurational=_jaccard(_literals(left), _literals(right)),
        membership=membership,
    )


@dataclass(frozen=True, slots=True)
class RobustnessAnalysis:
    """The result of sweeping a grid of analytical choices."""

    grid: RobustnessGrid
    runs: tuple[RobustnessRun, ...]
    baseline: Specification
    family: str
    data: pd.DataFrame = field(repr=False)

    @property
    def successful(self) -> tuple[RobustnessRun, ...]:
        """Return runs that produced a solution."""
        return tuple(run for run in self.runs if run.succeeded)

    @property
    def failed(self) -> tuple[RobustnessRun, ...]:
        """Return specifications under which the model produced nothing."""
        return tuple(run for run in self.runs if not run.succeeded)

    @property
    def baseline_terms(self) -> frozenset[str]:
        """Return the terms of the baseline specification, if it succeeded."""
        for run in self.runs:
            if run.specification == self.baseline and run.succeeded:
                return run.terms
        return frozenset()

    def term_stability(self) -> tuple[TermStability, ...]:
        """Return every term seen, with how often it survived."""
        successful = self.successful
        total = len(successful)
        seen: dict[str, int] = {}
        for run in successful:
            for term in run.terms:
                seen[term] = seen.get(term, 0) + 1
        baseline = self.baseline_terms
        return tuple(
            sorted(
                (
                    TermStability(
                        term=term,
                        appearances=count,
                        total=total,
                        in_baseline=term in baseline,
                    )
                    for term, count in seen.items()
                ),
                key=lambda item: (-item.appearances, item.term),
            )
        )

    def stable_terms(self, threshold: float = DEFAULT_STABILITY) -> tuple[str, ...]:
        """Return terms appearing in at least ``threshold`` of successful runs."""
        return tuple(item.term for item in self.term_stability() if item.stable(threshold))

    def fragile_terms(self, threshold: float = DEFAULT_STABILITY) -> tuple[str, ...]:
        """Return terms that appear, but in fewer than ``threshold`` of runs."""
        return tuple(item.term for item in self.term_stability() if not item.stable(threshold))

    def disappearing_terms(self, threshold: float = DEFAULT_STABILITY) -> tuple[str, ...]:
        """Return baseline terms that do not survive the sweep."""
        return tuple(
            item.term
            for item in self.term_stability()
            if item.in_baseline and not item.stable(threshold)
        )

    def emerging_terms(self, threshold: float = DEFAULT_STABILITY) -> tuple[str, ...]:
        """Return stable terms the baseline did not report."""
        return tuple(
            item.term
            for item in self.term_stability()
            if not item.in_baseline and item.stable(threshold)
        )

    def similarity_to_baseline(self) -> tuple[tuple[Specification, SolutionSimilarity], ...]:
        """Compare every successful run against the baseline solution."""
        baseline = self.baseline_terms
        return tuple(
            (run.specification, solution_similarity(baseline, run.terms, self.data))
            for run in self.successful
        )

    def to_frame(self) -> pd.DataFrame:
        """Return one row per specification.

        Returns
        -------
        pandas.DataFrame
            Columns ``consistency_cutoff``, ``pri_cutoff``, ``frequency_cutoff``,
            ``anchors``, ``solution``, ``consistency``, ``coverage``,
            ``n_implicants``, ``n_literals``, ``n_solutions`` and ``failure``.
        """
        return pd.DataFrame(
            {
                "consistency_cutoff": [run.specification.consistency for run in self.runs],
                "pri_cutoff": [run.specification.pri for run in self.runs],
                "frequency_cutoff": [run.specification.frequency for run in self.runs],
                "anchors": [
                    "; ".join(f"{name}={anchor}" for name, anchor in run.specification.anchors)
                    for run in self.runs
                ],
                "solution": [" + ".join(sorted(run.terms)) for run in self.runs],
                "consistency": [run.consistency for run in self.runs],
                "coverage": [run.coverage for run in self.runs],
                "n_implicants": [run.implicants for run in self.runs],
                "n_literals": [run.literals for run in self.runs],
                "n_solutions": [run.solutions for run in self.runs],
                "failure": [run.failure for run in self.runs],
            }
        )

    def __str__(self) -> str:
        stable = self.stable_terms()
        fragile = self.fragile_terms()
        lines = [
            f"Robustness of the {self.family} solution",
            f"Specifications: {len(self.runs)} "
            f"({len(self.successful)} produced a solution, {len(self.failed)} did not)",
            f"Baseline: {self.baseline}",
            "",
            f"Stable terms ({len(stable)}):",
        ]
        lines.extend(
            f"  {item.term} — {item.appearances}/{item.total} specifications"
            for item in self.term_stability()
            if item.stable()
        )
        if fragile:
            lines.append(f"Threshold-sensitive terms ({len(fragile)}):")
            lines.extend(
                f"  {item.term} — {item.appearances}/{item.total} specifications"
                for item in self.term_stability()
                if not item.stable()
            )
        if self.disappearing_terms():
            lines.append(
                "Baseline terms that do not survive: " + ", ".join(self.disappearing_terms())
            )
        if self.emerging_terms():
            lines.append(
                "Stable terms absent from the baseline: " + ", ".join(self.emerging_terms())
            )
        lines.append("")
        lines.append("Stability is not validity: a mis-specified model can be perfectly stable.")
        return "\n".join(lines)


def _run_one(
    data: pd.DataFrame,
    specification: Specification,
    *,
    outcome: str,
    conditions: Sequence[str],
    family: str,
    directional_expectations: Mapping[str, Direction] | None,
    case_id: str | None,
) -> RobustnessRun:
    model = FSQCA(
        consistency=specification.consistency,
        pri=specification.pri,
        frequency=specification.frequency,
        directional_expectations=dict(directional_expectations or {}),
    )
    try:
        result = model.fit(data, outcome=outcome, conditions=list(conditions), case_id=case_id)
        solutions = result.solutions(family)
        if not solutions:
            raise ValueError(f"No {family} solution under this specification.")
    except (ValueError, KeyError, RuntimeError) as error:
        return RobustnessRun(
            specification=specification,
            terms=frozenset(),
            consistency=float("nan"),
            coverage=float("nan"),
            implicants=0,
            literals=0,
            solutions=0,
            failure=str(error),
        )

    # Model ambiguity is reported through `solutions`; the first cover is used
    # for term accounting so that every specification contributes one row.
    chosen = solutions[0]
    terms = frozenset(chosen.expression(result.conditions).split(" + "))
    return RobustnessRun(
        specification=specification,
        terms=terms,
        consistency=chosen.fit.consistency,
        coverage=chosen.fit.coverage,
        implicants=len(chosen.boolean.implicants),
        literals=chosen.boolean.literal_count,
        solutions=len(solutions),
    )


def robustness_analysis(
    data: pd.DataFrame,
    *,
    outcome: str,
    conditions: Sequence[str],
    grid: RobustnessGrid | None = None,
    family: str = "conservative",
    directional_expectations: Mapping[str, Direction] | None = None,
    case_id: str | None = None,
) -> RobustnessAnalysis:
    """Sweep truth-table thresholds and report which paths survive.

    Parameters
    ----------
    data : pandas.DataFrame
        Calibrated memberships in ``[0, 1]``.
    outcome : str
        Name of the outcome column.
    conditions : sequence of str
        Condition columns.
    grid : RobustnessGrid, optional
        Choices to sweep. Defaults to three consistency cutoffs.
    family : str, default "conservative"
        Which solution family to track.
    directional_expectations : mapping, optional
        Required when ``family`` is ``"intermediate"``.
    case_id : str, optional
        Column holding case labels.

    Returns
    -------
    RobustnessAnalysis
        Every specification's result, with term stability across the sweep.

    Raises
    ------
    ValueError
        If the grid specifies calibration anchors, which need raw data — use
        :func:`calibration_robustness` for those.
    """
    grid = grid or RobustnessGrid()
    if grid.anchors:
        raise ValueError(
            "Calibration anchors need raw, uncalibrated data; use calibration_robustness instead."
        )
    specifications = list(grid.specifications())
    runs = tuple(
        _run_one(
            data,
            specification,
            outcome=outcome,
            conditions=conditions,
            family=family,
            directional_expectations=directional_expectations,
            case_id=case_id,
        )
        for specification in specifications
    )
    return RobustnessAnalysis(
        grid=grid,
        runs=runs,
        baseline=specifications[len(specifications) // 2],
        family=family,
        data=data,
    )


def calibration_robustness(
    raw: pd.DataFrame,
    *,
    outcome: str,
    conditions: Sequence[str],
    grid: RobustnessGrid,
    outcome_anchors: tuple[float, float, float],
    base_anchors: Mapping[str, tuple[float, float, float]] | None = None,
    family: str = "conservative",
    directional_expectations: Mapping[str, Direction] | None = None,
    case_id: str | None = None,
) -> RobustnessAnalysis:
    """Sweep calibration anchors as well as thresholds, starting from raw data.

    Calibration is where substantive judgement enters, so it is also where a
    result is most easily manufactured. This recalibrates from the raw measures
    for every anchor combination in the grid.

    Parameters
    ----------
    raw : pandas.DataFrame
        **Uncalibrated** measures.
    outcome : str
        Name of the outcome column.
    conditions : sequence of str
        Condition columns.
    grid : RobustnessGrid
        Must specify ``anchors`` for at least one condition.
    outcome_anchors : tuple of float
        Anchors used to calibrate the outcome, held fixed across the sweep.
    base_anchors : mapping of str to tuple of float, optional
        Anchors for conditions the grid does **not** sweep. Every condition
        needs anchors from one source or the other, since the input is raw.
    family : str, default "conservative"
        Which solution family to track.
    directional_expectations : mapping, optional
        Required when ``family`` is ``"intermediate"``.
    case_id : str, optional
        Column holding case labels.

    Returns
    -------
    RobustnessAnalysis
        As :func:`robustness_analysis`, with anchors recorded per specification.

    Raises
    ------
    ValueError
        If the grid specifies no anchors, or a condition has no anchors at all.
    KeyError
        If anchors name a condition outside the model.
    """
    if not grid.anchors:
        raise ValueError("calibration_robustness needs a grid with anchors.")
    base = dict(base_anchors or {})
    unknown = (set(grid.anchors) | set(base)) - set(conditions)
    if unknown:
        raise KeyError(f"Anchors reference unknown conditions: {sorted(unknown)}")
    unanchored = [name for name in conditions if name not in grid.anchors and name not in base]
    if unanchored:
        raise ValueError(
            f"The input is raw, so every condition needs anchors; missing: {unanchored}. "
            "Supply them through base_anchors, or sweep them in the grid."
        )

    calibrated_outcome = calibrate_direct(
        raw[outcome].to_numpy(),
        full_out=outcome_anchors[0],
        crossover=outcome_anchors[1],
        full_in=outcome_anchors[2],
    )

    specifications = list(grid.specifications())
    runs: list[RobustnessRun] = []
    reference: pd.DataFrame | None = None

    for specification in specifications:
        frame = pd.DataFrame(index=raw.index)
        anchors = dict(specification.anchors)
        for name in conditions:
            low, crossover, high = anchors.get(name) or base[name]
            frame[name] = calibrate_direct(
                raw[name].to_numpy(), full_out=low, crossover=crossover, full_in=high
            )
        frame[outcome] = calibrated_outcome
        if case_id is not None:
            frame[case_id] = raw[case_id].to_numpy()
        if reference is None:
            reference = frame

        runs.append(
            _run_one(
                frame,
                specification,
                outcome=outcome,
                conditions=conditions,
                family=family,
                directional_expectations=directional_expectations,
                case_id=case_id,
            )
        )

    assert reference is not None
    return RobustnessAnalysis(
        grid=grid,
        runs=tuple(runs),
        baseline=specifications[len(specifications) // 2],
        family=family,
        data=reference,
    )
