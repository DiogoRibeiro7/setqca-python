"""Diagnostics for calibrated memberships, and quantile helpers for anchors.

A calibration can be arithmetically valid and analytically useless. These
checks catch the failures that quietly ruin a truth table: everything piled at
the crossover, everything crushed to 0 and 1, or no variation at all.

None of these is fatal by itself. They are reported so the researcher can
decide, not enforced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from setqca._validation import as_float_array, validate_membership

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    import numpy.typing as npt

    from setqca._validation import FloatArray

CROSSOVER_BAND = 0.05
EXTREME_BAND = 0.05
PILE_UP_SHARE = 0.25
COMPRESSION_SHARE = 0.90
LOW_VARIANCE = 0.05


@dataclass(frozen=True, slots=True)
class CalibrationDiagnostics:
    """What a calibrated vector looks like, and what is worrying about it.

    Attributes
    ----------
    n
        Number of cases.
    at_crossover
        Cases with membership exactly 0.5. Their truth-table corner is
        undefined, so these block analysis rather than merely warning.
    near_crossover
        Cases within ``CROSSOVER_BAND`` of 0.5.
    extreme
        Cases within ``EXTREME_BAND`` of 0 or 1.
    minimum, maximum, mean, standard_deviation
        Summary statistics of the calibrated values.
    above_crossover
        Cases that will be assigned the "present" corner.
    warnings
        Human-readable descriptions of every issue found.
    """

    n: int
    at_crossover: int
    near_crossover: int
    extreme: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    above_crossover: int
    warnings: tuple[str, ...]

    @property
    def usable(self) -> bool:
        """Return whether the vector can be used for a truth table at all.

        Only an exact crossover membership makes a vector unusable; everything
        else is a judgement call left to the researcher.
        """
        return self.at_crossover == 0

    @property
    def pile_up_share(self) -> float:
        """Return the proportion of cases bunched around the crossover."""
        return self.near_crossover / self.n if self.n else 0.0

    @property
    def compression_share(self) -> float:
        """Return the proportion of cases pushed to the extremes."""
        return self.extreme / self.n if self.n else 0.0

    def __str__(self) -> str:
        lines = [
            f"n={self.n}, mean={self.mean:.3f}, sd={self.standard_deviation:.3f}, "
            f"range=[{self.minimum:.3f}, {self.maximum:.3f}]",
            f"above crossover: {self.above_crossover}/{self.n}",
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        if not self.warnings:
            lines.append("  no issues found")
        return "\n".join(lines)


def diagnose_calibration(values: npt.ArrayLike, *, name: str = "values") -> CalibrationDiagnostics:
    """Inspect a calibrated vector for the failures that spoil a truth table.

    Parameters
    ----------
    values : array_like
        Calibrated memberships in ``[0, 1]``.
    name : str, default "values"
        Name used in the warning messages.

    Returns
    -------
    CalibrationDiagnostics
        Summary statistics and a list of warnings.

    Raises
    ------
    ValueError
        If the values are not calibrated memberships.
    """
    membership = validate_membership(values, name=name)
    n = int(membership.size)

    at_crossover = int(np.sum(np.isclose(membership, 0.5, atol=1e-12)))
    near_crossover = int(np.sum(np.abs(membership - 0.5) < CROSSOVER_BAND))
    extreme = int(np.sum((membership < EXTREME_BAND) | (membership > 1.0 - EXTREME_BAND)))
    above = int(np.sum(membership >= 0.5))
    deviation = float(np.std(membership))

    warnings: list[str] = []
    if at_crossover:
        warnings.append(
            f"{at_crossover} case(s) sit exactly at 0.5, where the truth-table "
            "corner is undefined; resolve them or set allow_crossover_cases"
        )
    if n and near_crossover / n > PILE_UP_SHARE:
        warnings.append(
            f"{near_crossover}/{n} cases lie within {CROSSOVER_BAND} of the crossover; "
            "small anchor changes will move them between corners"
        )
    if n and extreme / n > COMPRESSION_SHARE:
        warnings.append(
            f"{extreme}/{n} cases are at the extremes; the calibration is close to "
            "crisp and the fuzzy detail has been squeezed out"
        )
    if deviation < LOW_VARIANCE:
        warnings.append(
            f"standard deviation is {deviation:.3f}; the condition barely varies and "
            "will carry little information"
        )
    if above == 0:
        warnings.append("no case is above the crossover; the condition is never present")
    elif above == n:
        warnings.append("every case is above the crossover; the condition is never absent")

    return CalibrationDiagnostics(
        n=n,
        at_crossover=at_crossover,
        near_crossover=near_crossover,
        extreme=extreme,
        minimum=float(np.min(membership)),
        maximum=float(np.max(membership)),
        mean=float(np.mean(membership)),
        standard_deviation=deviation,
        above_crossover=above,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True, slots=True)
class AnchorSuggestion:
    """Quantiles of a raw variable, offered as a starting point only.

    Attributes
    ----------
    quantiles
        The probabilities used.
    values
        The corresponding raw values, in ascending order.
    caveat
        A standing reminder that these are not a calibration.
    """

    quantiles: tuple[float, float, float]
    values: tuple[float, float, float]
    caveat: str = (
        "Quantiles describe the sample, not the concept. Anchors must be "
        "justified substantively; these are a starting point for that argument, "
        "not a substitute for it."
    )

    @property
    def anchors(self) -> tuple[float, float, float]:
        """Return the suggested ``(full_out, crossover, full_in)``."""
        return self.values

    def __str__(self) -> str:
        low, mid, high = self.values
        return (
            f"Suggested from quantiles {self.quantiles}: "
            f"full_out={low:g}, crossover={mid:g}, full_in={high:g}\n"
            f"  {self.caveat}"
        )


def suggest_anchors(
    values: npt.ArrayLike,
    *,
    quantiles: tuple[float, float, float] = (0.05, 0.50, 0.95),
) -> AnchorSuggestion:
    """Report quantiles of a raw variable to inform an anchor decision.

    This is a **diagnostic**, not a calibration. Data-driven anchors describe
    the sample rather than the concept, and a set defined by its own
    distribution cannot support a claim about set membership. Use the output to
    see where your cases actually lie, then argue for anchors on substantive
    grounds.

    Parameters
    ----------
    values : array_like
        Raw, uncalibrated measures.
    quantiles : tuple of float, default (0.05, 0.50, 0.95)
        Probabilities for the exclusion, crossover and inclusion anchors.

    Returns
    -------
    AnchorSuggestion
        The quantile values, with the caveat attached.

    Raises
    ------
    ValueError
        If the quantiles are not strictly increasing and inside ``[0, 1]``.
    """
    if not all(0.0 <= q <= 1.0 for q in quantiles):
        raise ValueError("Quantiles must lie in [0, 1].")
    if not quantiles[0] < quantiles[1] < quantiles[2]:
        raise ValueError("Quantiles must be strictly increasing.")

    raw = as_float_array(values, name="values")
    low, mid, high = (float(value) for value in np.quantile(raw, quantiles))
    if not low < mid < high:
        raise ValueError(
            "The requested quantiles are not distinct in this sample, so they "
            "cannot serve as anchors; the variable may be too concentrated."
        )
    return AnchorSuggestion(quantiles=quantiles, values=(low, mid, high))


def diagnose_frame(
    data: pd.DataFrame, columns: list[str] | tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Diagnose every calibrated column and return a tidy summary.

    Parameters
    ----------
    data : pandas.DataFrame
        Calibrated memberships.
    columns : list of str, optional
        Columns to check. Defaults to every column.

    Returns
    -------
    pandas.DataFrame
        One row per column, with the summary statistics and a joined warning
        string.
    """
    names = list(data.columns) if columns is None else list(columns)
    records = []
    for name in names:
        diagnostics = diagnose_calibration(data[name].to_numpy(), name=name)
        records.append(
            {
                "condition": name,
                "n": diagnostics.n,
                "mean": diagnostics.mean,
                "sd": diagnostics.standard_deviation,
                "min": diagnostics.minimum,
                "max": diagnostics.maximum,
                "at_crossover": diagnostics.at_crossover,
                "near_crossover": diagnostics.near_crossover,
                "extreme": diagnostics.extreme,
                "above_crossover": diagnostics.above_crossover,
                "usable": diagnostics.usable,
                "warnings": "; ".join(diagnostics.warnings),
            }
        )
    return pd.DataFrame.from_records(records)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Calibrated values together with the specification that produced them."""

    spec: object
    values: FloatArray
    diagnostics: CalibrationDiagnostics

    def to_frame(self) -> pd.DataFrame:
        """Return the calibrated values as a one-column frame."""
        condition = getattr(self.spec, "condition", "values")
        return pd.DataFrame({condition: self.values})

    def __str__(self) -> str:
        condition = getattr(self.spec, "condition", "values")
        return f"Calibration of {condition}\n{self.diagnostics}"
