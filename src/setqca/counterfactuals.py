"""Directional expectations and counterfactual classification.

Intermediate solutions sit between the conservative solution, which uses no
logical remainders, and the parsimonious one, which uses any remainder that
simplifies the result. The choice of which remainders are admissible is a
theoretical claim, not a computational convenience, so it is made explicit
here.

The procedure follows Ragin and Sonnett (2005):

1. A **simplifying assumption** is a remainder the parsimonious solution relies
   on — a row with no cases that had to be treated as sufficient for the
   parsimonious result to hold.
2. A simplifying assumption is an **easy counterfactual** when it can be reached
   from a configuration that *was* observed to be sufficient by changing
   conditions only in the direction the researcher expects to contribute to the
   outcome. Assuming that adding a helpful condition keeps the outcome is a
   mild claim.
3. Any other simplifying assumption is a **difficult counterfactual**: it
   requires assuming the outcome survives a change running against theory.
4. The intermediate solution admits the easy counterfactuals and refuses the
   difficult ones.

References
----------
Ragin, C. C. and Sonnett, J. (2005). Between complexity and parsimony: limited
diversity, counterfactual cases, and comparative analysis. In *Vergleichen in
der Politikwissenschaft*, 180-195.

Schneider, C. Q. and Wagemann, C. (2012). *Set-Theoretic Methods for the Social
Sciences*, chapter 8.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .minimize.qmc import BooleanSolution
    from .truth_table import TruthTable


class DirectionalExpectation(Enum):
    """How a condition is theorised to relate to the outcome.

    Attributes
    ----------
    POSITIVE
        The condition's **presence** is expected to contribute to the outcome.
    NEGATIVE
        The condition's **absence** is expected to contribute to the outcome.
    UNSPECIFIED
        No expectation. Counterfactuals that turn on this condition cannot be
        classified as easy, so they are treated as difficult.
    """

    POSITIVE = "+"
    NEGATIVE = "-"
    UNSPECIFIED = "0"

    @property
    def contributing_state(self) -> int | None:
        """Return the condition state expected to contribute, or ``None``."""
        if self is DirectionalExpectation.POSITIVE:
            return 1
        if self is DirectionalExpectation.NEGATIVE:
            return 0
        return None

    @classmethod
    def coerce(cls, value: DirectionalExpectation | str | int) -> DirectionalExpectation:
        """Accept the enum, the QCA symbols, or the integer coding.

        Parameters
        ----------
        value : DirectionalExpectation or str or int
            ``"+"``/``"present"``/``1`` for positive, ``"-"``/``"absent"``/``0``
            for negative, ``"0"``/``None``-like for unspecified.

        Raises
        ------
        ValueError
            If the value is not a recognised expectation.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):  # bool is an int subclass; reject it explicitly
            raise ValueError(f"Ambiguous directional expectation {value!r}; use '+', '-' or '0'.")
        if isinstance(value, int):
            if value == 1:
                return cls.POSITIVE
            if value == 0:
                return cls.NEGATIVE
            raise ValueError(f"Unknown directional expectation {value!r}; use 1, 0, '+' or '-'.")
        text = str(value).strip().lower()
        lookup = {
            "+": cls.POSITIVE,
            "1": cls.POSITIVE,
            "present": cls.POSITIVE,
            "positive": cls.POSITIVE,
            "-": cls.NEGATIVE,
            "absent": cls.NEGATIVE,
            "negative": cls.NEGATIVE,
            "0": cls.UNSPECIFIED,
            "": cls.UNSPECIFIED,
            "unspecified": cls.UNSPECIFIED,
        }
        # "0" is ambiguous between "absent" and "no expectation". The QCA
        # literature writes no-expectation as "0", so that reading wins, and
        # callers who mean absence write "-".
        if text not in lookup:
            raise ValueError(
                f"Unknown directional expectation {value!r}; "
                "use '+' (present), '-' (absent) or '0' (no expectation)."
            )
        return lookup[text]


Expectations = dict[str, DirectionalExpectation]


@dataclass(frozen=True, slots=True)
class CounterfactualAnalysis:
    """Which remainders the intermediate solution relied on, and why.

    Attributes
    ----------
    expectations
        The directional expectations that produced this classification.
    simplifying_assumptions
        Remainders the parsimonious solution relies on.
    easy
        Simplifying assumptions consistent with the expectations, admitted by
        the intermediate solution.
    difficult
        Simplifying assumptions that contradict the expectations, refused by
        the intermediate solution.
    """

    expectations: Expectations
    simplifying_assumptions: frozenset[int]
    easy: frozenset[int]
    difficult: frozenset[int]

    @property
    def admitted(self) -> frozenset[int]:
        """Return the remainders admitted as don't-cares for the intermediate solution."""
        return self.easy

    def __str__(self) -> str:
        stated = ", ".join(
            f"{name}{expectation.value}"
            for name, expectation in sorted(self.expectations.items())
            if expectation is not DirectionalExpectation.UNSPECIFIED
        )
        return (
            f"Expectations: {stated or 'none'}\n"
            f"Simplifying assumptions: {len(self.simplifying_assumptions)}\n"
            f"  easy (admitted):   {sorted(self.easy)}\n"
            f"  difficult (refused): {sorted(self.difficult)}"
        )


def coerce_expectations(
    expectations: Mapping[str, DirectionalExpectation | str | int],
    conditions: tuple[str, ...],
) -> Expectations:
    """Normalise and validate directional expectations against the conditions.

    Raises
    ------
    KeyError
        If an expectation names a condition absent from the model.
    ValueError
        If an expectation value is not recognised.
    """
    coerced: Expectations = {}
    for name, value in expectations.items():
        if name not in conditions:
            raise KeyError(f"Directional expectation references unknown condition {name!r}.")
        coerced[name] = DirectionalExpectation.coerce(value)
    return coerced


def _bits(minterm: int, width: int) -> tuple[int, ...]:
    return tuple((minterm >> shift) & 1 for shift in reversed(range(width)))


def is_easy_counterfactual(
    remainder: int,
    observed_sufficient: frozenset[int],
    expectations: Expectations,
    conditions: tuple[str, ...],
) -> bool:
    """Return whether a remainder is reachable from an observed sufficient row.

    Reachable means every condition on which the two differ takes, in the
    remainder, the state the expectations say contributes to the outcome. A
    condition with no expectation can never justify a difference.
    """
    width = len(conditions)
    remainder_bits = _bits(remainder, width)
    contributing = [
        expectations.get(name, DirectionalExpectation.UNSPECIFIED).contributing_state
        for name in conditions
    ]

    for observed in observed_sufficient:
        observed_bits = _bits(observed, width)
        justified = True
        for index in range(width):
            if remainder_bits[index] == observed_bits[index]:
                continue
            if contributing[index] is None or remainder_bits[index] != contributing[index]:
                justified = False
                break
        if justified:
            return True
    return False


def classify_counterfactuals(
    truth_table: TruthTable,
    parsimonious: tuple[BooleanSolution, ...],
    expectations: Expectations,
) -> CounterfactualAnalysis:
    """Classify the remainders the parsimonious solution relies on.

    Parameters
    ----------
    truth_table : TruthTable
        The fitted truth table, supplying the observed and remainder rows.
    parsimonious : tuple of BooleanSolution
        Parsimonious solutions, whose covered remainders are the simplifying
        assumptions.
    expectations : dict of str to DirectionalExpectation
        Normalised directional expectations.

    Returns
    -------
    CounterfactualAnalysis
        Simplifying assumptions split into easy and difficult.
    """
    conditions = truth_table.conditions
    remainders = truth_table.remainder_minterms
    observed_sufficient = frozenset(truth_table.positive_minterms)

    simplifying = {
        remainder
        for remainder in remainders
        for solution in parsimonious
        for implicant in solution.implicants
        if implicant.covers(remainder)
    }
    easy = {
        remainder
        for remainder in simplifying
        if is_easy_counterfactual(remainder, observed_sufficient, expectations, conditions)
    }

    return CounterfactualAnalysis(
        expectations=dict(expectations),
        simplifying_assumptions=frozenset(simplifying),
        easy=frozenset(easy),
        difficult=frozenset(simplifying - easy),
    )
