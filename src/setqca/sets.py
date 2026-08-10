"""Typed set expressions for calibrated QCA conditions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np

from ._validation import FloatArray, validate_membership

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    import pandas as pd


class SetExpression(ABC):
    """Abstract fuzzy-set expression over calibrated conditions.

    Expressions compose with the standard Python operators ``&`` (intersection,
    minimum t-norm), ``|`` (union, maximum s-norm) and ``~`` (negation).
    """

    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> FloatArray:
        """Evaluate membership of the expression for every case.

        Parameters
        ----------
        data : pandas.DataFrame
            Frame of calibrated condition memberships.

        Returns
        -------
        FloatArray
            Membership of each case in the expression.
        """

    def __and__(self, other: SetExpression) -> SetExpression:
        return Intersection((self, other))

    def __or__(self, other: SetExpression) -> SetExpression:
        return Union((self, other))

    def __invert__(self) -> SetExpression:
        return Negation(self)


@dataclass(frozen=True, slots=True)
class Condition(SetExpression):
    """Named calibrated condition drawn from a column of the data."""

    name: str

    def evaluate(self, data: pd.DataFrame) -> FloatArray:
        """Return the calibrated membership column for this condition."""
        if self.name not in data.columns:
            raise KeyError(f"Missing condition column: {self.name}")
        return validate_membership(data[self.name].to_numpy(), name=self.name)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Negation(SetExpression):
    """Fuzzy-set negation using ``1 - membership``."""

    operand: SetExpression

    def evaluate(self, data: pd.DataFrame) -> FloatArray:
        """Return one minus the membership of the negated operand."""
        return 1.0 - self.operand.evaluate(data)

    def __str__(self) -> str:
        from .expressions import format_expression

        return format_expression(self)


@dataclass(frozen=True, slots=True)
class Intersection(SetExpression):
    """Fuzzy conjunction using the minimum t-norm."""

    operands: tuple[SetExpression, ...]

    def evaluate(self, data: pd.DataFrame) -> FloatArray:
        """Return the elementwise minimum across all operands."""
        if not self.operands:
            raise ValueError("Intersection requires at least one operand.")
        arrays = (operand.evaluate(data) for operand in self.operands)
        return reduce(np.minimum, arrays)

    def __str__(self) -> str:
        from .expressions import format_expression

        return format_expression(self)


@dataclass(frozen=True, slots=True)
class Union(SetExpression):
    """Fuzzy disjunction using the maximum s-norm."""

    operands: tuple[SetExpression, ...]

    def evaluate(self, data: pd.DataFrame) -> FloatArray:
        """Return the elementwise maximum across all operands."""
        if not self.operands:
            raise ValueError("Union requires at least one operand.")
        arrays = (operand.evaluate(data) for operand in self.operands)
        return reduce(np.maximum, arrays)

    def __str__(self) -> str:
        from .expressions import format_expression

        return format_expression(self)
