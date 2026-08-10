"""Expression nodes layered on the set algebra, plus canonical form.

The set-algebra classes in :mod:`setqca.sets` are the expression nodes; this
module adds the nodes that have no membership of their own (implications and
configurations) and the canonicalisation used for structural comparison.

Only laws valid in the standard fuzzy algebra are applied. In particular the
complement laws of Boolean algebra do **not** hold: ``min(A, 1-A)`` is not the
empty set and ``max(A, 1-A)`` is not the universe, so ``A*~A`` and ``A+~A``
are never simplified away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from setqca.metrics import SufficiencyFit, sufficiency
from setqca.sets import Condition, Intersection, Negation, SetExpression, Union

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    import pandas as pd

    from setqca._validation import FloatArray

# Names used by the configurational literature for the two n-ary operators.
Conjunction = Intersection
Disjunction = Union

# Binding power, loosest first. Used by the pretty printer to decide where
# parentheses are required to preserve grouping.
_PRECEDENCE: dict[type, int] = {Union: 1, Intersection: 2, Negation: 3, Condition: 4}


def precedence(node: SetExpression) -> int:
    """Return the binding power of a node, higher binding more tightly."""
    return _PRECEDENCE.get(type(node), 4)


def format_expression(node: SetExpression) -> str:
    """Render a node, parenthesising only where grouping would otherwise be lost.

    Parameters
    ----------
    node : SetExpression
        Expression to render.

    Returns
    -------
    str
        Standard QCA notation, for example ``"(A+B)*~C"``.

    Examples
    --------
    >>> from setqca import Condition
    >>> a, b, c = Condition("A"), Condition("B"), Condition("C")
    >>> format_expression((a | b) & c)
    '(A+B)*C'
    >>> format_expression(a | (b & c))
    'A+B*C'
    """
    if isinstance(node, Condition):
        return node.name
    if isinstance(node, Negation):
        inner = node.operand
        text = format_expression(inner)
        # `~` binds tighter than `*` and `+`, so a compound operand needs bracketing.
        return f"~({text})" if precedence(inner) < precedence(node) else f"~{text}"
    if isinstance(node, Intersection | Union):
        separator = "*" if isinstance(node, Intersection) else "+"
        parts: list[str] = []
        for operand in node.operands:
            text = format_expression(operand)
            if precedence(operand) < precedence(node):
                text = f"({text})"
            parts.append(text)
        return separator.join(parts)
    raise TypeError(f"Cannot format node of type {type(node).__name__}.")


def _flatten(node: SetExpression) -> tuple[SetExpression, ...]:
    """Return the operands of an associative node, flattening nested same-type nodes."""
    assert isinstance(node, Intersection | Union)
    flat: list[SetExpression] = []
    for operand in node.operands:
        if type(operand) is type(node):
            flat.extend(_flatten(operand))
        else:
            flat.append(operand)
    return tuple(flat)


def canonical(node: SetExpression) -> SetExpression:
    """Return a structurally canonical form of an expression.

    Associativity, commutativity, idempotence and double negation are applied,
    so two expressions that differ only by those laws canonicalise to the same
    object and therefore compare equal.

    The complement laws are deliberately not applied; see the module docstring.
    """
    if isinstance(node, Condition):
        return node
    if isinstance(node, Negation):
        inner = canonical(node.operand)
        # ~~A = A holds in the fuzzy algebra because 1 - (1 - x) = x.
        if isinstance(inner, Negation):
            return inner.operand
        return Negation(inner)
    if isinstance(node, Intersection | Union):
        operands = [canonical(operand) for operand in _flatten(node)]
        unique: list[SetExpression] = []
        seen: set[str] = set()
        for operand in sorted(operands, key=format_expression):
            key = format_expression(operand)
            if key not in seen:  # idempotence: A*A = A and A+A = A
                seen.add(key)
                unique.append(operand)
        if len(unique) == 1:
            return unique[0]
        return type(node)(tuple(unique))
    raise TypeError(f"Cannot canonicalise node of type {type(node).__name__}.")


def _absorb(node: SetExpression) -> SetExpression:
    """Apply the absorption laws, which hold for min/max fuzzy operators.

    ``A + A*B = A`` and ``A * (A+B) = A``.
    """
    if not isinstance(node, Intersection | Union):
        return node

    dual = Intersection if isinstance(node, Union) else Union
    operands = list(node.operands)
    keep: list[SetExpression] = []

    for candidate in operands:
        others = [item for item in operands if item is not candidate]
        absorbed = False
        if isinstance(candidate, dual):
            # `candidate` is absorbed when one of its own operands stands alone.
            parts = {format_expression(part) for part in candidate.operands}
            absorbed = any(format_expression(other) in parts for other in others)
        if not absorbed:
            keep.append(candidate)

    if len(keep) == 1:
        return keep[0]
    if len(keep) != len(operands):
        return type(node)(tuple(keep))
    return node


def simplify(node: SetExpression) -> SetExpression:
    """Simplify an expression using only laws valid for fuzzy sets.

    Applies flattening, commutative ordering, idempotence, double negation and
    absorption. Never applies the complement laws, which are false for the
    minimum/maximum operators.

    Parameters
    ----------
    node : SetExpression
        Expression to simplify.

    Returns
    -------
    SetExpression
        A semantically identical expression, in canonical order.

    Examples
    --------
    >>> from setqca import Condition
    >>> from setqca.expressions import format_expression, simplify
    >>> a, b = Condition("A"), Condition("B")
    >>> format_expression(simplify(a | (a & b)))
    'A'
    """
    current = canonical(node)
    while True:
        reduced = canonical(_absorb(current))
        if format_expression(reduced) == format_expression(current):
            return reduced
        current = reduced


def equivalent(left: SetExpression, right: SetExpression) -> bool:
    """Return whether two expressions are equal after simplification."""
    return format_expression(simplify(left)) == format_expression(simplify(right))


@dataclass(frozen=True, slots=True)
class Implication:
    """A sufficiency claim ``antecedent -> consequent``.

    An implication has no membership of its own: it is a relation between two
    sets, evaluated as a set-theoretic subset relation rather than as a
    membership vector.
    """

    antecedent: SetExpression
    consequent: SetExpression

    def evaluate_relation(self, data: pd.DataFrame) -> SufficiencyFit:
        """Return the parameters of fit for the claim against the data."""
        return sufficiency(
            self.antecedent.evaluate(data),
            self.consequent.evaluate(data),
        )

    def __str__(self) -> str:
        return f"{format_expression(self.antecedent)} -> {format_expression(self.consequent)}"


@dataclass(frozen=True, slots=True)
class Configuration:
    """One corner of the property space: a state for every condition.

    Parameters
    ----------
    states : tuple of (str, bool)
        Condition name and whether it is present, in minterm order.
    """

    states: tuple[tuple[str, bool], ...]

    @property
    def conditions(self) -> tuple[str, ...]:
        """Return the condition names in order."""
        return tuple(name for name, _ in self.states)

    @property
    def minterm(self) -> int:
        """Return the big-endian minterm index of this corner."""
        value = 0
        for _, present in self.states:
            value = (value << 1) | int(present)
        return value

    def to_expression(self) -> SetExpression:
        """Return the conjunction of literals describing this corner."""
        if not self.states:
            raise ValueError("A configuration requires at least one condition.")
        literals: list[SetExpression] = [
            Condition(name) if present else Negation(Condition(name))
            for name, present in self.states
        ]
        if len(literals) == 1:
            return literals[0]
        return Intersection(tuple(literals))

    def evaluate(self, data: pd.DataFrame) -> FloatArray:
        """Return membership in this corner for every case."""
        return self.to_expression().evaluate(data)

    @classmethod
    def from_minterm(cls, minterm: int, conditions: tuple[str, ...]) -> Configuration:
        """Build a configuration from a big-endian minterm index."""
        width = len(conditions)
        if width == 0:
            raise ValueError("At least one condition is required.")
        if not 0 <= minterm < 2**width:
            raise ValueError("minterm is outside the truth-table domain.")
        bits = [(minterm >> shift) & 1 for shift in reversed(range(width))]
        return cls(tuple((name, bool(bit)) for name, bit in zip(conditions, bits, strict=True)))

    def __str__(self) -> str:
        return format_expression(self.to_expression())
