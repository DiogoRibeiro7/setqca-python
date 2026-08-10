"""Typed configurational expressions: parsing, canonical form and evaluation.

Expressions are parsed structurally into a typed tree. Nothing in the input is
ever evaluated as code, so an expression from a configuration file or a user
prompt cannot execute anything.

Examples
--------
>>> import pandas as pd
>>> from setqca.expressions import evaluate_expression, parse_expression
>>> data = pd.DataFrame({"A": [0.9, 0.2], "B": [0.8, 0.7]})
>>> evaluate_expression("A*~B", data).round(2)
array([0.2, 0.2])
>>> str(parse_expression("A*B -> Y"))
'A*B -> Y'
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from setqca.sets import Condition, Intersection, Negation, SetExpression, Union

from ._ast import (
    Configuration,
    Conjunction,
    Disjunction,
    Implication,
    canonical,
    equivalent,
    format_expression,
    precedence,
    simplify,
)
from ._parser import parse_expression, parse_set_expression
from ._tokenizer import ExpressionSyntaxError, Token, TokenKind, tokenize

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    import pandas as pd

    from setqca._validation import FloatArray

__all__ = [
    "Condition",
    "Configuration",
    "Conjunction",
    "Disjunction",
    "ExpressionSyntaxError",
    "Implication",
    "Intersection",
    "Negation",
    "SetExpression",
    "Token",
    "TokenKind",
    "Union",
    "canonical",
    "equivalent",
    "evaluate_expression",
    "format_expression",
    "parse_expression",
    "parse_set_expression",
    "precedence",
    "simplify",
    "simplify_expression",
    "tokenize",
]


def evaluate_expression(expression: str | SetExpression, data: pd.DataFrame) -> FloatArray:
    """Evaluate an expression against calibrated data.

    Parameters
    ----------
    expression : str or SetExpression
        Expression text, or an already-parsed tree.
    data : pandas.DataFrame
        Calibrated condition memberships in ``[0, 1]``.

    Returns
    -------
    FloatArray
        Membership of every case in the expression.

    Raises
    ------
    ExpressionSyntaxError
        If the text is malformed or is an implication, which has no membership
        of its own. Use :meth:`Implication.evaluate_relation` for those.
    """
    node = parse_set_expression(expression) if isinstance(expression, str) else expression
    return node.evaluate(data)


def simplify_expression(expression: str | SetExpression) -> SetExpression:
    """Parse if needed, then simplify using only fuzzy-valid laws.

    Parameters
    ----------
    expression : str or SetExpression
        Expression text, or an already-parsed tree.

    Returns
    -------
    SetExpression
        A semantically identical expression in canonical order.
    """
    node = parse_set_expression(expression) if isinstance(expression, str) else expression
    return simplify(node)
