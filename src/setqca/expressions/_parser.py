"""Recursive-descent parser for QCA configurational expressions.

Grammar, loosest binding first::

    implication := disjunction ( "->" disjunction )?
    disjunction := conjunction ( "+" conjunction )*
    conjunction := unary ( "*" unary )*
    unary       := ( "~" | "!" | "-" ) unary | primary
    primary     := IDENTIFIER | "(" disjunction ")"

Parsing is structural: no part of the input is ever evaluated as code.
"""

from __future__ import annotations

from setqca.sets import Condition, Intersection, Negation, SetExpression, Union

from ._ast import Implication
from ._tokenizer import ExpressionSyntaxError, Token, TokenKind, tokenize


class _Parser:
    """Single-use recursive-descent parser over a token list."""

    def __init__(self, expression: str, tokens: list[Token]) -> None:
        self._expression = expression
        self._tokens = tokens
        self._index = 0

    @property
    def _current(self) -> Token:
        return self._tokens[self._index]

    def _at(self, kind: TokenKind) -> bool:
        """Return whether the cursor sits on a token of this kind.

        Going through a method rather than comparing the property directly
        keeps the type checker from narrowing the token kind across the
        recursive calls that move the cursor.
        """
        return self._tokens[self._index].kind is kind

    def _advance(self) -> Token:
        token = self._tokens[self._index]
        self._index += 1
        return token

    def _fail(self, message: str) -> ExpressionSyntaxError:
        return ExpressionSyntaxError(
            message, expression=self._expression, position=self._current.position
        )

    def parse(self) -> SetExpression | Implication:
        """Parse the whole token stream and require that it is fully consumed."""
        node = self._implication()
        if not self._at(TokenKind.END):
            raise self._fail(f"Unexpected {self._current.text!r} after a complete expression")
        return node

    def _implication(self) -> SetExpression | Implication:
        left = self._disjunction()
        if self._at(TokenKind.IMPLIES):
            self._advance()
            right = self._disjunction()
            if self._at(TokenKind.IMPLIES):
                raise self._fail("Implications do not chain; use one '->' per expression")
            return Implication(left, right)
        return left

    def _disjunction(self) -> SetExpression:
        terms = [self._conjunction()]
        while self._at(TokenKind.OR):
            self._advance()
            terms.append(self._conjunction())
        if len(terms) == 1:
            return terms[0]
        return Union(tuple(terms))

    def _conjunction(self) -> SetExpression:
        factors = [self._unary()]
        while self._at(TokenKind.AND):
            self._advance()
            factors.append(self._unary())
        if len(factors) == 1:
            return factors[0]
        return Intersection(tuple(factors))

    def _unary(self) -> SetExpression:
        if self._at(TokenKind.NOT):
            self._advance()
            return Negation(self._unary())
        return self._primary()

    def _primary(self) -> SetExpression:
        token = self._current
        if token.kind is TokenKind.IDENTIFIER:
            self._advance()
            return Condition(token.text)
        if token.kind is TokenKind.LPAREN:
            self._advance()
            inner = self._disjunction()
            if not self._at(TokenKind.RPAREN):
                raise self._fail("Expected a closing ')'")
            self._advance()
            return inner
        if token.kind is TokenKind.END:
            raise self._fail("Expression ended unexpectedly")
        raise self._fail(f"Expected a condition name, found {token.text!r}")


def parse_expression(expression: str) -> SetExpression | Implication:
    """Parse a configurational expression into a typed tree.

    Parameters
    ----------
    expression : str
        Standard QCA notation. ``*`` is conjunction, ``+`` disjunction, ``~``
        (or ``!``/``-``) negation, and ``->`` (or ``=>``) implication.
        Parentheses group.

    Returns
    -------
    SetExpression or Implication
        An :class:`~setqca.expressions.Implication` when the text contains an
        arrow, otherwise a set expression.

    Raises
    ------
    ExpressionSyntaxError
        If the text is not a well-formed expression. The message includes the
        offending position.

    Examples
    --------
    >>> from setqca.expressions import parse_expression
    >>> str(parse_expression("A*~B + C"))
    'A*~B+C'
    >>> str(parse_expression("A*B -> Y"))
    'A*B -> Y'
    """
    return _Parser(expression, tokenize(expression)).parse()


def parse_set_expression(expression: str) -> SetExpression:
    """Parse an expression that must not be an implication.

    Use this when the caller needs a membership-valued expression and a
    relation would be a mistake rather than a variant.

    Raises
    ------
    ExpressionSyntaxError
        If the text is malformed, or is an implication.
    """
    node = parse_expression(expression)
    if isinstance(node, Implication):
        raise ExpressionSyntaxError(
            "Expected a set expression but found an implication",
            expression=expression,
            position=expression.find("->") if "->" in expression else 0,
        )
    return node
