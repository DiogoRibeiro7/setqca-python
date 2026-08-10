"""Tokenizer for QCA configurational expressions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TokenKind(Enum):
    """Lexical category of a token."""

    IDENTIFIER = "identifier"
    NOT = "not"
    AND = "and"
    OR = "or"
    IMPLIES = "implies"
    LPAREN = "lparen"
    RPAREN = "rparen"
    END = "end"


@dataclass(frozen=True, slots=True)
class Token:
    """A lexical token and where it started in the source text."""

    kind: TokenKind
    text: str
    position: int


class ExpressionSyntaxError(ValueError):
    """Raised when an expression cannot be tokenized or parsed.

    The message carries the offending position so the caller can point at it.
    """

    def __init__(self, message: str, *, expression: str, position: int) -> None:
        self.expression = expression
        self.position = position
        caret = " " * position + "^"
        super().__init__(f"{message}\n  {expression}\n  {caret}")


# Condition names follow Python identifier rules, which covers the uppercase
# single letters of the QCA literature and longer descriptive names alike.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Negation is written `~` or `-` in the QCA literature; lowercase-means-absent
# is deliberately not supported, because it makes case-sensitive condition
# names ambiguous.
_SIMPLE: dict[str, TokenKind] = {
    "~": TokenKind.NOT,
    "!": TokenKind.NOT,
    "-": TokenKind.NOT,
    "*": TokenKind.AND,
    "+": TokenKind.OR,
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
}


def tokenize(expression: str) -> list[Token]:
    """Split an expression into tokens.

    Parameters
    ----------
    expression : str
        Source text, for example ``"A*~B + C"`` or ``"A*B -> Y"``.

    Returns
    -------
    list of Token
        Tokens terminated by a single :attr:`TokenKind.END`.

    Raises
    ------
    ExpressionSyntaxError
        If the text contains a character that cannot begin a token.
    """
    tokens: list[Token] = []
    index = 0
    length = len(expression)

    while index < length:
        char = expression[index]

        if char.isspace():
            index += 1
            continue

        if expression.startswith("->", index):
            tokens.append(Token(TokenKind.IMPLIES, "->", index))
            index += 2
            continue

        if expression.startswith("=>", index):
            tokens.append(Token(TokenKind.IMPLIES, "=>", index))
            index += 2
            continue

        match = _IDENTIFIER.match(expression, index)
        if match is not None:
            tokens.append(Token(TokenKind.IDENTIFIER, match.group(), index))
            index = match.end()
            continue

        kind = _SIMPLE.get(char)
        if kind is not None:
            tokens.append(Token(kind, char, index))
            index += 1
            continue

        raise ExpressionSyntaxError(
            f"Unexpected character {char!r}", expression=expression, position=index
        )

    tokens.append(Token(TokenKind.END, "", length))
    return tokens
