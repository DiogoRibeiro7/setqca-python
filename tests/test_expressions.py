"""Tests for the configurational expression system."""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from setqca import Condition, SetExpression
from setqca.expressions import (
    Configuration,
    ExpressionSyntaxError,
    Implication,
    TokenKind,
    canonical,
    equivalent,
    evaluate_expression,
    format_expression,
    parse_expression,
    parse_set_expression,
    simplify,
    simplify_expression,
    tokenize,
)

DATA = pd.DataFrame(
    {
        "A": [0.9, 0.2, 0.5],
        "B": [0.8, 0.7, 0.1],
        "C": [0.3, 0.6, 0.9],
        "D": [0.4, 1.0, 0.0],
    }
)


class TestTokenizer:
    def test_operators_and_identifiers_are_recognised(self) -> None:
        kinds = [token.kind for token in tokenize("A*~B + C")]
        assert kinds == [
            TokenKind.IDENTIFIER,
            TokenKind.AND,
            TokenKind.NOT,
            TokenKind.IDENTIFIER,
            TokenKind.OR,
            TokenKind.IDENTIFIER,
            TokenKind.END,
        ]

    @pytest.mark.parametrize("text", ["A -> Y", "A => Y"])
    def test_both_arrow_spellings_are_accepted(self, text: str) -> None:
        assert any(token.kind is TokenKind.IMPLIES for token in tokenize(text))

    @pytest.mark.parametrize("negation", ["~A", "!A", "-A"])
    def test_all_negation_spellings_are_accepted(self, negation: str) -> None:
        assert str(parse_expression(negation)) == "~A"

    def test_whitespace_is_insignificant(self) -> None:
        assert str(parse_expression("  A  *  B  ")) == "A*B"

    def test_multi_character_condition_names_are_supported(self) -> None:
        assert str(parse_expression("urbanisation*~literacy")) == "urbanisation*~literacy"

    def test_an_unknown_character_is_reported_with_its_position(self) -> None:
        with pytest.raises(ExpressionSyntaxError, match="Unexpected") as excinfo:
            tokenize("A # B")
        assert excinfo.value.position == 2


class TestParser:
    def test_conjunction_binds_tighter_than_disjunction(self) -> None:
        node = parse_expression("A + B*C")
        assert format_expression(node) == "A+B*C"
        # Grouping is preserved, not merely the text.
        assert evaluate_expression(node, DATA) == pytest.approx([0.9, 0.6, 0.5])

    def test_parentheses_override_precedence(self) -> None:
        node = parse_expression("(A + B)*C")
        assert format_expression(node) == "(A+B)*C"
        assert evaluate_expression(node, DATA) == pytest.approx([0.3, 0.6, 0.5])

    def test_negation_binds_tighter_than_conjunction(self) -> None:
        assert str(parse_expression("~A*B")) == "~A*B"
        assert evaluate_expression("~A*B", DATA) == pytest.approx([0.1, 0.7, 0.1])

    def test_negation_of_a_group_is_parenthesised(self) -> None:
        node = parse_expression("~(A+B)")
        assert format_expression(node) == "~(A+B)"
        assert evaluate_expression(node, DATA) == pytest.approx([0.1, 0.3, 0.5])

    def test_implication_is_parsed_as_a_relation(self) -> None:
        node = parse_expression("A*B -> C")
        assert isinstance(node, Implication)
        assert str(node) == "A*B -> C"

    def test_an_implication_has_no_membership_of_its_own(self) -> None:
        with pytest.raises(ExpressionSyntaxError, match="implication"):
            parse_set_expression("A -> C")

    def test_implication_fit_is_computed_against_data(self) -> None:
        node = parse_expression("A -> B")
        assert isinstance(node, Implication)
        fit = node.evaluate_relation(DATA)
        assert 0.0 <= fit.consistency <= 1.0

    @pytest.mark.parametrize(
        "text",
        ["", "A*", "*A", "A B", "(A", "A)", "A + ", "~", "A -> B -> C", "A**B"],
    )
    def test_malformed_input_is_rejected(self, text: str) -> None:
        with pytest.raises(ExpressionSyntaxError):
            parse_expression(text)

    def test_nothing_in_the_input_is_evaluated_as_code(self) -> None:
        with pytest.raises(ExpressionSyntaxError):
            parse_expression("__import__('os').system('echo unsafe')")


class TestUnknownNodes:
    """An unrecognised node type must fail loudly rather than render wrongly."""

    class Custom(SetExpression):
        """A node the formatter and canonicaliser have never heard of."""

        def evaluate(self, data: pd.DataFrame) -> object:  # pragma: no cover - never called
            raise NotImplementedError

    def test_formatting_rejects_an_unknown_node(self) -> None:
        with pytest.raises(TypeError, match="Cannot format node"):
            format_expression(self.Custom())  # type: ignore[arg-type]

    def test_canonicalisation_rejects_an_unknown_node(self) -> None:
        with pytest.raises(TypeError, match="Cannot canonicalise node"):
            canonical(self.Custom())  # type: ignore[arg-type]


class TestFormatting:
    def test_a_union_inside_an_intersection_keeps_its_grouping(self) -> None:
        a, b, c = Condition("A"), Condition("B"), Condition("C")
        assert format_expression((a | b) & c) == "(A+B)*C"
        assert format_expression(a | (b & c)) == "A+B*C"

    def test_str_agrees_with_the_formatter(self) -> None:
        a, b, c = Condition("A"), Condition("B"), Condition("C")
        assert str((a | b) & c) == "(A+B)*C"


class TestCanonicalForm:
    def test_operand_order_does_not_affect_canonical_form(self) -> None:
        assert canonical(parse_set_expression("B*A")) == canonical(parse_set_expression("A*B"))

    def test_nested_same_operators_are_flattened(self) -> None:
        assert format_expression(canonical(parse_set_expression("A*(B*C)"))) == "A*B*C"

    def test_idempotence_is_applied(self) -> None:
        assert format_expression(canonical(parse_set_expression("A*A"))) == "A"
        assert format_expression(canonical(parse_set_expression("A+A"))) == "A"

    def test_double_negation_is_eliminated(self) -> None:
        assert format_expression(canonical(parse_set_expression("~~A"))) == "A"

    def test_absorption_is_applied(self) -> None:
        assert format_expression(simplify(parse_set_expression("A + A*B"))) == "A"
        assert format_expression(simplify(parse_set_expression("A*(A+B)"))) == "A"

    def test_equivalence_ignores_ordering_and_nesting(self) -> None:
        assert equivalent(parse_set_expression("A*B + C"), parse_set_expression("C + B*A"))

    def test_complement_laws_are_not_applied(self) -> None:
        """``A*~A`` is not empty and ``A+~A`` is not the universe for fuzzy sets."""
        assert format_expression(simplify(parse_set_expression("A*~A"))) == "A*~A"
        assert format_expression(simplify(parse_set_expression("A+~A"))) == "A+~A"
        # And the arithmetic confirms it: min(0.5, 0.5) = 0.5, not 0.
        assert evaluate_expression("A*~A", DATA) == pytest.approx([0.1, 0.2, 0.5])
        assert evaluate_expression("A+~A", DATA) == pytest.approx([0.9, 0.8, 0.5])

    def test_simplification_preserves_membership(self) -> None:
        original = "A + A*B + B*A"
        assert evaluate_expression(original, DATA) == pytest.approx(
            evaluate_expression(simplify_expression(original), DATA)
        )


class TestConfiguration:
    def test_a_configuration_renders_as_a_conjunction_of_literals(self) -> None:
        config = Configuration((("A", True), ("B", False), ("C", True)))
        assert str(config) == "A*~B*C"

    def test_minterm_encoding_is_big_endian(self) -> None:
        config = Configuration((("A", True), ("B", True), ("C", False)))
        assert config.minterm == 6

    def test_minterm_round_trips(self) -> None:
        conditions = ("A", "B", "C")
        for minterm in range(8):
            config = Configuration.from_minterm(minterm, conditions)
            assert config.minterm == minterm
            assert config.conditions == conditions

    def test_membership_matches_the_equivalent_expression(self) -> None:
        config = Configuration((("A", True), ("B", False)))
        assert config.evaluate(DATA) == pytest.approx(evaluate_expression("A*~B", DATA))

    def test_a_minterm_outside_the_domain_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside the truth-table domain"):
            Configuration.from_minterm(8, ("A", "B", "C"))

    def test_a_single_condition_configuration_is_a_bare_literal(self) -> None:
        assert str(Configuration((("A", True),))) == "A"
        assert str(Configuration((("A", False),))) == "~A"

    def test_an_empty_configuration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one condition"):
            Configuration(()).to_expression()
        with pytest.raises(ValueError, match="At least one condition"):
            Configuration.from_minterm(0, ())


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

_NAMES = st.sampled_from(["A", "B", "C", "D"])


@st.composite
def expressions(draw: st.DrawFn, depth: int = 3) -> str:
    """Draw a syntactically valid expression string."""
    if depth <= 0:
        return draw(_NAMES)
    choice = draw(st.integers(min_value=0, max_value=3))
    if choice == 0:
        return draw(_NAMES)
    if choice == 1:
        return f"~({draw(expressions(depth - 1))})"
    left = draw(expressions(depth - 1))
    right = draw(expressions(depth - 1))
    operator = "*" if choice == 2 else "+"
    return f"({left}){operator}({right})"


@given(text=expressions())
def test_parsing_round_trips_through_the_formatter(text: str) -> None:
    """Input -> tree -> string -> tree must stay semantically identical."""
    first = parse_set_expression(text)
    rendered = format_expression(first)
    second = parse_set_expression(rendered)

    assert format_expression(second) == rendered, "formatting must be a fixed point"
    assert canonical(first) == canonical(second)
    assert evaluate_expression(first, DATA) == pytest.approx(evaluate_expression(second, DATA))


@given(text=expressions())
def test_simplification_never_changes_membership(text: str) -> None:
    node = parse_set_expression(text)
    assert evaluate_expression(node, DATA) == pytest.approx(
        evaluate_expression(simplify(node), DATA)
    )
