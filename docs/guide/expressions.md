# Configurational expressions

QCA notation — `A*~B + C`, `A*B -> Y` — is parsed into a typed tree rather than
manipulated as text. Expressions are therefore comparable, simplifiable and
evaluable, and solutions come back as structured objects instead of strings.

## Parsing

```python
from setqca import parse_expression

node = parse_expression("A*~B + C")
```

| Notation | Meaning | Alternatives |
| --- | --- | --- |
| `*` | conjunction, minimum | |
| `+` | disjunction, maximum | |
| `~A` | negation, `1 - A` | `!A`, `-A` |
| `->` | implication (sufficiency claim) | `=>` |
| `( )` | grouping | |

Condition names follow Python identifier rules, so both the uppercase single
letters of the literature and longer descriptive names work.

!!! note "Nothing is evaluated as code"
    Parsing is structural — a tokenizer and a recursive-descent parser. An
    expression taken from a configuration file or from user input cannot execute
    anything. There is no `eval` anywhere in this package.

Malformed input raises `ExpressionSyntaxError`, which points at the position:

```python
>>> parse_expression("A * * B")
ExpressionSyntaxError: Expected a condition name, found '*'
  A * * B
      ^
```

## Precedence

Conjunction binds more tightly than disjunction, and negation more tightly
still, so `A + B*C` means `A + (B*C)`. Parentheses override this, and the
printer re-inserts them wherever grouping would otherwise be lost:

```python
>>> from setqca.expressions import format_expression
>>> format_expression(parse_expression("(A + B)*C"))
'(A+B)*C'
>>> format_expression(parse_expression("A + B*C"))
'A+B*C'
```

Parsing and printing round-trip: `text → tree → text → tree` gives back a
semantically identical tree, which is property-tested.

## Evaluation

```python
from setqca import evaluate_expression

membership = evaluate_expression("A*~B", data)
```

Fuzzy operators are the standard ones — minimum, maximum and `1 - x`.

An implication has no membership of its own, because it is a *relation* between
two sets rather than a set. Evaluate it as one:

```python
claim = parse_expression("A*B -> Y")
fit = claim.evaluate_relation(data)
print(fit.consistency, fit.coverage, fit.pri)
```

Asking for the membership of an implication is an error rather than a silent
guess.

## Simplification

```python
from setqca import simplify_expression
from setqca.expressions import format_expression

format_expression(simplify_expression("A + A*B"))  # 'A'
```

Applied: associativity, commutativity, idempotence (`A*A = A`), double negation
(`~~A = A`) and absorption (`A + A*B = A`).

!!! danger "The complement laws do not hold"
    In Boolean algebra `A*~A` is empty and `A+~A` is the universe. **Neither is
    true for fuzzy sets.** With `A = 0.5`, `min(A, 1-A) = 0.5` and
    `max(A, 1-A) = 0.5` — a case can be half in a set and half in its negation
    at the same time.

    `setqca` therefore never simplifies those away:

    ```python
    format_expression(simplify_expression("A*~A"))  # 'A*~A', not '0'
    ```

    This is the single most common way a Boolean-minded simplifier corrupts a
    fuzzy analysis. Every simplification here is verified to leave membership
    unchanged on real data.

## Comparing expressions

Two expressions that differ only by ordering or nesting are equal after
canonicalisation:

```python
from setqca.expressions import equivalent

equivalent(parse_expression("A*B + C"), parse_expression("C + B*A"))  # True
```

Note this is *structural* equivalence under the laws above, not semantic
equivalence over all possible data. Deciding the latter for fuzzy sets is a
different and much harder question.

## Configurations

A `Configuration` is one corner of the property space — a state for every
condition — and converts to and from a minterm index:

```python
from setqca.expressions import Configuration

config = Configuration.from_minterm(6, ("A", "B", "C"))
str(config)  # 'A*B*~C'
config.minterm  # 6
config.evaluate(data)
```

Minterm indices are big-endian over the condition order, matching the truth
table and the minimiser.

::: setqca.expressions
