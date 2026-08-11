# Multi-value QCA

A multi-value condition takes one of several unordered categories — regime type,
welfare regime, sector — rather than being present or absent. Forcing such a
condition into a binary set either loses information or invents a dichotomy the
concept does not have.

```python
from setqca.multivalue import MVQCA

result = MVQCA(consistency=0.8).fit(data, outcome="Y", conditions=["regime", "wealth"])
print(result)
print(result.truth_table.to_frame())
print(result.summary_frame("parsimonious"))
```

Conditions hold integer category codes from `0`; the outcome is a membership in
`[0, 1]`. The workflow deliberately mirrors `FSQCA` and `CSQCA` — moving between
them is a change of estimator, not a change of method.

## Notation

`regime{0,2}*wealth{1}` reads "regime is 0 or 2, and wealth is 1". A condition
allowing *every* level constrains nothing and is omitted from the expression, so
the binary case reduces to familiar QCA notation.

## The property space

```python
result.domain  # regime{0,1,2}, wealth{0,1}
result.domain.size  # 6 logically possible configurations
```

Configurations are indexed in mixed radix, which generalises the binary minterm
and reduces to it exactly when every condition has two levels.

!!! warning "Declare levels that have no cases"
    Levels are inferred from the data, which understates a category that is
    theoretically possible but happens to be unobserved. That matters: an
    unobserved level is a *remainder*, and remainders change the parsimonious
    solution.

    ```python
    MVQCA(levels={"regime": 4, "wealth": 2}).fit(...)
    ```

    Declaring fewer levels than the data contain is an error.

## Why not Boolean dummies

The obvious shortcut is to encode `A{0,1,2}` as three binary indicators and
reuse the binary minimiser. **That transformation does not preserve the
semantics.**

The binary space contains points such as `A_0 = A_1 = 1` — a case that is
simultaneously in two mutually exclusive categories, which corresponds to no
configuration at all. The minimiser is free to build implicants across those
points, producing terms that look valid and describe nothing. Recovering a
multi-value expression afterwards requires exactly the mutual-exclusivity
constraints the encoding threw away.

So the cube algebra is implemented directly. A cube allows a **set** of levels
per condition, and merging generalises the binary rule:

> two cubes that agree on every condition but one merge into a single cube whose
> set at that condition is the union of the two.

Because the two cubes agree everywhere else, the merged cube covers exactly
their union and nothing more — the same property the binary rule relies on. A
test asserts precisely that, and another asserts every cube covers only real
configurations.

The exact cover is then solved by the **same verified solver the binary engine
uses**, so both inherit one exactness guarantee rather than two implementations.
Minimisation is checked against exhaustive enumeration for four different level
combinations, and against the binary minimiser for every three-condition
problem.

## Agreement with R

R `QCA` supports multi-value and writes literals as `regime[2]`. The truth table
and the parsimonious solution match exactly on the benchmarks in
`validation/fixtures/r_qca.json`.

The conservative solution can differ in *representation*:

```text
R:      regime[2] + regime[1]*wealth[1]
setqca: regime{2} + regime{1,2}*wealth{1}
```

Both cover the same configurations and both cost two terms and three literals,
so both are minimal. The difference is that R writes single-value literals only,
while `setqca` also forms subset literals — and here R's `regime[1]*wealth[1]`
is a **proper subset** of `regime{1,2}*wealth{1}`, so R's term is not a prime
implicant. The parity tests therefore compare cost and coverage rather than
text, which is the comparison that carries meaning.

::: setqca.multivalue
