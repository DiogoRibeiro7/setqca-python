# Sufficiency diagnostics

Parameters of fit summarise a solution in a few numbers. They do not say which
cases produced those numbers — and that is usually the question you actually
have. Which cases support this path? Which contradict it? Which outcomes does it
fail to explain?

```python
from setqca import sufficiency_diagnostics

diagnostics = sufficiency_diagnostics(
    data,
    outcome="SURV",
    terms=["DEV*URB*LIT*IND*STB", "DEV*~URB*LIT*~IND*STB"],
)
print(diagnostics)
print(diagnostics.to_frame())
print(diagnostics.cases_frame())
```

Terms are given as expression strings and parsed, so a solution can be pasted
straight in. Case labels come from the frame index by default, or from a column
you name — no particular schema is assumed.

## The case typology

For a term `X` and outcome `Y`, with the crossover at 0.5:

| Membership | Role | What it means |
| --- | --- | --- |
| `X > 0.5`, `Y > 0.5`, `X ≤ Y` | **typical** | Supports the claim. These are the cases to study for the mechanism. |
| `X > 0.5`, `Y > 0.5`, `X > Y` | **deviant consistency (degree)** | Right corner, wrong magnitude — more in the term than in the outcome. |
| `X > 0.5`, `Y ≤ 0.5` | **deviant consistency (kind)** | The term holds and the outcome does not. This is the case-level contradiction. |
| `X ≤ 0.5`, `Y > 0.5` | **deviant coverage** | An outcome this term does not explain. |
| `X ≤ 0.5`, `Y ≤ 0.5` | **individually irrelevant** | Outside both sets. |

```python
term = diagnostics.terms[0]
term.typical  # ('BE', 'CZ', 'NL')
term.contradictory  # cases where the term holds but the outcome does not
term.deviant_coverage  # outcomes this term misses
term.deviant_consistency  # both kinds of consistency deviance
term.uniquely_covered  # cases no other term reaches
```

!!! note "Only consistency deviance counts against the claim"
    A deviant-coverage case is not evidence against sufficiency. It says the
    outcome occurred through some other path, which is exactly what a
    disjunctive solution expects. `CaseRole.contradicts_sufficiency` encodes
    the distinction.

## Unique coverage

Raw coverage counts the outcome membership a term accounts for. **Unique**
coverage counts only what no other term accounts for:

```text
covU_i = [ Σ min(Xᵢ, Y) − Σ min(Xᵢ, max_{j≠i} Xⱼ, Y) ] / Σ Y
```

A term with substantial raw coverage but near-zero unique coverage is redundant
in practice — drop it and the same cases are still explained:

```python
diagnostics.redundant_terms
```

Two identical terms each have unique coverage of exactly zero, which is the
degenerate case the property makes obvious.

!!! info "A small divergence from R"
    R reports `covU` as `NA` for a single-term solution, since there is no other
    term to be unique against. `setqca` reports the raw coverage instead: with
    nothing to share with, everything the term covers is uniquely covered by it.
    Verified against R for every multi-term solution on the Lipset data.

## Reading R's `cases` column

R's per-term `cases` column lists cases whose membership in the term exceeds the
crossover. The typology splits that same set further, so R's list corresponds to
**typical plus deviant-in-degree**, not to typical alone.

On the Lipset conservative solution R lists `BE, CZ, NL, UK` for the first term.
`setqca` agrees on all four being in the term, and additionally reports that UK
is deviant in degree — its membership in the term exceeds its membership in the
outcome. That distinction is the point of the typology, and it is not visible
from the `cases` column alone.

## Choosing cases to study

The typology exists to support case selection in multi-method work:

- **Typical** cases are where the proposed mechanism should be visible.
- **Deviant consistency** cases are where it should be visible and is not — the
  most informative cases for revising the theory.
- **Deviant coverage** cases point at paths the solution is missing.
- **Uniquely covered** cases are the ones that justify keeping a term at all.

::: setqca.analysis.sufficiency
