# Boolean minimisation

The minimiser reduces the set of sufficient truth-table rows to the smallest
equivalent Boolean expression. `setqca` solves this **exactly**: the result is a
proven minimum, not a good-enough approximation.

## The public engine

The engine is public so it can be tested and reused independently of the QCA
layer.

```python
from setqca.minimize import minimize

# AB~C + ABC -> AB
solutions = minimize({6, 7}, width=3)
print(solutions[0].as_expression(("A", "B", "C")))
# A*B
```

Logical remainders are supplied as explicit don't-cares:

```python
solutions = minimize({6, 7}, dont_cares={4, 5}, width=3)
print(solutions[0].as_expression(("A", "B", "C")))
# A
```

## The algorithm

1. **Cube generation.** Each minterm in the on-set and don't-care set becomes a
   fully specified cube.
2. **Iterative combination.** Cubes differing in exactly one fixed literal merge,
   replacing that literal with a don't-care. Cubes that never merge are prime.
3. **Pruning.** Primes built only from don't-cares are discarded: they cannot
   cover any row that actually needs covering.
4. **Chart solving.** The prime-implicant chart is solved by branch and bound,
   always branching on the hardest uncovered minterm first.
5. **Lexicographic optimisation.** Covers are ranked first by number of prime
   implicants, then by total literal count.

Steps 1–3 are classical Quine-McCluskey. Step 4 is what makes the result exact:
the greedy "pick the largest prime" heuristic used by many implementations can
miss the true minimum.

## Model ambiguity

A minimisation problem often has several distinct covers of identical minimal
cost. This is *model ambiguity*, and it is a property of the data, not a defect.

`minimize` returns **all** tied minimal covers, up to `max_solutions`:

```python
solutions = minimize(on_set, dont_cares=remainders, width=4, max_solutions=256)
for solution in solutions:
    print(solution.as_expression(conditions))
```

Reporting only the first solution when several exist misrepresents the evidence.
`QCAResult.summary_frame()` returns one row per solution precisely so ambiguity
stays visible.

## Complexity

Exact minimisation is worst-case exponential in the number of conditions, and no
implementation escapes that. What matters in practice is not the number of
conditions alone but the *shape* of the chart: how many prime implicants there
are, and how much they overlap.

The figures below come from `benchmarks/profile_phases.py` on ordinary hardware,
for a typical QCA design: 40 observed cases, an outcome that genuinely depends on
the conditions, and every unobserved row treated as a logical remainder — that
is, the **parsimonious** solution, which is the more expensive of the two
standard families because it hands the solver a large don't-care set.

| Conditions | Truth-table rows | Remainders | Parsimonious solution |
| --- | --- | --- | --- |
| 6 | 64 | 36 | 0.004 s |
| 7 | 128 | 95 | 0.009 s |
| 8 | 256 | 218 | 0.034 s |
| 9 | 512 | 473 | 0.113 s |
| 10 | 1 024 | 984 | 0.448 s |

Roughly a trebling per additional condition in this regime, where the cost is
carried by prime generation and chart construction rather than by the exponential
search. Conservative solutions are cheaper still at the same width, since they
use no don't-cares.

Dense tables — where a large fraction of *all* minterms is sufficient — are the
worst case, because they are the regime where the exact search itself dominates.
At seven conditions, timing the cover phase alone:

| Sufficient share | Positives | Primes | Cover solving |
| --- | --- | --- | --- |
| 0.10 | 11 | 10 | <0.0001 s |
| 0.25 | 33 | 26 | 0.0001 s |
| 0.50 | 65 | 53 | 0.0024 s |
| 0.75 | 91 | 78 | 0.60 s |

Do not extrapolate from either table to your own data; the chart shape matters
more than the width. If a run does not finish, the practical levers are reducing
the number of conditions — which is good QCA practice anyway — or lowering
`max_solutions` when the model is highly ambiguous.

This cost is the price of exactness. A faster minimiser (CCubes/eQMC-style) is a
roadmap item, but it will be added as an alternative engine rather than by
weakening the guarantee of the current one.

## Being told before it gets slow

Exact minimisation is worst-case exponential, and that cost is not negotiable
here: no heuristic is substituted when a problem gets hard, because a silently
approximate answer is worse than a slow one. What *is* negotiable is finding out
in advance.

Once the primes are generated — which is fast — the chart's shape is known, and
`minimize` warns before entering the exponential phase:

```python
>>> minimize(large_on_set, width=12)
MinimizationComplexityWarning: Exact minimisation of 400 configurations over 12
conditions produced 180 prime implicants (72000 chart cells). Solving the chart
exactly is worst-case exponential and may take a long time. The result will
still be exact. To reduce the cost, use fewer conditions, tighten the
consistency cutoff so fewer configurations qualify, or lower max_solutions if
the model is highly ambiguous.
```

The run still completes, and still returns an exact answer. Silence it with
`warnings.simplefilter`, or switch the check off with `complexity_guard=False`.

The trigger is the **number of prime implicants**, not the number of conditions:
the density table above is the evidence, and the threshold sits either side of
the climb between 53 and 78 primes. That climb is driven by how much the primes
*overlap*, which is why this is a warning rather than a prediction — a chart with
many primes and little overlap solves instantly.

## Where the time goes

`benchmarks/profile_phases.py` times each phase separately across four
dimensions — cases, conditions, sufficient configurations and remainders:

```bash
python benchmarks/profile_phases.py
python benchmarks/profile_phases.py --max-width 9 --markdown
```

Two different phases dominate in two different regimes:

- **Remainder-heavy problems** — the parsimonious case, where most of the
  property space is unobserved — are dominated by prime generation.
- **Dense on-sets** are dominated by solving the chart.

Truth-table construction barely moves with the number of cases — 25× the cases
costs under 2× the time, because the work is proportional to the property space,
not to the sample. Adding cases is cheap; adding *conditions* is not.

!!! note "Measure before optimising"
    Prime generation was originally the bottleneck at eight conditions, taking
    1.4 s. Rewriting it over integer masks rather than tuples of optional bits
    brought that to 0.010 s — the same algorithm, a different representation.
    The exactness and R-parity tests passed unchanged, which is what made the
    rewrite safe to keep.

## How the search stays tractable

Three reductions cut the search space without ever changing the answer:

1. **Essential prime implicants.** If a minterm is covered by exactly one prime,
   every possible cover must contain that prime. Essentials are selected up
   front instead of being rediscovered on every branch.
2. **An independent-set lower bound.** If several uncovered minterms have
   pairwise disjoint sets of candidate primes, each needs its own further prime.
   That count bounds the cost of any completion from below, so branches that
   cannot possibly reach the incumbent cost are abandoned early.
3. **State memoisation.** The covers reachable from a partial solution depend
   only on which minterms remain uncovered. Reaching the same remaining set at a
   strictly worse cost can therefore never produce a better or tied result.

Each is a standard result about prime-implicant charts, and each preserves both
minimality and the completeness of the returned tie set.

## Inspecting the chart

`minimize` reports its answer. `minimize_chart` reports its reasoning:

```python
from setqca import minimize_chart

result = minimize_chart(on_set, dont_cares=remainders, width=3)
print(result.summary(("A", "B", "C")))
```

```text
Configurations to cover: 6
Prime implicants: 6
Essential primes: 0
Dominated primes: 0
Minimum cost: 3 implicants, 6 literals
Minimum covers: 2
  ~A*~B + ~A*C + A*B
  ~A*~C + B*C + A*~B
```

Three questions the chart answers that a bare solution cannot:

**Why is this term in the solution?**

```python
print(result.covers[0].explain(("A", "B", "C")))
```

Each term is labelled either *essential* — the only prime covering some
configuration, so every possible solution contains it — or *selected among
interchangeable alternatives*.

**How could this configuration have been covered?**

```python
result.chart.explain(6, ("A", "B", "C"))
# 'Row 6 can be covered by any of: A*B, B*C.'
```

**Why did a plausible-looking term never appear?**

`result.chart.dominated` lists primes that another prime covers entirely at no
greater literal cost. They cannot appear in any minimum cover, which is usually
the answer to "why isn't `A*B` in my solution?".

The chart also exports as a table, one row per configuration and one column per
prime:

```python
result.chart.to_frame()
```

!!! note "The chart changes nothing"
    `minimize_chart` returns exactly the covers `minimize` returns — a
    parametrised test asserts the two agree. The extra structure is
    explanation, not a different algorithm.

`result.truncated` reports whether `max_solutions` cut the list of tied covers
short, so a capped result is never mistaken for an unambiguous one.

## Fitting solutions back to cases

A Boolean cover is a statement about truth-table rows. To evaluate it against
the original fuzzy cases, each implicant is re-evaluated as a conjunction under
the minimum t-norm and the cover as their disjunction under the maximum s-norm.
That is what produces the consistency, coverage and PRI reported for each
solution, and the per-term fits in `FittedSolution.term_fits`.

::: setqca.minimize.qmc

::: setqca.minimize.implicant
