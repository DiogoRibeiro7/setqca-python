# Methodology and implementation contract

`setqca` treats QCA as set-theoretic comparative analysis, not as a predictive machine-learning model.

## Fuzzy operations

For calibrated membership scores in `[0, 1]`:

- negation: `~A = 1 - A`
- conjunction: `A * B = min(A, B)`
- disjunction: `A + B = max(A, B)`

## Sufficiency

For cause/configuration `X` and outcome `Y`:

- consistency: `sum(min(X, Y)) / sum(X)`
- coverage: `sum(min(X, Y)) / sum(Y)`
- PRI follows the proportional-reduction-in-inconsistency calculation used by R `QCA`.

## Necessity

- consistency: `sum(min(X, Y)) / sum(Y)`
- coverage: `sum(min(X, Y)) / sum(X)`
- RoN follows the relevance-of-necessity calculation used by R `QCA`.

## Truth-table assignment

Fuzzy cases are assigned to a binary corner according to whether each membership is below or above the crossover `0.5`. Exact crossover scores are rejected by default because the crisp corner is ambiguous.

A row is:

- `R` when frequency is below the frequency cutoff;
- `1` when sufficiency consistency and PRI pass their cutoffs;
- `C` when consistency lies between the exclusion and inclusion cutoffs;
- `0` otherwise.

## Boolean minimisation

The current minimiser is classical exact Quine-McCluskey:

1. generate minterm cubes;
2. combine adjacent cubes until prime implicants remain;
3. discard primes derived only from don't-cares, since they cover no required row;
4. construct the prime-implicant chart;
5. solve exact minimum covers with branch-and-bound;
6. optimise lexicographically by number of prime implicants and then literal count.

All tied minimum covers are returned, up to `max_solutions`. Model ambiguity is a
property of the data and is reported rather than resolved arbitrarily.

Step 5 applies three reductions, each of which provably leaves the set of
minimum covers unchanged:

- **Essential primes.** A minterm covered by exactly one prime forces that prime
  into every cover, so it is selected before the search begins.
- **Independent-set lower bound.** Uncovered minterms whose candidate primes are
  pairwise disjoint each require a distinct further prime. The size of such a
  set bounds the cost of any completion from below, so branches that cannot
  reach the incumbent cost are abandoned.
- **State memoisation.** The completions available from a partial solution
  depend only on which minterms remain uncovered. Reaching the same remaining
  set at a strictly worse cost can therefore never yield a better or tied cover.

None of these is a heuristic: each is a sound inference about the chart, and the
result remains a proven minimum.

Conservative solutions use observed positive rows only. Parsimonious solutions allow logical remainders as don't-care minterms.

## Intermediate solutions

The v0.1 directional intermediate implementation is explicitly marked experimental. It admits only remainder rows whose complete configuration does not contradict the supplied directional expectations. Standard QCA intermediate solutions involve a richer treatment of simplifying assumptions; parity with R `QCA` is required before this feature is promoted to stable.
