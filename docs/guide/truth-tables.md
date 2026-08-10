# Truth tables

The truth table is the analytical heart of QCA. It enumerates every logically
possible configuration of the conditions — all \(2^k\) corners of the property
space — and reports what the evidence says about each.

## Construction

```python
from setqca import build_truth_table

table = build_truth_table(
    data,
    outcome="Y",
    conditions=["A", "B", "C"],
    inclusion_cutoff=0.8,
    exclusion_cutoff=0.5,
    pri_cutoff=0.6,
    frequency_cutoff=2,
    case_id="country",
)
```

## Corner assignment

Each case is assigned to the corner implied by whether each of its memberships
lies above or below the crossover. A case with `A=0.9, B=0.2` belongs to corner
`A=1, B=0`.

Corner *membership* is then computed for every case in every corner using the
minimum t-norm, with absent conditions negated as `1 - x`. This is what makes
consistency a fuzzy quantity rather than a simple count.

## Row coding

Rows are coded in this order:

| Code | Condition | Meaning |
| --- | --- | --- |
| `R` | `n < frequency_cutoff` | Logical remainder — too little evidence to judge |
| `1` | `consistency >= inclusion_cutoff` **and** `PRI >= pri_cutoff` | Sufficient for the outcome |
| `C` | `consistency >= exclusion_cutoff` | Contradictory — between the two cutoffs |
| `0` | otherwise | Not sufficient |

The frequency test comes first: a row with insufficient cases is a remainder no
matter how consistent the few cases it has happen to be.

### The contradictory band

By default `exclusion_cutoff` equals `inclusion_cutoff`, which collapses the
`C` band to nothing — every observed row is either `1` or `0`. Setting a lower
exclusion cutoff creates an explicit grey zone:

```python
build_truth_table(data, outcome="Y", conditions=[...], inclusion_cutoff=0.8, exclusion_cutoff=0.5)
```

Rows coded `C` participate in neither the on-set nor the don't-care set. They
are excluded from minimisation, which is deliberately conservative: an
ambiguous row should not silently drive a solution.

## Inspecting the table

```python
print(table.to_frame())
```

The tidy frame carries the condition states, the `minterm` index, case count
`n`, `consistency`, `PRI`, the `OUT` code, and the case labels.

Minterm indices are big-endian over the condition order you supplied, so with
conditions `["A", "B", "C"]` the configuration `A=1, B=1, C=0` is minterm 6.

Set-valued accessors give direct access to each group:

```python
table.positive_minterms  # coded "1"
table.negative_minterms  # coded "0"
table.contradictory_minterms  # coded "C"
table.remainder_minterms  # coded "R"
```

## Limited diversity

The gap between \(2^k\) logically possible configurations and the handful you
actually observe is *limited diversity*, and it is the central practical problem
in QCA. With 6 conditions there are 64 corners; a study of 25 cases can occupy
at most 25 of them.

Remainders are exactly what the conservative and parsimonious solutions disagree
about:

- the **conservative** solution uses no remainders, so it assumes nothing;
- the **parsimonious** solution treats every remainder as a don't-care, so it
  assumes each unobserved configuration behaves however is most convenient.

Neither is more correct in general. Report the number of remainders alongside
your solutions — if most of your property space is unobserved, the parsimonious
solution rests almost entirely on untested assumptions.

::: setqca.truth_table
