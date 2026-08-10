# Validation

A QCA implementation is only useful if its numbers are right. This page states
exactly how correctness is established in `setqca`, and — just as importantly —
what is **not** yet verified.

## Layers of verification

### 1. Unit tests against known results

Hand-computed cases with analytically known answers: the three calibration
anchors mapping to 0.05/0.5/0.95, a perfect subset relation yielding consistency
1.0, `AB~C + ABC` reducing to `AB`.

### 2. Brute-force exactness tests

The minimiser is checked against exhaustive enumeration. For random 3-condition
tables, every possible cube is enumerated, every combination of cubes is tested
for coverage, and the true minimum cost is computed by brute force. The QMC
engine must match it exactly — see `tests/test_qmc_exactness.py`.

This is the strongest guarantee in the package: for problems small enough to
enumerate, "exact" is verified rather than asserted.

### 3. Property-based tests

Hypothesis generates arbitrary admissible inputs and checks invariants that must
hold universally (`tests/test_properties.py`):

- all parameters of fit lie in `[0, 1]`;
- necessity of X for Y equals sufficiency of Y for X with roles exchanged;
- PRI never exceeds consistency;
- De Morgan's laws hold for the fuzzy operators;
- calibration is monotone and bounded;
- every returned minimal cover covers the on-set, avoids the off-set, and ties
  on cost with every other returned cover;
- no prime implicant is strictly contained in another.

Property tests find what example tests cannot. The numerical-stability fix in
the logistic calibration came from one of them.

### 4. Error-contract tests

Every documented failure mode has a test (`tests/test_errors.py`). This matters
more than usual here: the package's value rests on refusing to guess — about
crossover cases, uncalibrated input, or unknown conditions — and a guard that
silently stops firing is a correctness regression.

### 5. Reference parity against R `QCA`

The R `QCA` package is used as a **reference implementation for parity tests**,
never as a runtime dependency. `validation/r/parity.R` runs canonical datasets
through R so results can be compared.

```bash
Rscript validation/r/parity.R validation/parity_input.csv
```

## Parity status

Parity coverage before 1.0 is expected to include:

| # | Component | Status |
| --- | --- | --- |
| 1 | Direct calibration | Harness present, fixtures pending |
| 2 | Truth-table row assignment | Harness present, fixtures pending |
| 3 | Consistency, coverage, PRI, RoN | Harness present, fixtures pending |
| 4 | Prime implicants | Harness present, fixtures pending |
| 5 | Conservative solutions | Harness present, fixtures pending |
| 6 | Parsimonious solutions | Harness present, fixtures pending |
| 7 | Standard intermediate solutions | **Not implemented to standard** |

!!! danger "What this means for your paper"
    Item 7 is the one to watch. The 0.1 intermediate solution is a directional
    filter on remainders, not the standard simplifying-assumption algorithm.
    Results from `summary_frame("intermediate")` should not be reported as
    standard intermediate solutions.

## Reporting a divergence

A confirmed numerical divergence from R `QCA` in a stable component is treated
as a correctness bug with priority over any feature work. Please open a
[parity report](https://github.com/DiogoRibeiro7/setqca-python/issues/new?template=parity_report.yml)
including both the Python and R code and output, and the R `QCA` version.

## Continuous verification

Every push runs the full suite on Linux, macOS and Windows across Python 3.11,
3.12 and 3.13, under `mypy --strict`, with a coverage floor enforced in CI.
Warnings are errors in the test configuration, so a numerical warning such as an
overflow fails the build rather than scrolling past.
