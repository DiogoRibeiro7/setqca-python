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
never as a runtime dependency. Golden values are generated from CRAN `QCA` and
**committed** to `validation/fixtures/r_qca.json`, so the parity tests run in CI
and on any contributor's machine without R installed.

R is needed only to regenerate the fixtures:

```bash
Rscript validation/r/generate_fixtures.R validation/fixtures/r_qca.json
```

A change to that fixture is a change in what the reference implementation says,
and is reviewed as carefully as a change to the source.

The canonical Lipset datasets shipped with R `QCA` are used: `LF` (fuzzy) and
`LC` (crisp), across four analyses — two inclusion cutoffs, a crisp analysis,
and a reduced three-condition model. Intermediate solutions are checked
separately across three sets of directional expectations.

## Parity status

Verified against **R `QCA` 3.25** under R 4.5.1:

| # | Component | Status |
| --- | --- | --- |
| 1 | Direct calibration | ✅ Matches to double precision — one documented divergence, below |
| 2 | Truth-table row assignment | ✅ Matches: row coding, case counts, consistency and PRI |
| 3 | Consistency, coverage, PRI, RoN | ✅ Matches for sufficiency and necessity, including disjunctions |
| 4 | Prime implicants | ✅ Covered indirectly — solutions are built from them |
| 5 | Conservative solutions | ✅ Matches on all four analyses |
| 6 | Parsimonious solutions | ✅ Matches on all four analyses |
| 7 | Standard intermediate solutions | ✅ Matches, including the easy/difficult counterfactual split |

Solutions are compared as canonical sets of literal sets, so agreement is
genuine set equality rather than string formatting agreeing by luck.

### The one known divergence

`QCA::calibrate` ends with:

```r
fs[fs < 1e-04] <- 0
fs[fs > 0.9999] <- 1
```

R therefore reports a membership below `1e-4` as exactly 0, and above `0.9999`
as exactly 1. `setqca` reports the value of the transformation itself.

Within the anchors the two agree to machine precision (≤ 1e-16). They differ
only for values far outside the anchor range — for anchors 20/50/80, a raw score
of −50 calibrates to `5.46e-05` in `setqca` and to `0` in R.

This is deliberate. Snapping introduces a discontinuity into a continuous
transformation, and asserting *full* non-membership is a stronger claim than the
data supports. The difference is bounded by `1e-4` and cannot change a
truth-table corner assignment. `tests/test_parity.py` pins the divergence, so if
either implementation changes the test fails rather than drifting quietly.

!!! success "Intermediate solutions reached parity"
    Item 7 was the last outstanding gap. Intermediate solutions now follow Ragin
    and Sonnett (2005) and reproduce R's result on the canonical Lipset data —
    `DEV*URB*LIT*STB + DEV*LIT*~IND*STB` — along with R's classification of the
    twelve simplifying assumptions into one easy and eleven difficult
    counterfactuals.

    Parity is verified for three different expectation sets, so the tests would
    catch a change that happened to leave one of them unaffected.

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

The parity tests run in that same suite, because the golden values are committed
rather than computed at test time. A regression against R therefore fails CI on
every platform, not only on a maintainer's machine with R installed.

Run only the parity tests with:

```bash
poetry run pytest -m parity
```
