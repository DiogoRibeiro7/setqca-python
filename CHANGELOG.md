# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Parity test suite against the reference R `QCA` implementation
  (`tests/test_parity.py`), verified against R `QCA` 3.25 on the canonical
  Lipset `LF` and `LC` datasets across four analyses. Direct calibration,
  truth-table coding, case counts, consistency, PRI, sufficiency and necessity
  fit including RoN, and both conservative and parsimonious solutions all match.
  Solutions are compared as canonical sets of literal sets rather than as
  formatted strings.
- `validation/r/generate_fixtures.R`, which emits golden values to
  `validation/fixtures/r_qca.json`. The fixtures are committed, so parity tests
  run in CI and for contributors without R installed; R is needed only to
  regenerate them.
- `make parity` and `make fixtures` targets.

### Documented

- One known divergence from R in direct calibration. `QCA::calibrate` ends with
  `fs[fs < 1e-04] <- 0; fs[fs > 0.9999] <- 1`, so R reports extreme memberships
  as exactly 0 or 1 while setqca reports the value of the transformation. Within
  the anchors the two agree to machine precision. The divergence is bounded by
  `1e-4`, cannot change a truth-table corner assignment, and is pinned by a test
  so it cannot widen unnoticed.

### Removed

- `validation/r/parity.R` and `validation/parity_input.csv`, a stub that printed
  one truth table for a toy dataset. Superseded by the fixture generator, which
  produces machine-checked golden values on canonical datasets.

## [0.1.0] — 2026-08-10

First public release.
Archived on Zenodo: [10.5281/zenodo.21879360](https://doi.org/10.5281/zenodo.21879360)
(concept DOI [10.5281/zenodo.21879359](https://doi.org/10.5281/zenodo.21879359)).

### Added

#### Calibration

- Three-anchor direct fuzzy calibration in logistic and piecewise linear/power
  forms, for both increasing and decreasing sets. The logistic transformation is
  evaluated in a numerically stable form, so values far outside the anchors
  saturate cleanly instead of overflowing.
- `DirectCalibration` as a reusable frozen specification, so one calibration can
  be applied to further cases without re-anchoring.
- Crisp calibration into ordered categories, with `findInterval` threshold
  semantics.

#### Set theory and parameters of fit

- Typed fuzzy-set algebra over calibrated conditions, composing with `&`, `|`
  and `~` under the minimum t-norm, maximum s-norm and `1 - x` negation.
- Sufficiency consistency, coverage and PRI; necessity consistency, coverage and
  RoN.

#### Truth tables

- Complete binary truth tables covering every corner of the property space.
- Frequency, inclusion, exclusion and PRI cutoffs, with explicit classification
  of positive, negative, contradictory and logical-remainder rows.
- Membership scores of exactly 0.5 are rejected by default, because the crisp
  corner is ambiguous; `allow_crossover_cases` opts in explicitly.
- Tidy `pandas` export via `TruthTable.to_frame()`.

#### Boolean minimisation

- Exact classical Quine-McCluskey prime-implicant generation.
- Exact branch-and-bound solution of the prime-implicant chart, optimising
  lexicographically by implicant count and then literal count, and returning
  every tied minimal cover up to `max_solutions` so model ambiguity stays
  visible.
- Three exactness-preserving reductions keep the search tractable:
  essential-prime selection, an independent-set lower bound, and memoisation of
  the uncovered-minterm state. None is a heuristic; the result remains a proven
  minimum.

#### Estimators and results

- `FSQCA` and `CSQCA` estimators, the latter rejecting any condition or outcome
  that is not already binary.
- Conservative and parsimonious solutions, plus an experimental directional
  intermediate solution that admits only remainders consistent with the supplied
  expectations.
- Typed result objects with overall and per-term parameters of fit,
  `QCAResult.solutions()` for retrieving a family by name, and
  `summary_frame()` for tidy `pandas` export.

#### Engineering

- Fully type annotated, shipping a `py.typed` marker and passing `mypy --strict`.
- 112 tests at 100% coverage: known-result unit tests, brute-force verification
  of minimiser exactness against exhaustive enumeration, property-based
  invariant tests, and a test for every documented error contract.
- Continuous integration on Linux, macOS and Windows across Python 3.11, 3.12
  and 3.13, covering lint, formatting, strict typing, tests, the documented
  example, distribution build and documentation build.
- Release automation publishing to PyPI via trusted publishing, weekly
  dependency auditing and CodeQL analysis.
- Documentation site with a user guide, methodology, validation policy and a
  generated API reference.
- R `QCA` reference-validation harness for parity testing.
- Citation metadata in `CITATION.cff`, `codemeta.json` and `.zenodo.json`.

### Known limitations

- Directional intermediate solutions are **experimental**. The implementation
  filters remainders against directional expectations rather than applying the
  standard simplifying-assumption algorithm, and results should not be reported
  as standard intermediate solutions until parity is established.
- Parity fixtures against the reference R `QCA` implementation are not yet
  populated; the harness is present but the golden suite is a 0.2 item.
- Multi-value and temporal QCA are out of scope for this release.

[Unreleased]: https://github.com/DiogoRibeiro7/setqca-python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/setqca-python/releases/tag/v0.1.0
