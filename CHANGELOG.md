# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-08-11

Every component now matches the reference R `QCA` implementation on the
canonical datasets, including intermediate solutions, which 0.1.0 computed
incorrectly. Multi-value QCA, necessity screening, robustness sweeps and
case-level diagnostics are new.

### Fixed

- **Intermediate solutions were wrong, not merely experimental.** 0.1.0 admitted
  any remainder that did not contradict the directional expectations. The
  standard procedure additionally requires the remainder to be reachable from a
  configuration that *was observed* to be sufficient. With that condition the
  implementation reproduces R exactly on the Lipset data, including R's split of
  the twelve simplifying assumptions into one easy and eleven difficult
  counterfactuals. Results from `summary_frame("intermediate")` will change, and
  the new values are the correct ones.
- Nested groups were lost when printing an expression: `(A+B)*C` rendered as
  `A+B*C`, which re-parses as a different set. The printer now parenthesises by
  precedence, and round-tripping is property-tested.

### Added

#### Analysis

- `necessity_analysis` screens conditions and their negations for necessity,
  separating genuine findings from **trivial necessity** — a prevalent condition
  scoring high consistency while explaining nothing. Disjunctions can be
  screened for SUIN conditions; conjunctions are excluded because
  `consistency(A*B) <= min` over the parts, so they can never help.
- `sufficiency_diagnostics` classifies every case against every term as typical,
  deviant for consistency in kind or degree, deviant for coverage, or
  individually irrelevant, and reports **unique coverage**, which 0.1.0 did not
  compute at all.
- `robustness_analysis` and `calibration_robustness` sweep consistency, PRI and
  frequency cutoffs and calibration anchors, reporting which paths are stable,
  threshold-sensitive, disappearing or emerging, with four solution-similarity
  measures. Specifications that produce no solution are recorded rather than
  dropped.
- `MVQCA` and `setqca.multivalue` bring multi-value QCA: categorical conditions
  minimised as multi-value cubes, not as Boolean indicators. The dummy encoding
  admits points where two indicators for one condition are both true, which
  correspond to no configuration, so the cube algebra is implemented directly.

#### Expressions

- A typed expression system: tokenizer, recursive-descent parser, AST, canonical
  form, simplification and evaluation, through `parse_expression`,
  `evaluate_expression` and `simplify_expression`. Parsing is structural — there
  is no `eval` anywhere, so an expression from a configuration file cannot
  execute anything.
- Simplification applies only laws valid for fuzzy sets. The complement laws are
  deliberately **not** applied: at `A = 0.5`, `min(A, 1-A)` is not empty.

#### Calibration

- `CalibrationSpec` makes a calibration a serialisable value, with `direct`,
  `crisp`, `indirect` and `identity` methods, validated when written rather than
  when applied. `indirect_spec` expresses shapes the three-anchor form cannot.
- `diagnose_calibration` and `diagnose_frame` report crossover pile-up,
  compression to the extremes, low variance and never-present conditions.
- `suggest_anchors` reports quantiles **with a caveat attached to the result**,
  and nothing applies them automatically: a set defined by its own distribution
  cannot support a claim about set membership.

#### Truth tables and minimisation

- Rows record `exclusion_reason` in words, distinguishing exclusion by the
  frequency cutoff, by consistency and by PRI. `excluded_rows()` returns only
  those a threshold held back.
- Truth tables serialise to JSON and re-minimise without the original data.
- `minimize_chart` exposes the prime-implicant chart: essential primes,
  dominated primes, per-row explanations, and whether the tie list was
  truncated.

#### Validation

- Parity fixtures generated from CRAN `QCA` 3.25 and committed, so the suite
  runs in CI without R. Covers calibration, truth tables, fit measures,
  necessity screening, per-term fit with unique coverage, intermediate solutions
  with counterfactual classification, and multi-value QCA.
- `docs/mathematical_validation.md` audits every exported quantity against its
  definition, and `tests/test_mathematical_core.py` pins the boundary and
  degenerate cases.

### Changed

- The exact cover solver was extracted as `solve_minimum_cover`, so the binary
  and multi-value engines share one verified implementation.
- `setqca.calibration` became a package; every existing import still works.
- Analysis modules live under `setqca.analysis` so that a module never shadows a
  function of the same name on the package namespace.
- `TruthTable.to_frame()` gained an `excluded_because` column.
- `QCAResult` gained `counterfactuals`, and `intermediate_experimental` is now
  always `False` on fitted results, retained for compatibility.

### Documented

- One known divergence from R in direct calibration. `QCA::calibrate` ends with
  `fs[fs < 1e-04] <- 0; fs[fs > 0.9999] <- 1`, so R reports extreme memberships
  as exactly 0 or 1 while setqca reports the value of the transformation. Within
  the anchors the two agree to machine precision. The divergence is bounded by
  `1e-4`, cannot change a truth-table corner assignment, and is pinned by a test
  so it cannot widen unnoticed.
- A second divergence in multi-value conservative solutions: R writes
  single-value literals only, so a term such as `regime[1]*wealth[1]` can be a
  proper subset of the prime implicant `regime{1,2}*wealth{1}`. Both covers are
  minimal and cover the same configurations, so parity is asserted on cost and
  coverage rather than on text.
- Guides for expressions, necessity, sufficiency diagnostics, robustness and
  multi-value QCA.

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
  (Multi-value QCA arrived in 0.2.0.)

[Unreleased]: https://github.com/DiogoRibeiro7/setqca-python/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/DiogoRibeiro7/setqca-python/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DiogoRibeiro7/setqca-python/releases/tag/v0.1.0
