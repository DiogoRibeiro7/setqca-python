# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `QCAResult.solutions()` for retrieving a solution family by name, with
  validation of the family name.
- `setqca.__version__` is now read from the installed distribution metadata, so
  it can no longer drift from `pyproject.toml`.
- Public re-exports of `BooleanSolution`, `Implicant`, `minimize`, `QCAResult`,
  `FittedSolution`, `Intersection`, `Negation`, `Union`, `Direction` and
  `TruthCode` from the top-level namespace.
- Property-based test suite covering fit-parameter bounds, sufficiency/necessity
  duality, De Morgan's laws, calibration monotonicity, minimal-cover validity
  and prime-implicant irredundancy.
- Error-contract test suite covering every documented failure mode.
- Documentation site (MkDocs Material) with a user guide, methodology,
  validation policy and generated API reference.
- Community health files: `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue and pull
  request templates, `CODEOWNERS` and Dependabot configuration.
- Release automation publishing to PyPI via trusted publishing, and citation
  metadata in `codemeta.json` and `.zenodo.json`.

### Performance

- The prime-implicant chart solver now applies three exactness-preserving
  reductions — essential-prime selection, an independent-set lower bound, and
  memoisation of the uncovered-minterm state. On a random 8-condition table the
  solve time fell from roughly 270 seconds to under a second. Prime generation
  was never the bottleneck and is unchanged.
- `build_truth_table` no longer re-reads and re-validates every condition column
  once per truth-table row.

### Fixed

- The logistic direct calibration overflowed for values far outside the anchors,
  emitting a numerical warning. It is now evaluated in a numerically stable form
  that saturates cleanly across the whole real line. Found by a property test.
- `CSQCA.fit` raised `TypeError` on every call. `dataclass(slots=True)` rebuilds
  the class object, which invalidates the implicit `__class__` cell used by
  zero-argument `super()`; the estimator now dispatches explicitly.

### Changed

- Packaging migrated from legacy `[tool.poetry]` metadata to PEP 621
  `[project]`, with project URLs, maintainer metadata and expanded classifiers.
- `build_truth_table` computes corner memberships from a single validated array
  instead of re-reading and re-validating each condition column for every one of
  the `2**k` rows.
- The estimator method name is now a class attribute rather than being patched
  onto the result after fitting.
- Intermediate solutions are reported in `QCAResult.__str__`, labelled
  experimental.
- `benchmarks/benchmark_qmc.py` takes command-line options, reports prime
  generation and chart solving separately, and defaults to a width range that is
  safe to run in CI.
- When a model is ambiguous enough to exceed `max_solutions`, *which* subset of
  the tied minimum covers is returned may differ from 0.1.0. The cost of every
  returned cover is unchanged and still provably minimal.

### Removed

- `MANIFEST.csv`, a generated file-hash manifest that became stale on the first
  edit after generation.

## [0.1.0] — 2026-08-09

### Added

- Native Python csQCA/fsQCA package foundation.
- Direct fuzzy and crisp calibration.
- Typed fuzzy-set algebra.
- Sufficiency and necessity parameters of fit.
- Binary truth-table engine.
- Exact classical Quine-McCluskey minimisation.
- Exact prime-implicant chart solving.
- Conservative and parsimonious solutions.
- Experimental directional intermediate solution.
- R `QCA` reference-validation harness.
- Test, typing, lint and CI configuration.

[Unreleased]: https://github.com/DiogoRibeiro7/setqca-python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/setqca-python/releases/tag/v0.1.0
