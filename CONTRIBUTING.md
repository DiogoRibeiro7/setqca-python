# Contributing to setqca

Thank you for considering a contribution. Please read the
[Code of Conduct](https://github.com/DiogoRibeiro7/setqca-python/blob/main/CODE_OF_CONDUCT.md)
before participating.

**Scientific correctness takes precedence over API convenience or speed.**

## The governing rule

Do not introduce heuristic Boolean minimisation under an API described as exact.
More generally: if a computation is approximate, its name, its docstring and its
documentation must say so. A user reading a solution should never have to guess
whether it is a proven minimum.

## Development setup

```bash
git clone https://github.com/DiogoRibeiro7/setqca-python.git
cd setqca-python
poetry install --with dev,docs
poetry run pre-commit install
```

## The quality gate

Every change must pass all four checks. CI runs them on Linux, macOS and
Windows across Python 3.11, 3.12 and 3.13.

```bash
make check
```

Or individually:

```bash
poetry run ruff check .            # lint
poetry run ruff format --check .   # formatting
poetry run mypy                    # strict typing
poetry run pytest --cov=setqca     # tests and coverage floor
```

Notes on the toolchain:

- **Coverage** is enforced at 95%. The package currently sits at 100%; please
  keep it there.
- **Warnings are errors** in the pytest configuration. A numerical warning such
  as an overflow is a bug, not noise.
- **mypy** runs under Python 3.12 semantics because `numpy` ships PEP 695 stubs,
  while the runtime still supports 3.11. Ruff's `target-version = "py311"`
  enforces 3.11-compatible syntax.

## Changes to the mathematical core

Calibration, parameters of fit, truth-table coding and Boolean minimisation are
held to a higher standard than the rest of the codebase. A change to any of them
must include:

1. **A mathematical description.** State what is computed, in what notation, and
   cite the source — a paper, a textbook section, or the equivalent R `QCA`
   function.
2. **Unit tests.** Preferably against a result that is known analytically or
   published.
3. **At least one edge-case test.** Empty sets, zero denominators, single cases,
   perfect consistency, complete limited diversity.
4. **A parity fixture**, where an equivalent R `QCA` feature exists — or a
   documented reason for the divergence.

Property-based tests using Hypothesis are strongly encouraged for anything that
should hold for all admissible inputs. They have already caught at least one
real numerical bug in this codebase that example-based tests missed.

## Test layout

| File | Purpose |
| --- | --- |
| `tests/test_*.py` | Behaviour of each module against known results |
| `tests/test_properties.py` | Invariants that must hold for all admissible inputs |
| `tests/test_qmc_exactness.py` | Minimiser verified against brute-force enumeration |
| `tests/test_errors.py` | Every documented failure mode |
| `tests/test_package.py` | Packaging and public-API contracts |

`tests/test_errors.py` deserves particular care. Much of this package's value is
in refusing to guess; a guard that silently stops firing is a correctness
regression even though nothing appears to break.

## Documentation

Public API changes must be reflected in:

- the numpydoc-style docstring (rendered into the API reference by
  mkdocstrings);
- the relevant user-guide page under `docs/guide/`;
- `CHANGELOG.md`, under `Unreleased`.

Build the site locally with `make docs`, which serves it at
<http://127.0.0.1:8000> with live reload. CI builds it with `--strict`, so a
broken cross-reference fails the build.

## Pull requests

1. Branch from `main`.
2. Keep the change focused; unrelated refactors belong in their own PR.
3. Fill in the PR template, including the scientific-correctness checklist where
   it applies.
4. Ensure CI is green.

Commits should explain *why* rather than restating *what* the diff shows.

## Reporting problems

- **Bugs** — [bug report](https://github.com/DiogoRibeiro7/setqca-python/issues/new?template=bug_report.yml)
- **Numerical disagreement with R `QCA`** — [parity report](https://github.com/DiogoRibeiro7/setqca-python/issues/new?template=parity_report.yml).
  A confirmed divergence in a stable component takes priority over feature work.
- **Security vulnerabilities** — privately, per
  [SECURITY.md](https://github.com/DiogoRibeiro7/setqca-python/blob/main/SECURITY.md).
  Never in a public issue.

## Releases

Maintainers only:

1. Update the version in `pyproject.toml`, `CITATION.cff`, `codemeta.json` and
   `.zenodo.json`. `tests/test_package.py` enforces that the first two agree.
2. Move the `Unreleased` section of `CHANGELOG.md` under the new version.
3. Tag `vX.Y.Z` and push. The release workflow verifies that the tag matches the
   declared version, re-runs the full quality gate, builds the distributions and
   publishes to PyPI via trusted publishing. No API token is stored anywhere.
