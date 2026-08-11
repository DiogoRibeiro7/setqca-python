# setqca

[![CI](https://github.com/DiogoRibeiro7/setqca-python/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/setqca-python/actions/workflows/ci.yml)
[![Documentation](https://github.com/DiogoRibeiro7/setqca-python/actions/workflows/docs.yml/badge.svg)](https://diogoribeiro7.github.io/setqca-python/)
[![PyPI](https://img.shields.io/pypi/v/setqca?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/setqca/)
[![Python](https://img.shields.io/pypi/pyversions/setqca?logo=python&logoColor=white&label=Python)](https://pypi.org/project/setqca/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21879359.svg)](https://doi.org/10.5281/zenodo.21879359)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A native, typed Python implementation of **Qualitative Comparative Analysis (QCA)**.

`setqca` is not an R wrapper. It provides an auditable Python implementation of
the mathematical core of crisp-set and fuzzy-set QCA, with exact Boolean
minimisation and data-science-friendly result objects.

> **Status: 0.2.0 alpha.** Conservative, parsimonious and intermediate
> csQCA/fsQCA all match the reference R `QCA` implementation on the canonical
> Lipset datasets. See the validation page for the one documented divergence.

📖 **[Documentation](https://diogoribeiro7.github.io/setqca-python/)** ·
🚀 **[Getting started](https://diogoribeiro7.github.io/setqca-python/getting-started/)** ·
🔬 **[Validation policy](https://diogoribeiro7.github.io/setqca-python/VALIDATION/)**

## Why this project

The mature R `QCA` ecosystem supports crisp-set, fuzzy-set, multi-value and
temporal QCA with exact Boolean minimisation. Python has individual QCA-related
projects, but there is still room for a general-purpose, typed and thoroughly
validated scientific implementation that lives natively in the Python data
stack.

Four commitments shape the design:

| Commitment | What it means in practice |
| --- | --- |
| **Exact, not heuristic** | Classical Quine-McCluskey with branch-and-bound solution of the prime-implicant chart. All tied minimal covers are returned, not an arbitrary one. |
| **Explicit, not implicit** | Every threshold is a named parameter. Ambiguous input — a membership of exactly 0.5, an uncalibrated column — raises instead of being silently resolved. |
| **Typed end to end** | Ships `py.typed`; passes `mypy --strict`; 100% test coverage enforced in CI. |
| **Honest about maturity** | Anything short of parity with R `QCA` is documented as such rather than quietly approximated. |

## Features

- crisp calibration
- three-anchor direct fuzzy calibration
  - logistic (numerically stable across the whole real line)
  - piecewise linear/power
  - increasing and decreasing sets
- typed set algebra with `&`, `|` and `~`
- sufficiency consistency, coverage and PRI
- necessity consistency, coverage and RoN
- complete binary truth tables
- frequency, consistency and PRI cutoffs
- contradiction and logical-remainder classification
- exact classical **Quine-McCluskey** prime-implicant generation
- exact branch-and-bound solution of the prime-implicant chart
- conservative solutions
- parsimonious solutions
- intermediate solutions with easy/difficult counterfactual reporting
- tidy pandas exports
- optional parity harness against R `QCA`

## Installation

```bash
pip install setqca
```

Requires Python 3.11+. Runtime dependencies are `numpy` and `pandas` only.

For a development checkout:

```bash
git clone https://github.com/DiogoRibeiro7/setqca-python.git
cd setqca-python
poetry install
```

## Usage

### Calibration

```python
from setqca import calibrate_direct

innovation = calibrate_direct(
    raw_innovation,
    full_out=10,
    crossover=50,
    full_in=90,
)
```

The default `idm=0.95` maps the three anchors to approximately `0.05`, `0.5` and
`0.95` for increasing sets.

### Typed fuzzy-set algebra

```python
from setqca import Condition

A = Condition("A")
B = Condition("B")
C = Condition("C")

configuration = A & B & ~C
membership = configuration.evaluate(data)
```

### fsQCA

```python
from setqca import FSQCA

model = FSQCA(
    consistency=0.85,
    pri=0.70,
    frequency=2,
)

result = model.fit(
    data,
    outcome="Y",
    conditions=["A", "B", "C", "D"],
    case_id="case",
)

print(result)
print(result.truth_table.to_frame())
print(result.summary_frame("parsimonious"))
```

### csQCA

```python
from setqca import CSQCA

result = CSQCA().fit(
    crisp_data,
    outcome="Y",
    conditions=["A", "B", "C"],
)
```

`CSQCA` rejects non-binary condition or outcome columns.

### Exact minimisation

The low-level engine is public for testing and research:

```python
from setqca.minimize import minimize

# AB~C + ABC -> AB
solutions = minimize({6, 7}, width=3)
print(solutions[0].as_expression(("A", "B", "C")))
# A*B
```

Logical remainders are explicit don't-cares:

```python
solutions = minimize(
    {6, 7},
    dont_cares={4, 5},
    width=3,
)
```

## Scientific validation policy

The R `QCA` package is used as a **reference implementation for parity tests**,
not as a runtime dependency. Golden values are generated from CRAN `QCA` and
committed to `validation/fixtures/r_qca.json`, so parity tests run in CI and on
any machine without R installed.

Verified against **R `QCA` 3.25** on the canonical Lipset datasets:

| Component | Status |
| --- | --- |
| Direct calibration | ✅ to double precision, one documented divergence |
| Truth-table coding, case counts, consistency, PRI | ✅ |
| Sufficiency and necessity fit, incl. PRI and RoN | ✅ |
| Conservative solutions | ✅ |
| Parsimonious solutions | ✅ |
| Intermediate solutions, easy/difficult counterfactuals | ✅ |

Correctness rests on five layers: unit tests against known results, brute-force
exactness tests of the minimiser, property-based invariant tests, error-contract
tests, and these R parity fixtures. See
[the validation page](https://diogoribeiro7.github.io/setqca-python/VALIDATION/)
for the single known divergence and what is still unverified, and
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the formal implementation
contract.

## Non-goals for 0.1

- claiming complete parity with R `QCA`;
- tQCA;
- CCubes/eQMC performance parity.

These are roadmap items rather than hidden approximations. See
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Development

```bash
poetry install --with dev,docs
poetry run pre-commit install

make check     # lint, format, types and tests
make docs      # serve the documentation locally
```

Without `make`:

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy
poetry run pytest --cov=setqca
```

Contributions are welcome — please read [CONTRIBUTING.md](CONTRIBUTING.md)
first, particularly the scientific-correctness requirements for changes to the
mathematical core.

## Citing

If you use `setqca` in published research, please cite the archived release:

> Ribeiro, D. (2026). *setqca: Native Python Crisp-Set and Fuzzy-Set Qualitative
> Comparative Analysis* (version 0.1.0) [Computer software]. Zenodo.
> <https://doi.org/10.5281/zenodo.21879360>

Two DOIs are available. Cite the **version DOI**
([10.5281/zenodo.21879360](https://doi.org/10.5281/zenodo.21879360)) when the
exact version matters for reproducibility, which for a set-theoretic method it
usually does. Cite the **concept DOI**
([10.5281/zenodo.21879359](https://doi.org/10.5281/zenodo.21879359)) to refer to
the project as a whole; it always resolves to the latest archived release.

Machine-readable metadata is provided in [`CITATION.cff`](CITATION.cff),
[`codemeta.json`](codemeta.json) and [`.zenodo.json`](.zenodo.json). GitHub
renders "Cite this repository" from the first of these.

## License

MIT — see [LICENSE](LICENSE).
