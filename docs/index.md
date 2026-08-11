# setqca

A native, typed Python implementation of **Qualitative Comparative Analysis
(QCA)**.

`setqca` is not an R wrapper. It provides an auditable Python implementation of
the mathematical core of crisp-set and fuzzy-set QCA, with exact Boolean
minimisation and data-science-friendly result objects.

!!! warning "Status: 0.1.0 alpha"
    Conservative and parsimonious csQCA/fsQCA are the stable focus. Directional
    intermediate solutions now follow Ragin and Sonnett (2005) and match the
    reference R `QCA` implementation on the canonical Lipset datasets.

## Design commitments

| Commitment | What it means in practice |
| --- | --- |
| **Exact, not heuristic** | Minimisation is classical Quine-McCluskey with a branch-and-bound solution of the prime-implicant chart. All tied minimal covers are returned, not an arbitrary one. |
| **Explicit, not implicit** | Every threshold is a named parameter. Ambiguous cases — such as a membership of exactly 0.5 — raise rather than being silently resolved. |
| **Typed end to end** | The package ships `py.typed` and passes `mypy --strict`. |
| **Honest about maturity** | Anything short of parity with R `QCA` is documented as such rather than quietly approximated. |

## Quick start

```python
import pandas as pd
from setqca import FSQCA, calibrate_direct

data = pd.DataFrame({"digital": [...], "skills": [...], "innovation": [...]})
for column in data.columns:
    data[column] = calibrate_direct(data[column], full_out=20, crossover=50, full_in=80)

result = FSQCA(consistency=0.85, pri=0.70, frequency=2).fit(
    data, outcome="innovation", conditions=["digital", "skills"]
)
print(result)
print(result.summary_frame("parsimonious"))
```

Continue with [Getting started](getting-started.md), or jump to the
[API reference](reference.md).

## Where to go next

- **[Getting started](getting-started.md)** — installation and a complete worked analysis.
- **[Calibration](guide/calibration.md)** — turning raw measures into set memberships.
- **[Truth tables](guide/truth-tables.md)** — cutoffs, row coding and remainders.
- **[Minimisation](guide/minimisation.md)** — the exact Boolean engine.
- **[Methodology](METHODOLOGY.md)** — the formal implementation contract.
- **[Validation](VALIDATION.md)** — how correctness is established and what is not yet verified.

## Citing

If you use `setqca` in published research, please cite the archived release:

> Ribeiro, D. (2026). *setqca: Native Python Crisp-Set and Fuzzy-Set Qualitative
> Comparative Analysis* (version 0.1.0) [Computer software]. Zenodo.
> <https://doi.org/10.5281/zenodo.21879360>

Cite the version DOI above when the exact version matters for reproducibility;
cite the concept DOI [10.5281/zenodo.21879359](https://doi.org/10.5281/zenodo.21879359)
to refer to the project as a whole. Citation metadata is provided in
[`CITATION.cff`](https://github.com/DiogoRibeiro7/setqca-python/blob/main/CITATION.cff).

## License

Released under the [MIT License](https://github.com/DiogoRibeiro7/setqca-python/blob/main/LICENSE).
