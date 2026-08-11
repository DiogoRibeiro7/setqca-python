# Mathematical validation

An audit of every mathematical quantity `setqca` exports: what it is defined to
compute, where that is implemented, where it is tested, and what has actually
been verified.

Notation throughout: `X` is the membership of a cause or configuration, `Y` the
membership of the outcome, both vectors in `[0, 1]` over the same cases. Sums
run over all cases.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| ✅ Verified | Hand-computed test, plus agreement with R `QCA` where an equivalent exists |
| ✅ Tested | Hand-computed test; no R equivalent to compare against |
| ⚠️ Experimental | Implemented, but not to the standard definition — see notes |
| ❌ Not implemented | Listed for completeness; absent from the package |

## Core quantities

| Object | Equation | Implementation | Test | Tolerance | Status |
| --- | --- | --- | --- | --- | --- |
| Crisp membership | `x ∈ {0, 1}` | [`calibrate_crisp`](https://github.com/DiogoRibeiro7/setqca-python/blob/main/src/setqca/calibration.py) | `TestMembership`, `TestCalibration` | exact | ✅ Tested |
| Fuzzy membership | `x ∈ [0, 1]` | `_validation.validate_membership` | `TestMembership` | exact | ✅ Tested |
| Negation | `~A = 1 - A` | `sets.Negation` | `TestFuzzyOperators` | `1e-12` | ✅ Verified |
| Intersection | `A * B = min(A, B)` | `sets.Intersection` | `TestFuzzyOperators` | `1e-12` | ✅ Verified |
| Union | `A + B = max(A, B)` | `sets.Union` | `TestFuzzyOperators` | `1e-12` | ✅ Verified |
| Subset relation | `X ⊆ Y ⟺ ∀i: xᵢ ≤ yᵢ` | expressed through consistency | `TestSubsetRelations` | `1e-12` | ✅ Tested |
| Sufficiency consistency | `Σ min(X,Y) / Σ X` | `metrics.sufficiency` | `TestSufficiency` | `1e-9` vs R | ✅ Verified |
| Raw coverage | `Σ min(X,Y) / Σ Y` | `metrics.sufficiency` | `TestSufficiency` | `1e-9` vs R | ✅ Verified |
| PRI | `(Σ min(X,Y) − Σ min(X,Y,1−Y)) / (Σ X − Σ min(X,Y,1−Y))` | `metrics.sufficiency` | `TestSufficiency` | `1e-9` vs R | ✅ Verified |
| Necessity consistency | `Σ min(X,Y) / Σ Y` | `metrics.necessity` | `TestNecessity` | `1e-9` vs R | ✅ Verified |
| Necessity coverage | `Σ min(X,Y) / Σ X` | `metrics.necessity` | `TestNecessity` | `1e-9` vs R | ✅ Verified |
| Relevance of necessity | `Σ (1−X) / Σ (1 − min(X,Y))` | `metrics.necessity` | `TestNecessity` | `1e-9` vs R | ✅ Verified |
| Unique coverage | `cov(Tᵢ) − cov(⋃ⱼ≠ᵢ Tⱼ)` | — | — | — | ❌ Not implemented |
| Trivial necessity | `RoN` below threshold with high consistency | `analysis.necessity` | `test_necessity`, parity | `1e-9` vs R | ✅ Verified |
| SUIN disjunction | `consistency(A+B) ≥ max over parts` | `analysis.necessity` | `test_necessity` | `1e-12` | ✅ Tested |
| Direct calibration, logistic | see below | `calibration.DirectCalibration` | `TestCalibration`, parity | `1e-9` vs R | ✅ Verified |
| Direct calibration, piecewise | see below | `calibration.DirectCalibration` | `TestCalibration`, parity | `1e-9` vs R | ✅ Verified |
| Truth-table corner assignment | `xᵢ ≥ 0.5` | `truth_table.build_truth_table` | `TestTruthTableAndMinimisation`, parity | exact | ✅ Verified |
| Row coding | see below | `truth_table.build_truth_table` | parity | exact | ✅ Verified |
| Contradiction handling | `exclusion ≤ consistency < inclusion` | `truth_table.build_truth_table` | `TestTruthTableAndMinimisation` | exact | ✅ Tested |
| Prime implicants | classical Quine-McCluskey | `minimize.prime_implicants` | `test_qmc_exactness` | exact | ✅ Verified |
| Minimal cover | exact branch and bound | `minimize.exact_minimum_covers` | `test_qmc_exactness`, `test_qmc_reductions` | exact | ✅ Verified |
| Conservative solution | on-set only, no remainders | `models.FSQCA` | parity | exact | ✅ Verified |
| Parsimonious solution | remainders as don't-cares | `models.FSQCA` | parity | exact | ✅ Verified |
| Intermediate solution | Ragin-Sonnett easy counterfactuals | `counterfactuals`, `models.FSQCA` | `test_counterfactuals`, parity | exact | ✅ Verified |

## Definitions in full

### Direct calibration, logistic

With anchors `full_out`, `crossover`, `full_in` and `idm ∈ (0.5, 1)`:

```text
odds  = log(idm / (1 - idm))
scale = full_out - crossover   below the crossover
        full_in  - crossover   at or above it
z     = sign · (x - crossover) · odds / scale
μ     = 1 / (1 + exp(z))
```

The crossover maps to exactly `0.5`; the exclusion and inclusion anchors map to
`1 - idm` and `idm`. Evaluated through a numerically stable logistic, so the
transformation saturates rather than overflowing for extreme inputs.

### Direct calibration, piecewise

```text
μ = 0                                                       x ≤ full_out
μ = ((full_out - x) / (full_out - crossover))^below / 2      full_out < x ≤ crossover
μ = 1 - ((full_in - x) / (full_in - crossover))^above / 2    crossover < x ≤ full_in
μ = 1                                                       x > full_in
```

Unlike the logistic form this attains exactly 0 and 1 at the outer anchors.

### Row coding

Evaluated in this order, so an under-observed row is a remainder no matter how
consistent its few cases happen to be:

```text
R   n < frequency_cutoff
1   consistency ≥ inclusion_cutoff  and  PRI ≥ pri_cutoff
C   consistency ≥ exclusion_cutoff
0   otherwise
```

## Behaviour at the boundaries

Verified in `tests/test_mathematical_core.py`:

| Situation | Behaviour |
| --- | --- |
| `Σ X = 0` (cause empty) | Consistency, coverage and PRI are `0`, not `NaN` |
| `Σ Y = 0` (outcome empty) | Coverage and necessity consistency are `0` |
| `Σ (1 − min(X,Y)) = 0` | RoN is `0` |
| Membership exactly `0` or `1` | Admissible; operators reduce to crisp logic |
| Membership exactly `0.5` | Rejected for truth-table corner assignment unless `allow_crossover_cases=True` |
| `NaN` or infinite input | `ValueError`; never imputed or dropped |
| Membership outside `[0, 1]` | `ValueError`; never clipped |
| Single case | Admissible |
| Mismatched vector lengths | `ValueError` |

Returning `0` for an undefined ratio is a deliberate convention rather than a
mathematical claim: an empty cause has no cases to be consistent about. It is
recorded here because the alternative — propagating `NaN` — would silently
poison downstream aggregation.

## Findings

1. **Unique coverage is absent.** Raw coverage is implemented and verified; the
   per-term unique coverage reported by other QCA software is not. Solutions
   currently expose overall and per-term fit through `FittedSolution.term_fits`,
   which is raw coverage per term. This is a gap, not a divergence.
2. **Intermediate solutions now follow the standard algorithm.** Simplifying
   assumptions are derived from the parsimonious solution and split into easy
   and difficult counterfactuals, matching R `QCA` on the Lipset data.
3. **No untested mathematical helper remains.** Every function in `metrics.py`,
   `sets.py`, `calibration.py` and `minimize/` is reached by the suite, which
   runs at 100% line and branch coverage.
4. **One divergence from R**, in calibration only, characterised and pinned. See
   [Validation](VALIDATION.md).

## How to re-run this audit

```bash
poetry run pytest tests/test_mathematical_core.py -v   # hand-computed definitions
poetry run pytest -m parity                            # agreement with R QCA
poetry run pytest --cov=setqca                         # coverage of the core
```
