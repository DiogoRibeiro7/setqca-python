# Roadmap

Released versions record what shipped. Everything below the line is intent, not
commitment, and the ordering matters more than the version numbers attached to
it.

## 0.1 — foundation (released 2026-08-10)

[10.5281/zenodo.21879360](https://doi.org/10.5281/zenodo.21879360)

- direct fuzzy calibration, logistic and piecewise
- crisp calibration
- typed fuzzy-set expressions
- necessity/sufficiency parameters of fit
- complete binary truth tables
- exact classical Quine-McCluskey
- conservative and parsimonious csQCA/fsQCA solutions
- pandas-native result objects
- parity harness against R `QCA`

## 0.2 — parity and robustness (released 2026-08-11)

[10.5281/zenodo.21887472](https://doi.org/10.5281/zenodo.21887472)

- correct intermediate solutions, with easy/difficult counterfactual
  classification matching R
- multi-value QCA
- enhanced necessity analysis: supersets, trivialness, relevance of necessity,
  SUIN conditions
- solution-specific unique coverage
- robustness sweeps over cutoffs and case removal
- Schneider–Rohlfing case typology
- calibration diagnostics
- prime-implicant chart inspection, so a solution can explain itself
- R parity extended to intermediate solutions, necessity screens, per-term fits
  and multi-value models, with the two known divergences pinned by tests

## Unreleased

- exact minimisation roughly two orders of magnitude faster, via a bitmask cube
  representation
- phase-level benchmark harness
- a complexity warning raised before the exponential search, not after

---

## Next

**A second minimisation engine.** A CCubes/eQMC-style backend, selected
explicitly rather than substituted silently, so a fast approximate answer is
never mistaken for the exact one the current engine guarantees.

**Cross-language validation as a running check.** The R fixtures are committed
golden values today; the generator should run on a schedule so divergences
surface when the reference implementation moves, rather than when someone next
looks.

**Simulation.** Generating data with a known causal structure is what makes it
possible to ask whether the method recovers what is actually there — coverage of
the true solution, behaviour under noise and limited diversity.

**Visualisation.** XY plots, truth-table and chart rendering. Deliberately after
simulation: a plot of an unvalidated result is a confident-looking wrong answer.

**Provenance.** Recording the calibration anchors, cutoffs and version that
produced a result, so a published analysis can be reproduced from the artefact
rather than from a description of it.

## 1.0

- a public API settled deliberately rather than by accretion, and then frozen
- temporal QCA (tQCA)
- multi-outcome models
- optional R-compatible calibration snapping, for replicating an existing
  analysis exactly
- benchmark corpus
- documentation rebuilt around tasks rather than modules
- published software paper
