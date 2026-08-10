# Roadmap

## 0.1 — foundation

- direct fuzzy calibration, logistic and piecewise
- crisp calibration
- typed fuzzy-set expressions
- necessity/sufficiency parameters of fit
- complete binary truth tables
- exact classical QMC
- conservative and parsimonious csQCA/fsQCA solutions
- experimental directional intermediate solution
- pandas-native result objects
- parity harness against R `QCA`

## 0.2 — parity and robustness

- standard intermediate-solution simplifying-assumption algorithm
- enhanced necessity analysis and supersets/subsets
- contradictory simplifying assumptions
- solution-specific unique coverage
- threshold multiverse and robustness API
- calibration diagnostics
- XY plots
- extend the R-QCA golden parity suite (calibration, truth tables, fit measures
  and conservative/parsimonious solutions are already covered as of 0.1;
  remaining: intermediate solutions, necessity supersets, multi-outcome models)
- optional R-compatible calibration snapping, so extreme memberships can be
  reported exactly as R does when replicating an existing analysis

## 0.3 — multi-value and performance

- mvQCA
- categorical-set expressions
- faster bitset/cube minimiser
- prime-implicant consistency filters
- row dominance
- optional Rust acceleration

## 1.0

- tQCA
- stable public API
- benchmark corpus
- exhaustive cross-software validation
- published algorithm and software paper
