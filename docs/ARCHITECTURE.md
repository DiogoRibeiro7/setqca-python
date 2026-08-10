# Architecture

```text
raw numeric data
      |
      v
calibration.py
      |
      v
calibrated DataFrame [0,1]
      |
      +-------------> sets.py ---------> metrics.py
      |
      v
truth_table.py
      |
      v
positive / negative / contradictory / remainder rows
      |
      v
minimize/implicant.py
      |
      v
minimize/qmc.py
  prime implicants
  exact PI chart
      |
      v
models.py
  conservative
  parsimonious
  experimental intermediate
      |
      v
results.py
  structured Python objects
  pandas exports
```

The numerical set-theoretic layer and the Boolean minimisation layer are deliberately independent. This allows each mathematical component to be validated separately and permits future alternative minimisers without changing calibration or truth-table semantics.
