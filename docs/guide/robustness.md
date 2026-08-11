# Robustness

A QCA solution is conditional on decisions the data do not make for you: where
the calibration anchors sit, how consistent a row must be to count as
sufficient, how many cases a row needs. Reporting one solution from one set of
those choices hides how much of the result was the choice rather than the
evidence.

```python
from setqca import RobustnessGrid, robustness_analysis

analysis = robustness_analysis(
    data,
    outcome="SURV",
    conditions=["DEV", "URB", "LIT"],
    grid=RobustnessGrid(
        consistency=[0.75, 0.80, 0.85, 0.90],
        pri=[0.50, 0.60, 0.70],
        frequency=[1, 2],
    ),
)
print(analysis)
print(analysis.to_frame())
```

## What comes back

```text
Robustness of the conservative solution
Specifications: 24 (21 produced a solution, 3 did not)
Baseline: cons=0.85, pri=0.6, n=1

Stable terms (1):
  DEV*LIT*STB — 21/21 specifications
Threshold-sensitive terms (2):
  DEV*URB — 6/21 specifications
  URB*STB — 4/21 specifications
Baseline terms that do not survive: URB*STB

Stability is not validity: a mis-specified model can be perfectly stable.
```

Four buckets, each answering a different question:

| Accessor | Question |
| --- | --- |
| `stable_terms()` | Which paths survive nearly every cutoff? |
| `fragile_terms()` | Which appear only under some? |
| `disappearing_terms()` | Which baseline paths do **not** survive? |
| `emerging_terms()` | Which stable paths does the baseline miss? |

The threshold defaults to 0.8 and is adjustable on each call.

!!! note "Failures are recorded, not dropped"
    A specification that produces no solution gets a row with a `failure`
    message and `NaN` fit, rather than vanishing. "The model collapses above
    0.9" is a finding about your data, and silently omitting those rows would
    make the surviving ones look more robust than they are.

## Sweeping calibration anchors

Calibration is where substantive judgement enters, so it is also where a result
is most easily manufactured. `calibration_robustness` recalibrates from the raw
measures for each anchor combination:

```python
from setqca.analysis.robustness import calibration_robustness

analysis = calibration_robustness(
    raw_data,
    outcome="SURV",
    conditions=["DEV", "URB"],
    grid=RobustnessGrid(
        consistency=[0.80],
        anchors={"DEV": [(10, 50, 90), (20, 50, 80), (10, 40, 90)]},
    ),
    outcome_anchors=(10, 50, 90),
    base_anchors={"URB": (10, 50, 90)},
)
```

The input is raw, so every condition needs anchors from one source or the
other — swept in the grid, or fixed through `base_anchors`. Passing calibrated
data to this function, or a grid with anchors to `robustness_analysis`, is an
error rather than a silent mis-read.

## Comparing solutions

Textual identity is the strictest comparison and often the least informative.
Four scales are available:

```python
from setqca.analysis.robustness import solution_similarity

similarity = solution_similarity(left_terms, right_terms, data)
similarity.identical  # exact set equality
similarity.term_overlap  # Jaccard over terms
similarity.configurational  # Jaccard over the literals used
similarity.membership  # fuzzy Jaccard over case membership
```

The last is the one that catches agreement the text hides: two solutions can be
written differently and still select the same cases. `A` and `A+A*B` are
textually distinct and have membership similarity 1.0, because the second term
adds nothing.

```python
analysis.similarity_to_baseline()
```

## Robustness is not validity

A path that appears under every threshold is **stable**, not **true**.

Stability says the finding does not depend on one arbitrary cutoff. It says
nothing about whether the conditions are causally relevant, whether the
calibration was substantively sensible, whether the cases were well chosen, or
whether an omitted condition is doing the work. A thoroughly mis-specified model
can be perfectly stable — sweeping thresholds cannot detect a problem that lives
in the model rather than the cutoffs.

Nothing in this module reports a verdict. The measures are descriptive; the
interpretation is yours.

::: setqca.analysis.robustness
