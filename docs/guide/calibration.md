# Calibration

Calibration converts raw measures into set memberships. It is the step where
substantive knowledge enters the analysis, and it determines everything
downstream. A truth table built on poor anchors is precisely wrong.

## Direct calibration

Three anchors define the transformation:

| Anchor | Membership | Meaning |
| --- | --- | --- |
| `full_out` | ≈ `1 - idm` | Fully outside the set |
| `crossover` | 0.5 | Maximum ambiguity |
| `full_in` | ≈ `idm` | Fully inside the set |

```python
from setqca import calibrate_direct

membership = calibrate_direct(
    raw_innovation,
    full_out=10,
    crossover=50,
    full_in=90,
)
```

With the default `idm=0.95`, the anchors map to approximately `0.05`, `0.5` and
`0.95` for an increasing set.

### Increasing and decreasing sets

The anchor *order* selects the direction. `full_out < crossover < full_in`
defines an increasing set; reversing the order defines a decreasing one.

```python
# "low corruption": higher raw scores mean lower membership
low_corruption = calibrate_direct(corruption, full_out=80, crossover=50, full_in=20)
```

Anchors that are not strictly ordered around the crossover raise `ValueError`
rather than producing an arbitrary curve.

### Logistic versus piecewise

```python
logistic = calibrate_direct(x, full_out=0, crossover=50, full_in=100)
piecewise = calibrate_direct(x, full_out=0, crossover=50, full_in=100, logistic=False)
```

| | Logistic (default) | Piecewise |
| --- | --- | --- |
| Endpoints | Approached asymptotically, never exactly reached | Exactly 0 and 1 at and beyond the anchors |
| Shape control | `idm` | `below` and `above` exponents |
| Use when | You want the standard smooth transformation | You need exact full membership for cases at or beyond an anchor |

The logistic transformation is evaluated in a numerically stable form, so values
far outside the anchors saturate cleanly to 0 or 1 rather than overflowing.

### Shaping the piecewise curve

`below` and `above` are positive exponents applied on either side of the
crossover. Raising `above` accelerates the approach to full membership:

```python
calibrate_direct([75], full_out=0, crossover=50, full_in=100, logistic=False)  # 0.750
calibrate_direct([75], full_out=0, crossover=50, full_in=100, logistic=False, above=2.0)  # 0.875
```

The anchors themselves always map to 0, 0.5 and 1 regardless of the exponents.

## Crisp calibration

`calibrate_crisp` cuts a raw variable into ordered integer categories using
threshold semantics equivalent to R's `findInterval`.

```python
from setqca import calibrate_crisp

binary = calibrate_crisp(gdp_per_capita, [20_000])
categories = calibrate_crisp(gdp_per_capita, [10_000, 20_000, 30_000])
```

One threshold yields a binary crisp set. Multiple thresholds yield categories
`0..k`; this signature is the forward-compatible API for multi-value QCA, which
is a roadmap item rather than a current feature.

Thresholds are sorted automatically, and duplicates raise `ValueError`.

## The 0.5 problem

A membership of exactly 0.5 has no defined truth-table corner: it is neither
more in than out nor more out than in. `setqca` refuses to guess.

```python
build_truth_table(data, outcome="Y", conditions=["A"])
# ValueError: At least one condition is exactly 0.5 ...
```

Resolve these cases substantively — by revisiting the anchors, or by making a
documented decision about the case — or opt in explicitly:

```python
build_truth_table(data, outcome="Y", conditions=["A"], allow_crossover_cases=True)
```

With the override, scores of exactly 0.5 are assigned to the *present* corner,
because corner assignment uses `x >= 0.5`.

## Specifications

A calibration is a decision worth recording. `CalibrationSpec` makes it a value
you can store, compare, ship in a replication package, and replay:

```python
from setqca import calibrate, direct_spec

spec = direct_spec(
    "innovation",
    full_out=20,
    crossover=50,
    full_in=80,
    note="OECD reporting threshold; see section 3.2",
)
result = calibrate(raw["innovation"], spec)

result.values  # the calibrated memberships
result.spec  # what produced them
result.diagnostics  # and what is worrying about them
```

The `note` carries the *reason* through serialisation, because the reason is
part of the specification:

```python
spec.to_json()
CalibrationSpec.from_json(text)  # round-trips exactly
```

A specification is validated when it is written, not when it eventually meets
data — badly ordered anchors raise immediately.

### Indirect calibration

When theory dictates a shape the three-anchor transformation cannot express — a
plateau, a step, an asymmetric ramp — give the mapping explicitly:

```python
from setqca import indirect_spec

spec = indirect_spec(
    "capacity",
    mapping=((0, 0.0), (30, 0.5), (70, 0.5), (100, 1.0)),
    note="no meaningful variation between 30 and 70",
)
```

Points are interpolated linearly and held flat beyond the ends. The mapping must
be non-decreasing, since a calibration that reverses direction is a different
concept, not a calibration.

## Diagnostics

A calibration can be arithmetically valid and analytically useless.

```python
from setqca import diagnose_calibration, diagnose_frame

print(diagnose_calibration(calibrated))
diagnose_frame(data)  # one row per condition
```

Five failures are reported:

| Warning | Why it matters |
| --- | --- |
| Cases exactly at 0.5 | The truth-table corner is undefined. This is the only one that makes the vector *unusable*. |
| Pile-up near the crossover | Small anchor changes will move cases between corners, so the result is fragile. |
| Compression to the extremes | The calibration is effectively crisp; the fuzzy detail has been squeezed out. |
| Low variance | The condition barely varies and carries little information. |
| Never present / never absent | Every case falls on one side, so the condition cannot discriminate. |

None is fatal by itself — they are reported so you can decide, not enforced.

## Quantile helpers, and why they are not a calibration

```python
from setqca import suggest_anchors

print(suggest_anchors(raw["innovation"]))
```

```text
Suggested from quantiles (0.05, 0.5, 0.95): full_out=12, crossover=48, full_in=91
  Quantiles describe the sample, not the concept. Anchors must be justified
  substantively; these are a starting point for that argument, not a substitute
  for it.
```

!!! danger "Data-driven anchors are not calibration"
    A set defined by its own distribution cannot support a claim about set
    membership. If the crossover is the sample median, then "more in than out"
    means "above average for these cases" — which changes when you add a case,
    and says nothing about the concept.

    The helper exists to show you where your cases actually lie so you can
    argue for anchors. It returns the caveat attached to the result, and
    nothing in this package will apply quantile anchors for you.

## Reusing a calibration

`DirectCalibration` is a frozen dataclass, so a calibration is a value you can
store, pass around, and apply to new cases — which is what you need when
extending an analysis to additional cases without silently re-anchoring.

```python
from setqca import DirectCalibration

spec = DirectCalibration(full_out=10, crossover=50, full_in=90)
train = spec.transform(raw_train)
holdout = spec.transform(raw_holdout)
```

::: setqca.calibration
