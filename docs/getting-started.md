# Getting started

## Installation

`setqca` requires Python 3.11 or newer and depends only on `numpy` and `pandas`.

=== "pip"

    ```bash
    pip install setqca
    ```

=== "Poetry"

    ```bash
    poetry add setqca
    ```

=== "From source"

    ```bash
    git clone https://github.com/DiogoRibeiro7/setqca-python.git
    cd setqca-python
    poetry install
    ```

## A complete analysis

A QCA workflow has four stages: calibrate, build the truth table, minimise, and
interpret. `setqca` keeps each stage separately inspectable.

### 1. Calibrate

Raw measures must become set memberships in `[0, 1]` before anything else
happens. Direct calibration maps three substantive anchors onto the membership
scale.

```python
import pandas as pd
from setqca import calibrate_direct

raw = pd.DataFrame(
    {
        "digital": [12, 24, 45, 60, 72, 88, 95, 35],
        "skills": [20, 35, 52, 64, 75, 82, 90, 44],
        "innovation": [15, 30, 48, 70, 78, 91, 96, 37],
    }
)

data = raw.copy()
for column in data.columns:
    data[column] = calibrate_direct(data[column], full_out=20, crossover=50, full_in=80)
```

The anchors are substantive claims about your cases, not statistical summaries.
`full_out=20` asserts that a score of 20 means fully outside the set.

### 2. Fit

```python
from setqca import FSQCA

model = FSQCA(consistency=0.8, pri=0.5, frequency=1)
result = model.fit(data, outcome="innovation", conditions=["digital", "skills"])
```

### 3. Inspect the truth table

Always read the truth table before reading the solution. It shows which
configurations were actually observed and which are logical remainders.

```python
print(result.truth_table.to_frame())
```

```text
   digital  skills  minterm  n  consistency   PRI OUT   cases
0        0       0        0  3     0.336634  ...   0   0, 1, 7
1        0       1        1  0     0.000000  ...   R
2        1       0        2  0     0.000000  ...   R
3        1       1        3  5     0.964427  ...   1   2, 3, 4, 5, 6
```

### 4. Read the solutions

```python
print(result)
print(result.summary_frame("parsimonious"))
```

`summary_frame` returns a tidy `DataFrame`, so solutions compose with the rest
of the Python data stack.

## Choosing a solution family

| Family | Remainders used | Interpretation |
| --- | --- | --- |
| `conservative` | none | Makes no assumptions about unobserved configurations. The most defensible, least parsimonious. |
| `parsimonious` | all | Uses every logical remainder as a don't-care. The simplest expression, but rests on untested simplifying assumptions. |
| `intermediate` | those consistent with theory | **Experimental.** Only remainders that do not contradict your directional expectations. |

```python
model = FSQCA(
    consistency=0.8,
    directional_expectations={"digital": "+", "skills": "+"},
)
result = model.fit(data, outcome="innovation", conditions=["digital", "skills"])
print(result.summary_frame("intermediate"))
```

!!! warning "Intermediate solutions are experimental"
    The 0.1 implementation admits only remainders whose configuration does not
    contradict the supplied expectations. Standard QCA intermediate solutions
    involve a richer treatment of simplifying assumptions. Do not report these
    results as standard intermediate solutions until parity is established.

## Crisp-set analysis

`CSQCA` rejects any condition or outcome that is not already binary, so a
miscalibrated column fails loudly rather than being silently coerced.

```python
from setqca import CSQCA

result = CSQCA().fit(crisp_data, outcome="Y", conditions=["A", "B", "C"])
```

## Working with set expressions directly

Conditions compose with `&`, `|` and `~` into typed expressions you can evaluate
against any calibrated frame — useful for testing a specific hypothesised
configuration outside the minimisation pipeline.

```python
from setqca import Condition, sufficiency

A, B, C = Condition("A"), Condition("B"), Condition("C")
configuration = A & B & ~C

membership = configuration.evaluate(data)
print(sufficiency(membership, data["Y"]))
```
