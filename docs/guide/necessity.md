# Necessity

A condition is **necessary** for an outcome when the outcome is a subset of the
condition: wherever the outcome appears, the condition appears too. It is the
mirror of sufficiency, and it answers a different question.

| | Question | Set relation | Consistency |
| --- | --- | --- | --- |
| **Sufficiency** | Is this enough to produce the outcome? | `X ⊆ Y` | `Σ min(X,Y) / Σ X` |
| **Necessity** | Can the outcome occur without this? | `Y ⊆ X` | `Σ min(X,Y) / Σ Y` |

The two are duals: necessity of `X` for `Y` is sufficiency of `Y` for `X` with
the roles exchanged. Neither implies the other, and a condition can be both,
either, or neither.

## Screening

```python
from setqca import necessity_analysis

analysis = necessity_analysis(
    data,
    outcome="SURV",
    conditions=["DEV", "URB", "LIT", "IND", "STB"],
    consistency_threshold=0.90,
)
print(analysis)
print(analysis.to_frame())
```

Every condition is screened in **both directions** by default. A condition's
absence can be necessary when its presence is not, and screening only presence
is a common way to miss the finding.

Results are typed objects, not just a frame:

```python
analysis.necessary  # consistent and non-trivial
analysis.trivial  # consistent but uninformative
analysis.candidates  # everything screened
```

## The trivialness problem

This is the part that matters most.

A condition present in almost every case is a superset of almost anything. It
will show near-perfect necessity consistency while telling you nothing — you
cannot explain a rare outcome with a ubiquitous condition.

```python
data = pd.DataFrame({"ubiquitous": [1.0, 1.0, 1.0, 1.0], "Y": [0.8, 0.7, 0.2, 0.1]})
```

`ubiquitous` scores consistency 1.000 — apparently a perfect necessary
condition. It is an artefact of prevalence.

**Relevance of necessity** is what exposes it:

```text
RoN = Σ (1 − X) / Σ (1 − min(X, Y))
```

The more prevalent `X` is, the smaller the numerator, and the closer RoN falls
to zero. In the example above RoN is exactly 0.

`setqca` therefore reports both, and separates the two lists:

```text
Necessary:
  ~C [cons=1.000, cov=0.900, RoN=0.909]
Consistent but trivial (prevalent enough to be uninformative):
  B [cons=1.000, RoN=0.000, prevalence=1.000]
```

A candidate is `necessary` only when it clears **both** thresholds. Clearing
consistency alone makes it `trivial`, which is reported rather than left for
the reader to notice.

!!! danger "Consistency alone is not evidence of necessity"
    A high necessity consistency with a low RoN is the single most common way a
    QCA writes up a finding that isn't there. Always report both.

## Compound conditions

Only **disjunctions** are screened, and this is a mathematical result rather
than a limitation:

```text
consistency(A*B) ≤ min(consistency(A), consistency(B))    because min(A,B,Y) ≤ min(A,Y)
consistency(A+B) ≥ max(consistency(A), consistency(B))    because min(max(A,B),Y) ≥ min(A,Y)
```

A conjunction can never be more necessary than its own components, so testing
conjunctions adds nothing. A union *can* be necessary when neither part is —
the **SUIN** condition of the literature, a sufficient part of an insufficient
but necessary condition.

```python
analysis = necessity_analysis(
    data,
    outcome="SURV",
    conditions=["DEV", "URB", "LIT"],
    max_disjunction_size=2,
)
```

Beware that the number of unions grows quickly: with `k` literals and size `n`
there are `C(k, n)` of them, and screening many raises the chance that one
clears the threshold by luck. Treat unions as hypotheses to examine, not
findings to report.

## Necessity is not causation

Necessity is a statement about set relations in the data you have. It does not
establish that the condition produces the outcome, that removing it would
prevent the outcome, or that the relation holds outside your cases. A
constant-across-cases condition is necessary in the data and may be causally
irrelevant, and vice versa.

::: setqca.analysis.necessity
