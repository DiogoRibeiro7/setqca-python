"""Multi-value QCA: categorical conditions with more than two levels.

A multi-value condition takes one of several unordered categories — regime type,
welfare regime, sector — rather than being present or absent. Forcing such a
condition into a binary set either loses information or invents a dichotomy the
concept does not have.

The cube algebra is implemented directly rather than by encoding categories as
Boolean indicators; see :mod:`setqca.multivalue._cube` for why that encoding is
unsound. The exact cover is solved by the same verified solver the binary
engine uses, so both inherit one exactness guarantee.

Examples
--------
>>> import pandas as pd
>>> from setqca.multivalue import MVQCA
>>> data = pd.DataFrame(
...     {"regime": [0, 1, 2, 1], "wealth": [0, 1, 1, 0], "Y": [0.1, 0.9, 0.9, 0.2]}
... )
>>> result = MVQCA(consistency=0.8).fit(data, outcome="Y", conditions=["regime", "wealth"])
>>> print(result.summary_frame())  # doctest: +SKIP
"""

from __future__ import annotations

from ._cube import (
    MultiValueCube,
    MultiValueSolution,
    minimize_multivalue,
    prime_cubes,
)
from ._domain import MultiValueDomain
from ._model import (
    MVQCA,
    MultiValueResult,
    MultiValueRow,
    MultiValueTruthTable,
    build_multivalue_truth_table,
)

__all__ = [
    "MVQCA",
    "MultiValueCube",
    "MultiValueDomain",
    "MultiValueResult",
    "MultiValueRow",
    "MultiValueSolution",
    "MultiValueTruthTable",
    "build_multivalue_truth_table",
    "minimize_multivalue",
    "prime_cubes",
]
