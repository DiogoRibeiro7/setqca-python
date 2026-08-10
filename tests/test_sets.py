import pandas as pd
import pytest

from setqca import Condition


def test_typed_fuzzy_set_algebra() -> None:
    data = pd.DataFrame({"A": [0.2, 0.9], "B": [0.8, 0.6], "C": [0.1, 0.7]})
    A, B, C = Condition("A"), Condition("B"), Condition("C")
    result = (A & B & ~C).evaluate(data)
    assert result.tolist() == pytest.approx([0.2, 0.3])


def test_union_uses_maximum() -> None:
    data = pd.DataFrame({"A": [0.2, 0.9], "B": [0.8, 0.6]})
    result = (Condition("A") | Condition("B")).evaluate(data)
    assert result.tolist() == pytest.approx([0.8, 0.9])
