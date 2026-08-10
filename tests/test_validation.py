"""Tests for the shared input-validation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from setqca._validation import as_float_array, validate_columns, validate_membership


def test_as_float_array_rejects_multidimensional_input() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        as_float_array(np.zeros((2, 2)), name="values")


def test_as_float_array_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        as_float_array([], name="values")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_as_float_array_rejects_non_finite_values(bad: float) -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        as_float_array([1.0, bad], name="values")


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_validate_membership_rejects_scores_outside_the_unit_interval(bad: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_membership([0.5, bad], name="A")


def test_validate_membership_accepts_the_closed_interval() -> None:
    assert validate_membership([0.0, 1.0], name="A").tolist() == [0.0, 1.0]


def test_validate_columns_reports_every_missing_column() -> None:
    data = pd.DataFrame({"A": [1]})
    with pytest.raises(KeyError, match="Missing columns"):
        validate_columns(data, ["A", "B", "C"])


def test_validate_columns_rejects_duplicates() -> None:
    data = pd.DataFrame({"A": [1]})
    with pytest.raises(ValueError, match="unique"):
        validate_columns(data, ["A", "A"])


def test_validate_columns_preserves_order() -> None:
    data = pd.DataFrame({"A": [1], "B": [1]})
    assert validate_columns(data, ["B", "A"]) == ["B", "A"]
