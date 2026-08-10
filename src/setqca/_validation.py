"""Internal validation helpers shared across the public modules."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
"""One-dimensional array of double-precision membership or raw scores."""


def as_float_array(values: npt.ArrayLike, *, name: str) -> FloatArray:
    """Return a finite one-dimensional float array.

    Parameters
    ----------
    values : array_like
        Input values coercible to ``float64``.
    name : str
        Human-readable variable name used in error messages.

    Returns
    -------
    FloatArray
        Validated one-dimensional ``float64`` array.

    Raises
    ------
    ValueError
        If the input is not one-dimensional, is empty, or contains ``NaN``
        or infinite values.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def validate_membership(values: npt.ArrayLike, *, name: str) -> FloatArray:
    """Validate fuzzy-set membership scores in the closed interval ``[0, 1]``.

    Parameters
    ----------
    values : array_like
        Calibrated membership scores.
    name : str
        Human-readable variable name used in error messages.

    Returns
    -------
    FloatArray
        Validated membership scores.

    Raises
    ------
    ValueError
        If any score falls outside ``[0, 1]``.
    """
    array = as_float_array(values, name=name)
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain calibrated memberships in [0, 1].")
    return array


def validate_columns(data: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Validate that columns exist and are unique, and return them as a list.

    Parameters
    ----------
    data : pandas.DataFrame
        Frame whose columns are checked.
    columns : iterable of str
        Column names required to be present.

    Returns
    -------
    list of str
        The requested column names, in the order supplied.

    Raises
    ------
    KeyError
        If any requested column is absent from ``data``.
    ValueError
        If the requested column names contain duplicates.
    """
    cols = list(columns)
    missing = [column for column in cols if column not in data.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    if len(set(cols)) != len(cols):
        raise ValueError("Condition names must be unique.")
    return cols
