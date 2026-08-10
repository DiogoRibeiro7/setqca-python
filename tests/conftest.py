"""Shared fixtures for the setqca test suite."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def crisp_data() -> pd.DataFrame:
    """Return a small crisp dataset where Y is present exactly when A is."""
    return pd.DataFrame(
        {
            "case": ["c1", "c2", "c3"],
            "A": [1, 1, 0],
            "B": [1, 0, 0],
            "Y": [1, 1, 0],
        }
    )


@pytest.fixture
def fuzzy_data() -> pd.DataFrame:
    """Return a small calibrated fuzzy dataset with two conditions."""
    return pd.DataFrame(
        {
            "A": [0.9, 0.8, 0.2, 0.1],
            "B": [0.9, 0.2, 0.8, 0.1],
            "Y": [0.95, 0.85, 0.2, 0.1],
        }
    )
