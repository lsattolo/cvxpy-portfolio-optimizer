"""Conftest."""

import pandas as pd
import pytest


@pytest.fixture()
def returns_data() -> pd.DataFrame:
    """Retruns data fixture."""
    return pd.read_csv("tests/test_data/prices.csv", index_col=0).pct_change().dropna()
