"""Conftest."""

import pandas as pd
import pytest


@pytest.fixture()
def returns_data() -> pd.DataFrame:
    """Retruns data fixture."""
    rets = pd.read_csv("tests/test_data/prices.csv", index_col=0).pct_change().dropna()
    rets.index = pd.to_datetime(rets.index)
    return rets
