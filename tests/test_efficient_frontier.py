"""Test efficient frontier module."""

import cvxpy as cp
import pandas as pd

from cvxpy_portfolio_optimizer.effiticent_frontier import EfficientFrontier
from cvxpy_portfolio_optimizer.objective_function import CVaRObjectiveFunction


def test_cvar_efficient_frontier(returns_data: pd.DataFrame) -> None:
    """Test CVaR efficient frontier."""
    cvar_ef = EfficientFrontier(
        returns=returns_data,
        objective_function=CVaRObjectiveFunction(confidence_level=0.95),
    )
    ef_responses = cvar_ef.compute_efficient_frontier(npoints=10, solver=cp.CLARABEL)
    assert len(ef_responses) == 10
    assert ef_responses[0].risk <= ef_responses[-1].risk
    assert ef_responses[0].expected_return <= ef_responses[-1].expected_return
