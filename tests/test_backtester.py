"""Test backtester module."""

import pandas as pd

from cvxpy_portfolio_optimizer._enums import RebalanceFrequency
from cvxpy_portfolio_optimizer.backtester import Backtester, BacktesterOut
from cvxpy_portfolio_optimizer.constraint_function import (
    NoShortSellConstraint,
    SumToOneConstraint,
)
from cvxpy_portfolio_optimizer.objective_function import VarianceObjectiveFunction


def test_variance_optimization(returns_data: pd.DataFrame) -> None:
    """Test minimum variance optimization."""
    bt = Backtester(
        learning_days=365,
        rebalance_frequency=RebalanceFrequency.ONE_MONTH,
        returns=returns_data,
        objective_functions=[VarianceObjectiveFunction()],
        constraint_functions=[SumToOneConstraint(), NoShortSellConstraint()],
    )
    bt_out = bt.run()
    assert isinstance(bt_out, BacktesterOut)
    assert isinstance(bt_out.portfolios, dict)
    assert isinstance(bt_out.portfolio_returns, pd.Series)
