"""Test portfolio optimization."""

import cvxpy as cp
import pandas as pd

from cvxpy_portfolio_optimizer._enums import ObjectiveFunctionName
from cvxpy_portfolio_optimizer.constraint_function import (
    NoShortSellConstraint,
    SumToOneConstraint,
    TrackingErrorConstraint,
)
from cvxpy_portfolio_optimizer.objective_function import (
    CVaRObjectiveFunction,
    VarianceObjectiveFunction,
)
from cvxpy_portfolio_optimizer.portfolio_optimization_problem import PortfolioOptimizationProblem


def test_variance_optimization(returns_data: pd.DataFrame) -> None:
    """Test minimum variance optimization."""
    pop = PortfolioOptimizationProblem(
        returns=returns_data,
        objective_functions=[VarianceObjectiveFunction()],
        constraint_functions=[
            SumToOneConstraint(),
            NoShortSellConstraint(),
        ],
    )
    ptf = pop.solve(solver=cp.CLARABEL)
    assert abs(ptf.weights.sum() - 1.0) < 1e-6
    assert (ptf.weights >= 0.0).all()
    annualized_sigma = 252 * returns_data.cov()
    ptf_variance = ptf.weights @ annualized_sigma @ ptf.weights
    variance_obj_val = ptf.get_objective_value(ObjectiveFunctionName.VARIANCE)
    assert abs(ptf_variance - variance_obj_val) < 1e-6
    assert isinstance(ptf.portfolio_timeseries(), pd.Series)


def test_tracking_error_constraint(returns_data: pd.DataFrame) -> None:
    """Test CVaR optimization with Tracking Error constraint."""
    benchmark = returns_data["AAPL"]
    tracking_error_ub = 0.0
    pop = PortfolioOptimizationProblem(
        returns=returns_data,
        objective_functions=[
            CVaRObjectiveFunction(confidence_level=0.9),
        ],
        constraint_functions=[
            SumToOneConstraint(),
            NoShortSellConstraint(),
            TrackingErrorConstraint(benchmark_returns=benchmark, upper_bound=tracking_error_ub),
        ],
    )
    ptf = pop.solve(solver=cp.CLARABEL)
    tracking_error = (returns_data @ ptf.weights - benchmark).std()
    assert abs(tracking_error - tracking_error_ub) < 1e-6
    assert all(w == "AAPL" for w in ptf.weights[ptf.weights > 0].index)
