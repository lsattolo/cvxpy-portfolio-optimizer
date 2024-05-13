"""Test portfolio optimization."""

import cvxpy as cp
import pandas as pd

from cvxpy_portfolio_optimizer._enums import ObjectiveFunctionName
from cvxpy_portfolio_optimizer.constraint_function import NoShortSellConstraint, SumToOneConstraint
from cvxpy_portfolio_optimizer.objective_function import VarianceObjectiveFunction
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
    variance_obj_val = next(
        obj_dict[ObjectiveFunctionName.VARIANCE]
        for obj_dict in ptf.objective_values
        if ObjectiveFunctionName.VARIANCE in obj_dict
    )
    assert abs(ptf_variance - variance_obj_val) < 1e-6
