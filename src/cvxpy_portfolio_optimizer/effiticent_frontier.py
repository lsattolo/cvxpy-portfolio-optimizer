"""Portfolio optimization problem module."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cvxpy_portfolio_optimizer._models import EfficientFrontierResponse
from cvxpy_portfolio_optimizer.constraint_function import (
    ExpectedReturnConstraint,
    NoShortSellConstraint,
    SumToOneConstraint,
)
from cvxpy_portfolio_optimizer.objective_function import ObjectiveFunction
from cvxpy_portfolio_optimizer.portfolio_optimization_problem import PortfolioOptimizationProblem


class EfficientFrontier:
    """Efficient frontier class."""

    def __init__(
        self,
        returns: pd.DataFrame,
        objective_function: ObjectiveFunction,
    ):
        """Initialize efficient frontier class."""
        self.returns = returns
        self.objective_function = objective_function
        self._default_constraints = [
            SumToOneConstraint(),
            NoShortSellConstraint(),
        ]
        self.pop = PortfolioOptimizationProblem(
            returns=self.returns,
            objective_functions=[self.objective_function],
            constraint_functions=self._default_constraints,
        )

    def compute_efficient_frontier(
        self,
        npoints: int = 10,
        weights_tolerance: float | None = 1e-6,
        **kwargs: Any,
    ) -> list[EfficientFrontierResponse]:
        """Compute efficient frontier.

        Parameters
        ----------
        npoints : int, optional
            Number of points of the frontier to compute, by default 10.
        weights_tolerance : float, optional
            An optional float, if provided the weights resulting smaller then weights_tolerance
            after an optimization will be set to 0.
        kwargs : Any
            All the supported params of cvxpy.problems.problem.Problem.solve().


        Returns
        -------
        list[EfficientFrontierResponse]
            List of efficient frontier responses.
        """
        ef_responses = []
        # Compute expected returns vector, will be useful later
        expected_returns = self.returns.mean()
        # Compute minimum risk portfolio
        min_risk_ptf = self.pop.solve(weights_tolerance=weights_tolerance, **kwargs)
        min_risk_ptf_expected_return = expected_returns @ min_risk_ptf.weights
        min_risk = min_risk_ptf.get_objective_value(self.objective_function.name)
        ef_responses.append(
            EfficientFrontierResponse(
                risk=min_risk,
                expected_return=min_risk_ptf_expected_return,
                optimal_portfolio_weights=min_risk_ptf.weights,
            )
        )
        # Compute maximum return portfolio have the target expected returns
        # from min risk portfolio expected return to the maximum expected return of the instrument
        max_ret_instr = expected_returns.max()
        target_exp_rets = np.linspace(min_risk_ptf_expected_return, max_ret_instr, npoints)[1:]
        # Compute efficient frontier
        for exp_ret in target_exp_rets:
            pop = PortfolioOptimizationProblem(
                returns=self.returns,
                objective_functions=[self.objective_function],
                constraint_functions=[
                    *self._default_constraints,
                    ExpectedReturnConstraint(lower_bound=exp_ret, upper_bound=exp_ret),
                ],
            )
            ptf = pop.solve(weights_tolerance=weights_tolerance, **kwargs)
            ef_responses.append(
                EfficientFrontierResponse(
                    risk=ptf.get_objective_value(self.objective_function.name),
                    expected_return=expected_returns @ ptf.weights,
                    optimal_portfolio_weights=ptf.weights,
                )
            )
        return ef_responses

    def plot_efficient_frontier(
        self,
        npoints: int = 10,
        weights_tolerance: float | None = 1e-6,
        ef_responses: list[EfficientFrontierResponse] | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot efficient frontier.

        Parameters
        ----------
        npoints : int, optional
            Number of points of the frontier to compute, by default 10.
        weights_tolerance : float, optional
            An optional float, if provided the weights resulting smaller then weights_tolerance
            after an optimization will be set to 0.
        ef_responses : list[EfficientFrontierResponse]
            List of efficient frontier responses.
        kwargs : Any
            All the supported params of cvxpy.problems.problem.Problem.solve().
        """
        ef_responses = ef_responses or self.compute_efficient_frontier(
            npoints, weights_tolerance, **kwargs
        )
        plt.figure(figsize=(12, 8))
        plt.plot([ef.risk for ef in ef_responses], [ef.expected_return for ef in ef_responses])
        plt.title("Efficient Frontier")
        plt.xlabel(self.objective_function.name.value)  # type: ignore[union-attr]
        plt.ylabel("Expected Return")
        plt.show()
        # Show portfolios weights
        portfolios_weights = pd.DataFrame(
            [ef.optimal_portfolio_weights for ef in ef_responses],
            columns=self.returns.columns,
        )
        portfolios_weights.plot.bar(stacked=True, figsize=(12, 8))
        plt.title("Portfolios Weights")
        plt.xlabel("Efficient Frontier Index (from min risk to max return)")
        plt.ylabel("Weights")
        plt.show()
