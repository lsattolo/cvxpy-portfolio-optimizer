"""Backtester module."""

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from cvxpy_portfolio_optimizer._enums import RebalanceFrequency
from cvxpy_portfolio_optimizer.constraint_function import ConstraintFunction
from cvxpy_portfolio_optimizer.objective_function import ObjectiveFunction
from cvxpy_portfolio_optimizer.portfolio import Portfolio
from cvxpy_portfolio_optimizer.portfolio_optimization_problem import PortfolioOptimizationProblem


class BacktesterOut(BaseModel):
    """Backtester output model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    portfolios: dict[pd.Timestamp, Portfolio]
    portfolio_returns: pd.Series


class Backtester:
    """Backtester."""

    def __init__(  # noqa: PLR0913
        self,
        learning_days: int,
        rebalance_frequency: RebalanceFrequency,
        returns: pd.DataFrame,
        objective_functions: list[ObjectiveFunction],
        constraint_functions: list[ConstraintFunction],
    ) -> None:
        """Backtester."""
        if returns.isna().sum().sum():
            raise AssertionError("Passed `returns` contains NaN.")
        self.learning_days = learning_days
        self.returns = returns
        self.objective_functions = objective_functions
        self.constraint_functions = constraint_functions
        self.rebalance_dates = pd.date_range(
            start=self.returns.index[0] + pd.DateOffset(days=learning_days),
            end=self.returns.index[-1],
            freq=rebalance_frequency.value,
            normalize=True,
        )

    def run(self, **kwargs: Any) -> BacktesterOut:
        """Run the backtest.

        Parameters
        ----------
        kwargs
            All the supported params of cvxpy.problems.problem.Problem.solve().

        Returns
        -------
        BacktesterOut
            The backtest model results.
        """
        portfolios = {}
        ptf_returns_list = []
        for idx, date in enumerate(self.rebalance_dates):
            pop = PortfolioOptimizationProblem(
                returns=self.returns.loc[date - pd.DateOffset(days=self.learning_days) : date],
                objective_functions=self.objective_functions,
                constraint_functions=self.constraint_functions,
            )
            ptf = pop.solve(**kwargs)
            portfolios[date] = ptf
            next_date_or_end = (
                self.rebalance_dates[idx + 1]
                if idx + 1 < len(self.rebalance_dates)
                else self.returns.index[-1]
            )
            ptf_returns = (
                ptf.weights @ self.returns.loc[date : next_date_or_end - pd.Timedelta(days=1)].T
            )
            ptf_returns_list.append(ptf_returns)
        portfolio_returns = pd.concat(ptf_returns_list)
        return BacktesterOut(portfolios=portfolios, portfolio_returns=portfolio_returns)
