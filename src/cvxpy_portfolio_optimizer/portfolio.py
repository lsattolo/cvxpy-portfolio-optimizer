"""Portfolio module."""

import pandas as pd
import yfinance as yf

from cvxpy_portfolio_optimizer.objective_function import ObjectiveFunction


class Portfolio:
    """Portfolio class."""

    def __init__(
        self,
        weights: pd.Series,
        objective_values: list[dict[ObjectiveFunction | str, float]],
    ) -> None:
        self.weights = weights
        self.objective_values = objective_values

    def portfolio_timeseries(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
        """Get portfolio timeseries."""
        returns = (
            yf.download(
                tickers=list(self.weights.index), start=start_date, end=end_date, show_errors=False
            )["Adj Close"]
            .pct_change()
            .dropna()
        )
        ptf_rets = returns @ self.weights
        return (1 + ptf_rets).cumprod()
