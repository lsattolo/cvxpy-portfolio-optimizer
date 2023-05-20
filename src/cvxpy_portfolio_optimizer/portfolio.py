"""Portfolio module."""
import pandas as pd

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
