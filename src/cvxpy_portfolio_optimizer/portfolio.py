"""Portfolio module."""

import pandas as pd

from cvxpy_portfolio_optimizer._enums import ObjectiveFunctionName
from cvxpy_portfolio_optimizer._models import ObjectiveModel


class Portfolio:
    """Portfolio class."""

    def __init__(
        self,
        weights: pd.Series,
        objectives: list[ObjectiveModel],
        returns_data: pd.DataFrame,
    ) -> None:
        self.weights = weights
        self.objectives = objectives
        self.returns_data = returns_data

    def portfolio_timeseries(
        self,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.Series:
        """Get portfolio timeseries."""
        start_date = start_date or self.returns_data.index[0]
        end_date = end_date or self.returns_data.index[-1]
        ptf_rets = self.returns_data.loc[start_date:end_date] @ self.weights
        return (1 + ptf_rets).cumprod()

    def get_total_objective_value(self) -> float:
        """Get total objective value."""
        return sum(obj.function.value for obj in self.objectives)

    def get_objective_value(self, name: ObjectiveFunctionName | str) -> float:
        """Get objective value."""
        # We can have multiple objectives with the same name,
        # in case there are we throw an error
        obj_values = [obj.function.value for obj in self.objectives if obj.name == name]
        if len(obj_values) > 1:
            raise ValueError(
                f"Multiple objectives with the same name: {name}."
                "When defining the optimization problem, provide unique names for each objective."
            )
        return obj_values[0]
