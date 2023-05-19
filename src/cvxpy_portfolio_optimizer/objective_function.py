"""Objectives."""
from abc import ABCMeta, abstractmethod

import cvxpy as cp
import pandas as pd
from cvxpy.problems.objective import Objective

from cvxpy_portfolio_optimizer._enums import OptimizationVariableName


class ObjectiveFunction(metaclass=ABCMeta):
    """Objective function abstract class."""

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    @abstractmethod
    def get_matrices(
        self,
        returns: pd.DataFrame,
        weights_variable: cp.Variable,
    ) -> tuple[Objective, list[cp.Constraint]]:
        """Get optimization matrices."""


class CVaRObjectiveFunction(ObjectiveFunction):
    """CVaR objective function."""

    def __init__(
        self,
        confidence_level: float,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight=weight)
        self.confidence_level = confidence_level

    def get_matrices(
        self,
        returns: pd.DataFrame,
        weights_variable: cp.Variable,
    ) -> tuple[cp.Minimize, list[cp.Constraint]]:
        """Get CVaR optimization matrices."""
        rets = returns.values
        n_obs = rets.shape[0]
        cvar_devs = cp.Variable(n_obs, name=OptimizationVariableName.LOSSES_MINUS_VAR, nonneg=True)
        value_at_risk = cp.Variable(1, name=OptimizationVariableName.VAR)
        objective_function = value_at_risk + 1 / ((1 - self.confidence_level) * n_obs) * cp.sum(
            cvar_devs
        )
        return cp.Minimize(self.weight * objective_function), [
            -rets @ weights_variable - value_at_risk - cvar_devs <= 0
        ]


class VarianceObjectiveFunction(ObjectiveFunction):
    """Variance objective function."""

    def __init__(
        self,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight=weight)

    def get_matrices(  # type: ignore[override]
        self,
        returns: pd.DataFrame,
        weights_variable: cp.Variable,
    ) -> tuple[cp.Minimize, list[cp.Constraint]]:
        """Get Variance optimization matrices."""
        objective_function = weights_variable @ returns.cov().values @ weights_variable
        return cp.Minimize(self.weight * objective_function), []
