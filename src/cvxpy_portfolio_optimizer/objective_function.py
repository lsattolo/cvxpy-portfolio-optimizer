"""Objectives."""
from abc import ABCMeta, abstractmethod

import cvxpy as cp
import pandas as pd

from cvxpy_portfolio_optimizer._enums import ObjectiveFunctionName, OptimizationVariableName


class ObjectiveFunction(metaclass=ABCMeta):
    """Objective function abstract class."""

    def __init__(self, weight: float = 1.0, name: ObjectiveFunctionName | str = "") -> None:
        self.weight = weight
        self.name = name

    @abstractmethod
    def get_matrices(
        self,
        returns: pd.DataFrame,
        weights_variable: cp.Variable,
    ) -> tuple[dict[ObjectiveFunctionName | str, cp.Minimize], list[cp.Constraint]]:
        """Get optimization matrices."""


class CVaRObjectiveFunction(ObjectiveFunction):
    """CVaR objective function."""

    def __init__(
        self,
        confidence_level: float,
        weight: float = 1.0,
        name: ObjectiveFunctionName | str = ObjectiveFunctionName.CVAR,
    ) -> None:
        super().__init__(weight=weight, name=name)
        self.confidence_level = confidence_level

    def get_matrices(
        self,
        returns: pd.DataFrame,
        weights_variable: cp.Variable,
    ) -> tuple[dict[ObjectiveFunctionName | str, cp.Minimize], list[cp.Constraint]]:
        """Get CVaR optimization matrices."""
        rets = returns.values
        n_obs = rets.shape[0]
        cvar_devs = cp.Variable(n_obs, name=OptimizationVariableName.LOSSES_MINUS_VAR, nonneg=True)
        value_at_risk = cp.Variable(1, name=OptimizationVariableName.VAR)
        objective_function = value_at_risk + 1 / ((1 - self.confidence_level) * n_obs) * cp.sum(
            cvar_devs
        )
        return {self.name: cp.Minimize(self.weight * objective_function)}, [
            -rets @ weights_variable - value_at_risk - cvar_devs <= 0
        ]


class VarianceObjectiveFunction(ObjectiveFunction):
    """Variance objective function."""

    def __init__(
        self,
        weight: float = 1.0,
        name: ObjectiveFunctionName | str = ObjectiveFunctionName.VARIANCE,
    ) -> None:
        super().__init__(weight=weight, name=name)

    def get_matrices(
        self,
        returns: pd.DataFrame,
        weights_variable: cp.Variable,
    ) -> tuple[dict[ObjectiveFunctionName | str, cp.Minimize], list[cp.Constraint]]:
        """Get Variance optimization matrices."""
        # Annualize the cov mat
        sigma = 252 * returns.cov().values
        objective_function = weights_variable @ sigma @ weights_variable
        return {self.name: cp.Minimize(self.weight * objective_function)}, []


class ExpectedReturnsbjectiveFunction(ObjectiveFunction):
    """Expected Returns objective function."""

    def __init__(
        self,
        weight: float = 0.25,
        name: ObjectiveFunctionName | str = ObjectiveFunctionName.EXPECTED_RETURNS,
    ) -> None:
        super().__init__(weight=weight, name=name)

    def get_matrices(
        self,
        returns: pd.DataFrame,
        weights_variable: cp.Variable,
    ) -> tuple[dict[ObjectiveFunctionName | str, cp.Minimize], list[cp.Constraint]]:
        """Get Expected Returns optimization matrices."""
        # Annualize expected returns and put minus in front to maximize
        exp_rets = -252 * returns.mean().values
        objective_function = weights_variable @ exp_rets
        return {self.name: cp.Minimize(self.weight * objective_function)}, []
