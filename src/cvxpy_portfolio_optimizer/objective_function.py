"""Objectives."""

from abc import ABCMeta, abstractmethod

import cvxpy as cp
import pandas as pd

from cvxpy_portfolio_optimizer._enums import ObjectiveFunctionName
from cvxpy_portfolio_optimizer._models import ObjectiveModel


class ObjectiveFunction(metaclass=ABCMeta):
    """Objective function abstract class."""

    def __init__(
        self,
        weight: float = 1.0,
        name: ObjectiveFunctionName | str = "",
    ) -> None:
        self.weight = weight
        self.name = name

    @abstractmethod
    def get_objective_and_auxiliary_constraints(
        self,
        returns: pd.DataFrame,
        weights_variable: cp.Variable,
    ) -> tuple[ObjectiveModel, list[cp.Constraint]]:
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

    def get_objective_and_auxiliary_constraints(
        self,
        returns: pd.DataFrame,
        weights_variable: cp.Variable,
    ) -> tuple[ObjectiveModel, list[cp.Constraint]]:
        """Get CVaR optimization matrices."""
        rets_vals = returns.values
        n_obs = rets_vals.shape[0]
        losses_minus_var = cp.Variable(n_obs, nonneg=True)
        value_at_risk = cp.Variable(1)
        objective_function = value_at_risk + 1 / ((1 - self.confidence_level) * n_obs) * cp.sum(
            losses_minus_var
        )
        return ObjectiveModel(
            name=self.name, function=cp.Minimize(self.weight * objective_function)
        ), [-rets_vals @ weights_variable - value_at_risk - losses_minus_var <= 0]


class VarianceObjectiveFunction(ObjectiveFunction):
    """Variance objective function."""

    def __init__(
        self,
        weight: float = 1.0,
        name: ObjectiveFunctionName | str = ObjectiveFunctionName.VARIANCE,
    ) -> None:
        super().__init__(weight=weight, name=name)

    def get_objective_and_auxiliary_constraints(
        self,
        returns: pd.DataFrame,
        weights_variable: cp.Variable,
    ) -> tuple[ObjectiveModel, list[cp.Constraint]]:
        """Get Variance optimization matrices."""
        # Annualize the cov mat
        sigma = 252 * returns.cov().values
        objective_function = weights_variable @ sigma @ weights_variable
        return (
            ObjectiveModel(name=self.name, function=cp.Minimize(self.weight * objective_function)),
            [],
        )


class ExpectedReturnsObjectiveFunction(ObjectiveFunction):
    """Expected Returns objective function."""

    def __init__(
        self,
        weight: float = 0.25,
        name: ObjectiveFunctionName | str = ObjectiveFunctionName.EXPECTED_RETURNS,
    ) -> None:
        super().__init__(weight=weight, name=name)

    def get_objective_and_auxiliary_constraints(
        self,
        returns: pd.DataFrame,
        weights_variable: cp.Variable,
    ) -> tuple[ObjectiveModel, list[cp.Constraint]]:
        """Get Expected Returns optimization matrices."""
        # Annualize expected returns and put minus in front to maximize
        exp_rets = -252 * returns.mean().values
        objective_function = weights_variable @ exp_rets
        return (
            ObjectiveModel(name=self.name, function=cp.Minimize(self.weight * objective_function)),
            [],
        )


class MADObjectiveFunction(ObjectiveFunction):
    """Mean Absolute Deviation objective function."""

    def __init__(
        self,
        weight: float = 1.0,
        name: ObjectiveFunctionName | str = ObjectiveFunctionName.MEAN_ABSOLUTE_DEVIATION,
    ) -> None:
        super().__init__(weight=weight, name=name)

    def get_objective_and_auxiliary_constraints(
        self,
        returns: pd.DataFrame,
        weights_variable: cp.Variable,
    ) -> tuple[ObjectiveModel, list[cp.Constraint]]:
        """Get Mean Absolute Deviation optimization matrices."""
        rets_vals = returns.values
        n_obs = rets_vals.shape[0]
        abs_devs = cp.Variable(n_obs, nonneg=True)
        rets_minus_mu = rets_vals - returns.mean().values
        objective_function = cp.sum(abs_devs) / n_obs
        return ObjectiveModel(
            name=self.name, function=cp.Minimize(self.weight * objective_function)
        ), [
            rets_minus_mu @ weights_variable - abs_devs <= 0.0,
            -rets_minus_mu @ weights_variable - abs_devs <= 0.0,
        ]


class TrackingErrorObjectiveFunction(ObjectiveFunction):
    """TrackingError objective function."""

    def __init__(
        self,
        benchmark_returns: pd.Series,
        weight: float = 1.0,
        name: ObjectiveFunctionName | str = ObjectiveFunctionName.TRACKING_ERROR,
    ) -> None:
        super().__init__(weight=weight, name=name)
        self.benchmark_returns = benchmark_returns

    def get_objective_and_auxiliary_constraints(
        self,
        returns: pd.DataFrame,
        weights_variable: cp.Variable,
    ) -> tuple[ObjectiveModel, list[cp.Constraint]]:
        """Get TrackingError optimization matrices."""
        ret_vals = returns.values
        assert ret_vals.shape[0] == len(
            self.benchmark_returns
        ), "Number of observation of the universe returns are not the same as the benchmark returns to track."
        qmat = (ret_vals).T.dot(ret_vals)
        linvector = -2 * self.benchmark_returns.values.dot(ret_vals)
        objective_function = cp.sum(
            [
                cp.Minimize(weights_variable @ qmat @ weights_variable),
                cp.Minimize(weights_variable @ linvector),
            ]
        )
        return ObjectiveModel(name=self.name, function=objective_function * self.weight), []


class WorstRealizationObjectiveFunction(ObjectiveFunction):
    """WorstRealization objective function."""

    def __init__(
        self,
        weight: float = 1.0,
        name: ObjectiveFunctionName | str = ObjectiveFunctionName.WORST_REALIZATION,
    ) -> None:
        super().__init__(weight=weight, name=name)

    def get_objective_and_auxiliary_constraints(
        self,
        returns: pd.DataFrame,
        weights_variable: cp.Variable,
    ) -> tuple[ObjectiveModel, list[cp.Constraint]]:
        """Get WorstRealization optimization matrices."""
        ret_vals = returns.values
        max_variable = cp.Variable(1)
        objective_function = cp.Minimize(-max_variable)
        return ObjectiveModel(name=self.name, function=objective_function * self.weight), [
            max_variable - ret_vals @ weights_variable <= 0.0
        ]
