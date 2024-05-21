"""Portfolio optimization problem module."""

from typing import Any

import cvxpy as cp
import pandas as pd

from cvxpy_portfolio_optimizer._models import ObjectiveModel
from cvxpy_portfolio_optimizer.constraint_function import (
    ConstraintFunction,
    WeightsToBoundConstraint,
)
from cvxpy_portfolio_optimizer.objective_function import ObjectiveFunction
from cvxpy_portfolio_optimizer.portfolio import Portfolio


class PortfolioOptimizationProblem:
    """Portfolio optimization problem."""

    def __init__(
        self,
        returns: pd.DataFrame,
        objective_functions: list[ObjectiveFunction],
        constraint_functions: list[ConstraintFunction],
    ) -> None:
        if returns.isna().sum().sum():
            raise AssertionError("Passed `returns` contains NaN.")
        self.returns = returns
        self.objective_functions = objective_functions
        self.constraint_functions = constraint_functions
        self._universe = list(self.returns.columns)

    def solve(self, weights_tolerance: float | None = 1e-6, **kwargs: Any) -> Portfolio:
        """Solve a portfolio optimization problem.

        Parameters
        ----------
        weights_tolerance
            An optional float, if provided the weights resulting smaller then weights_tolerance
            after an optimization will be set to 0.
        kwargs
            All the supported params of cvxpy.problems.problem.Problem.solve().

        Returns
        -------
        Portfolio
            The portfolio object containing the weights and the objective values.
        """
        self._check_constraints_applicability()
        weights_var = cp.Variable(len(self._universe))
        cvxpy_objectives, cvxpy_constraints = self._get_cvxpy_objectives_and_constraints(
            weights_var
        )
        problem = cp.Problem(
            objective=cp.sum([obj_model.function for obj_model in cvxpy_objectives]),
            constraints=cvxpy_constraints,
        )
        problem.solve(**kwargs)
        if problem.status != "optimal":
            raise AssertionError(f"Problem status is not optimal but: {problem.status}")
        weights_series = pd.Series(dict(zip(self._universe, weights_var.value, strict=True)))
        if weights_tolerance is not None:
            weights_series[abs(weights_series) < weights_tolerance] = 0.0
        return Portfolio(
            weights=weights_series,
            objectives=cvxpy_objectives,
            returns_data=self.returns,
        )

    def _get_cvxpy_objectives_and_constraints(
        self, weights_variable: cp.Variable
    ) -> tuple[list[ObjectiveModel], list[cp.Constraint]]:
        """Get portfolio optimization problem."""
        assert (
            self.objective_functions
        ), "To get a portfolio optimization problem, at least one objective is needed."
        cvxpy_objectives: list[ObjectiveModel] = []
        cvxpy_constraints: list[cp.Constraint] = []
        for obj_fun in self.objective_functions:
            objective, constr_list = obj_fun.get_objective_and_auxiliary_constraints(
                returns=self.returns,
                weights_variable=weights_variable,
            )
            cvxpy_objectives.append(objective)
            cvxpy_constraints.extend(constr_list)
        for constr_fun in self.constraint_functions:
            cvxpy_constraints.extend(
                constr_fun.get_constraints_list(
                    returns=self.returns,
                    weights_variable=weights_variable,
                    universe=self._universe,
                )
            )
        return cvxpy_objectives, cvxpy_constraints

    def _check_constraints_applicability(self):
        """Check if the constraints are applicable to the universe."""
        for constr in self.constraint_functions:
            if isinstance(constr, WeightsToBoundConstraint):
                if isinstance(constr.lower_bound, dict):
                    for instr in constr.lower_bound:
                        if instr not in self._universe:
                            raise AssertionError(
                                f"Instrument: {instr} in lower_bound not in universe."
                            )
                if isinstance(constr.upper_bound, dict):
                    for instr in constr.upper_bound:
                        if instr not in self._universe:
                            raise AssertionError(
                                f"Instrument: {instr} in upper_bound not in universe."
                            )
