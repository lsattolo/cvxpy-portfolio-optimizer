"""Portfolio optimization problem module."""
import cvxpy as cp
import pandas as pd
from cvxpy.problems.objective import Objective

from cvxpy_portfolio_optimizer._enums import OptimizationVariableName
from cvxpy_portfolio_optimizer.constraint_function import (
    ConstraintFunction,
    NoShortSellConstraint,
    SumToOneConstraint,
)
from cvxpy_portfolio_optimizer.objective_function import ObjectiveFunction


class PortfolioOptimizationProblem:
    """Portfolio optimization problem."""

    def __init__(
        self,
        returns: pd.DataFrame,
        objective_functions: list[ObjectiveFunction] | None = None,
        constraint_functions: list[ConstraintFunction] | None = None,
    ) -> None:
        if returns.isna().sum().sum():
            raise AssertionError("`returns` contains NaN.")
        self.returns = returns
        self.objective_functions = objective_functions or []
        self.constraint_functions = constraint_functions or [
            NoShortSellConstraint(),
            SumToOneConstraint(),
        ]
        self._universe = list(self.returns.columns)

    def get_problem(self) -> cp.Problem:
        """Get portfolio optimization problem."""
        assert (
            self.objective_functions
        ), "To get a portfolio optimization problem, at least one objective is needed."
        weights_var = cp.Variable(
            len(self._universe), name=OptimizationVariableName.WEIGHTS, nonneg=True
        )
        objectives: list[Objective] = []
        constraints: list[cp.Constraint] = []
        for obj_fun in self.objective_functions:
            objective, constr_list = obj_fun.get_matrices(
                returns=self.returns, weights_variable=weights_var
            )
            objectives.append(objective)
            constraints.extend(constr_list)
        for constr_fun in self.constraint_functions:
            constraints.extend(constr_fun.get_constraints_list(weights_variable=weights_var))
        return cp.Problem(cp.sum(objectives), constraints)

    def solve(self, solver: str | None = None) -> pd.Series:
        """Solve a portfolio optimization problem."""
        problem = self.get_problem()
        problem.solve(solver=solver)
        if problem.status != "optimal":
            raise AssertionError(f"Problem status not optimal but: {problem.status}")
        weights_var = [v for v in problem.variables() if v.name() == "Weights"][0]
        return pd.Series(dict(zip(self._universe, weights_var.value, strict=True)))
