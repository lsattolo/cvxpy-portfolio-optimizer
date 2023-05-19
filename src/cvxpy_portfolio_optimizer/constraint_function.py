"""Objectives."""
from abc import ABCMeta, abstractmethod

import cvxpy as cp


class ConstraintFunction(metaclass=ABCMeta):
    """Objective function abstract class."""

    @abstractmethod
    def get_constraints_list(self, weights_variable: cp.Variable) -> list[cp.Constraint]:
        """Get optimization matrices."""


class NoShortSellConstraint(ConstraintFunction):
    """NoShortSell constraint."""

    def get_constraints_list(self, weights_variable: cp.Variable) -> list[cp.Constraint]:
        """Get no short sell constraint matrices."""
        return [weights_variable >= 0.0]


class SumToOneConstraint(ConstraintFunction):
    """NoShortSell constraint."""

    def get_constraints_list(self, weights_variable: cp.Variable) -> list[cp.Constraint]:
        """Get sum to one constraint matrices."""
        return [cp.sum(weights_variable) == 1.0]
