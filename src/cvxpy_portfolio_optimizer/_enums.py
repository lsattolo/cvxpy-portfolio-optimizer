"""Enums."""
from enum import Enum


class OptimizationVariableName(str, Enum):
    """Optimization variable name."""

    WEIGHTS = "WEIGHTS"
    LOSSES_MINUS_VAR = "LOSSES_MINUS_VAR"
    VAR = "VAR"


class ObjectiveFunctionName(str, Enum):
    """Objective function name."""

    CVAR = "CVaR"
    VARIANCE = "VARIANCE"
    EXPECTED_RETURNS = "EXPECTED_RETURNS"
