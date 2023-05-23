"""Enums."""
from enum import Enum


class ObjectiveFunctionName(str, Enum):
    """Objective function name."""

    CVAR = "CVaR"
    VARIANCE = "VARIANCE"
    EXPECTED_RETURNS = "EXPECTED_RETURNS"
    MEAN_ABSOLUTE_DEVIATION = "MEAN_ABSOLUTE_DEVIATION"
    TRACKING_ERROR = "TRACKING_ERROR"
    WORST_REALIZATION = "WORST_REALIZATION"
