"""Enums."""
from enum import Enum


class OptimizationVariableName(str, Enum):
    """Optimization variable name."""

    WEIGHTS = "WEIGHTS"
    LOSSES_MINUS_VAR = "LOSSES_MINUS_VAR"
    VAR = "VAR"
