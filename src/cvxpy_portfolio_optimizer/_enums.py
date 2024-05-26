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


class RebalanceFrequency(str, Enum):
    """How often the portfolio is rebalanced."""

    DAILY = "B"
    WEEKLY_MON = "W-MON"
    WEEKLY_FRI = "W-FRI"
    ONE_MONTH = "BMS"
    TWO_MONTHS = "2BMS"
    THREE_MONTHS = "3BMS"
    QUARTER_START = "BQS"
    QUARTER_END = "BQ"
    YEAR_END = "BYE"
    YEAR_START = "BYS"
