"""Models."""

import cvxpy as cp
import pandas as pd
from pydantic import BaseModel, ConfigDict

from cvxpy_portfolio_optimizer._enums import ObjectiveFunctionName


class EfficientFrontierResponse(BaseModel):
    """Efficient frontier response model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    risk: float
    expected_return: float
    optimal_portfolio_weights: pd.Series


class ObjectiveModel(BaseModel):
    """Objective function model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: ObjectiveFunctionName | str
    function: cp.Minimize
