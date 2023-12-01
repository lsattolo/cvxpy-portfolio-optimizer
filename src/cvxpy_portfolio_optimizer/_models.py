"""Models."""

import pandas as pd
from pydantic import BaseModel


class EfficientFrontierResponse(BaseModel):
    """Efficient frontier response model."""

    risk: float
    expected_return: float
    optimal_portfolio_weights: pd.Series

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        extra = "forbid"
