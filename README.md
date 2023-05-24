[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/lsattolo/cvxpy-portfolio-optimizer)

# cvxpy-portfolio-optimizer

Python package that uses cvxpy to perform multi objective portoflio optimization problems.

## Example
```python3
import yfinance as yf

from cvxpy_portfolio_optimizer.constraint_function import (
    NoShortSellConstraint,
    NumberOfAssetsConstraint,
    SumToOneConstraint,
)
from cvxpy_portfolio_optimizer.objective_function import (
    CVaRObjectiveFunction,
    ExpectedReturnsObjectiveFunction,
    TrackingErrorObjectiveFunction,
    VarianceObjectiveFunction,
)
from cvxpy_portfolio_optimizer.portfolio_optimization_problem import PortfolioOptimizationProblem

tickers = ["TSLA", "MSFT", "IBM", "GOOG", "AAPL", "AMZN", "ADBE"]
rets = yf.download(tickers, period="1y")["Adj Close"].loc[:, tickers].pct_change().iloc[1:, :].ffill()
pop = PortfolioOptimizationProblem(
    returns=rets,
    objective_functions=[
        CVaRObjectiveFunction(confidence_level=0.95, weight=1.0), 
        VarianceObjectiveFunction(weight=1.0),
        ExpectedReturnsObjectiveFunction(weight=0.3)
    ],
    constraint_functions=[
        SumToOneConstraint(),
        NoShortSellConstraint(),
    ]
)
ptf = pop.solve(solver="ECOS")

print(ptf.weights)
# TSLA    0.000000
# MSFT    0.338102
# IBM     0.479004
# GOOG    0.000000
# AAPL    0.182895
# AMZN    0.000000
# ADBE    0.000000
# dtype: float64

print(ptf.objective_values)
# [{<ObjectiveFunctionName.CVAR: 'CVaR'>: 0.029028499211609417},
#  {<ObjectiveFunctionName.VARIANCE: 'VARIANCE'>: 0.04999096599731992},
#  {<ObjectiveFunctionName.EXPECTED_RETURNS: 'EXPECTED_RETURNS'>: -0.04792116480264919}]
```
