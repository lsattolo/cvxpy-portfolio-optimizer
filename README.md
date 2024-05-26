[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/lsattolo/cvxpy-portfolio-optimizer)

# cvxpy-portfolio-optimizer

Python package that uses cvxpy to perform multi objective portoflio optimization problems.
Backtesting included

## Portfolio Optimization Example
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

## Backtesting Example
```python3
from cvxpy_portfolio_optimizer._enums import RebalanceFrequency
from cvxpy_portfolio_optimizer.backtester import Backtester
from cvxpy_portfolio_optimizer.constraint_function import (
    NoShortSellConstraint,
    SumToOneConstraint,
)
from cvxpy_portfolio_optimizer.objective_function import VarianceObjectiveFunction
import yfinance as yf

tickers = ["TSLA", "MSFT", "IBM", "GOOG", "AAPL", "AMZN", "ADBE"]
rets = (
    yf.download(tickers, period="2y")["Adj Close"].loc[:, tickers].pct_change().iloc[1:, :].ffill()
)
# Create a backtester object and run backtest
bt = Backtester(
    learning_days=365,
    rebalance_frequency=RebalanceFrequency.ONE_MONTH,
    returns=rets,
    objective_functions=[VarianceObjectiveFunction()],
    constraint_functions=[SumToOneConstraint(), NoShortSellConstraint()],
)
bt_out = bt.run()

print(bt_out.portfolio_returns)
# Date
# 2023-06-01    0.010153
# 2023-06-02    0.019196
# 2023-06-05    0.002466
# 2023-06-06    0.000606
# 2023-06-07    0.007930
#                 ...   
# 2024-05-17    0.000572
# 2024-05-20    0.006491
# 2024-05-21    0.012641
# 2024-05-22   -0.002076
# 2024-05-23   -0.016943
# Length: 247, dtype: float64

# It's also possible to access all the rebalanced portfolios
print(bt_out.portfolios)
#{Timestamp('2023-06-01 00:00:00'): <cvxpy_portfolio_optimizer.portfolio.Portfolio at 0x1469013f0>,
# Timestamp('2023-07-03 00:00:00'): <cvxpy_portfolio_optimizer.portfolio.Portfolio at 0x1468d73d0>,
# Timestamp('2023-08-01 00:00:00'): <cvxpy_portfolio_optimizer.portfolio.Portfolio at 0x1469f8970>,
# Timestamp('2023-09-01 00:00:00'): <cvxpy_portfolio_optimizer.portfolio.Portfolio at 0x1469f8b50>,
# Timestamp('2023-10-02 00:00:00'): <cvxpy_portfolio_optimizer.portfolio.Portfolio at 0x1469f9ed0>,
# Timestamp('2023-11-01 00:00:00'): <cvxpy_portfolio_optimizer.portfolio.Portfolio at 0x1469fa860>,
# Timestamp('2023-12-01 00:00:00'): <cvxpy_portfolio_optimizer.portfolio.Portfolio at 0x1469fb1f0>,
# Timestamp('2024-01-01 00:00:00'): <cvxpy_portfolio_optimizer.portfolio.Portfolio at 0x1469fb730>,
# Timestamp('2024-02-01 00:00:00'): <cvxpy_portfolio_optimizer.portfolio.Portfolio at 0x146ad4580>,
# Timestamp('2024-03-01 00:00:00'): <cvxpy_portfolio_optimizer.portfolio.Portfolio at 0x146ad4ee0>,
# Timestamp('2024-04-01 00:00:00'): <cvxpy_portfolio_optimizer.portfolio.Portfolio at 0x146ad5870>,
# Timestamp('2024-05-01 00:00:00'): <cvxpy_portfolio_optimizer.portfolio.Portfolio at 0x146ad6200>}
```