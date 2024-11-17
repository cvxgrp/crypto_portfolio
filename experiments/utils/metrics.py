from functools import reduce

import numpy as np
import pandas as pd


class Metrics:
    def __init__(
        self, weights: pd.DataFrame, cash: pd.Series, realied_returns: pd.DataFrame
    ):
        times = reduce(
            np.intersect1d, [weights.index, cash.index, realied_returns.index]
        )

        self._times = times
        self._weights = weights.loc[times]
        self._cash = cash.loc[times]
        self._realied_returns = realied_returns.loc[times]
        self.summary = get_summary(self)

    @property
    def weights(self):
        return self._weights

    @property
    def cash(self):
        return self._cash

    @property
    def realized_returns(self):
        return self._realied_returns

    @property
    def portfolio_returns(self):
        costs = (
            pd.Series(
                np.abs(np.diff(self._weights, axis=0)).sum(axis=1),
                index=self._times[1:],
            )
            * 5
            * 0.01**2
        )
        return (self._weights * self._realied_returns).sum(axis=1) - costs

    @property
    def mean_return(self):
        return self.portfolio_returns.mean() * 250

    @property
    def times(self):
        return self._times

    @property
    def volatility(self):
        return self.portfolio_returns.std() * np.sqrt(250)

    @property
    def sharpe(self):
        return self.mean_return / self.volatility

    @property
    def portfolio_value(self):
        return (self.portfolio_returns + 1).cumprod()

    @property
    def drawdown(self):
        return 1 - self.portfolio_value / self.portfolio_value.cummax()

    @property
    def max_drawdown(self):
        return self.drawdown.max()


def get_summary(metrics: Metrics):
    summary = {
        "Mean return": metrics.mean_return,
        "Volatility": metrics.volatility,
        "Sharpe ratio": metrics.sharpe,
        "Max drawdown": metrics.max_drawdown,
    }

    return pd.Series(summary)
