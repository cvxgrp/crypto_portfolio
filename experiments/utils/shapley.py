from collections import namedtuple

import numpy as np
import pandas as pd
from utils.backtest import ProblemData, run_backtest
from cvx.covariance.ewma import iterated_ewma
from utils.metrics import Metrics


def get_metrics(returns, n_crypto=0):
    if isinstance(returns, pd.Series):
        returns = pd.DataFrame(returns)

    # Estimate covariances
    iewma_pair = (63, 125)
    iterator = iterated_ewma(
        returns,
        vola_halflife=iewma_pair[0],
        cov_halflife=iewma_pair[1],
        min_periods_vola=21,
        min_periods_cov=63,
    )
    covariances = {iterate.time: iterate.covariance for iterate in iterator}

    choleskies = {
        time: pd.DataFrame(
            np.linalg.cholesky(cov), index=cov.index, columns=cov.columns
        )
        for time, cov in covariances.items()
    }

    # Define problem data
    problem_data = ProblemData(
        n_assets=returns.shape[1],
        n_crypto=n_crypto,
        gamma=None,
        w_min=None,
        w_max=0.1,
        risk_limit=0.1,
        leverage_limit=None,
        asset_names=returns.columns,
        method="risk_parity",
        risk_allocations=np.full(returns.shape[1], 1 / returns.shape[1]),
        dilute=True,
    )

    # Run backtest
    backtest = run_backtest(choleskies, problem_data, returns)

    # Get metrics
    times = pd.read_csv("cache/times.csv", index_col=1, parse_dates=True).index
    metrics = Metrics(
        backtest.weights.loc[times],
        backtest.cash.loc[times],
        returns.loc[times].shift(-1),
    )

    # Get Sharpe ratio
    return metrics


Lifts = namedtuple("Lifts", ["sharpe", "vola", "drawdown", "mean_return"])


def get_lifts(permutation):
    returns_BTC = pd.read_csv(
        "../data/BTC_returns.csv", index_col=0, parse_dates=True
    ).squeeze()
    returns_BTC.name = "BTC"
    returns_ETH = pd.read_csv(
        "../data/ETH_returns.csv", index_col=0, parse_dates=True
    ).squeeze()
    returns_ETH.name = "ETH"
    returns_industry = (
        pd.read_csv("../data/industry_returns.csv", index_col=0, parse_dates=True) / 100
    )
    returns = pd.concat([returns_BTC, returns_ETH, returns_industry], axis=1).loc[
        "2017-01-01":
    ]
    returns = returns.drop(columns=["Other"])

    map_to_assets = {
        0: ["BTC", "ETH"],
        1: [returns.columns[2]],
        2: [returns.columns[3]],
        3: [returns.columns[4]],
        4: [returns.columns[5]],
    }

    lifts_sharpe = pd.Series(dtype=float)
    lifts_vola = pd.Series(dtype=float)
    lifts_drawdown = pd.Series(dtype=float)
    lifts_return = pd.Series(dtype=float)

    comb = map_to_assets[permutation[0]]
    comb_new = comb

    sharpe_old = 0
    vola_old = 0
    drawdown_old = 0
    return_old = 0
    for i in range(len(permutation)):
        n_crypto = 0
        if "BTC" in comb:
            n_crypto += 1
        if "ETH" in comb:
            n_crypto += 1

        comb = sorted(
            comb, key=lambda x: returns.columns.get_loc(x)
        )  # Sort the assets in the combination
        metrics_new = get_metrics(returns[comb], n_crypto)

        if comb_new[-1] == "ETH":
            name = "Crypto"
        else:
            name = comb_new[-1]

        lifts_sharpe[name] = metrics_new.sharpe - sharpe_old
        lifts_vola[name] = metrics_new.volatility - vola_old
        lifts_drawdown[name] = metrics_new.max_drawdown - drawdown_old
        lifts_return[name] = metrics_new.mean_return - return_old

        sharpe_old = metrics_new.sharpe
        vola_old = metrics_new.volatility
        drawdown_old = metrics_new.max_drawdown
        return_old = metrics_new.mean_return

        if i < len(permutation) - 1:
            comb_new = map_to_assets[permutation[i + 1]]
            comb += comb_new

    return Lifts(lifts_sharpe, lifts_vola, lifts_drawdown, lifts_return)
