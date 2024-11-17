import numpy as np
import pandas as pd
from stats_utils import corr, std
from tqdm import tqdm


def get_alphas(ICs, thetas, omegas, weights, permcos, method="cross_sec", seed=None):
    if seed is not None:
        np.random.seed(seed)

    all_permcos = np.unique(np.concatenate(list(permcos.values())))
    alphas = pd.DataFrame(index=ICs.index, columns=all_permcos, dtype=float)

    for t in tqdm(ICs.index):
        permcos_t = permcos[t]
        IC_t = ICs.loc[t]
        thetas_t = thetas.loc[t, permcos_t].values
        weights_t = weights.loc[t, permcos_t].values

        if method == "cross_sec":
            omega_t = omegas.loc[t]
            alphas.loc[t, permcos_t] = construct_alpha_cross_sec(
                IC_t, thetas_t, omega_t, weights_t
            )
        elif method == "time_series":
            omega_t = omegas.loc[t, permcos_t].values
            alphas.loc[t, permcos_t] = construct_alpha_time_series(
                IC_t, thetas_t, omega_t, weights_t
            )

    return alphas


def get_ICs(returns, thetas, weights, permcos, dates):
    """
    Parameters
    ----------
    returns : DataFrame
        Returns
    thetas : DataFrame
        Realized returns
    weights : DataFrame
        Weights
    permcos : dict
        Permcos

    Returns
    -------
    Series
        Information Coefficients
    """

    ICs = pd.Series(index=returns.index, dtype=float)

    for t in tqdm(dates):
        permcos_t = permcos[t]
        returns_t = returns.loc[t, permcos_t].values
        thetas_t = thetas.loc[t, permcos_t].values
        weights_t = weights.loc[t, permcos_t].values

        ICs.loc[t] = corr(returns_t, thetas_t, weights_t)

    return ICs


def construct_alpha_cross_sec(IC, thetas, omega, weights):
    """
    Parameters
    ----------
    IC : float
        Information Coefficient
    thetas : array
        Realized returns
    omega : float
        Cross-sectional volatility
    weights : array
        Statistical weights (typically inverse variance)

    Returns
    -------
    array
        Alphas (return predictions)
    """

    n_assets = len(thetas)
    s = np.random.normal(size=n_assets)
    z = s * np.sqrt(weights.sum() / n_assets) * 1 / np.sqrt(weights)

    return IC * (IC * thetas + omega * np.sqrt(1 - IC**2) * z)


def construct_alpha_time_series(IC, thetas, omegas, weights):
    """
    Parameters
    ----------
    IC : float
        Information Coefficient
    theta : array
        Realized returns
    omega : array
        Time-series volatilities
    weights : array
        Statistical weights (typically inverse variance)

    Returns
    -------
    array
        Alphas (return predictions)
    """

    n_assets = len(thetas)
    z = np.random.normal(size=n_assets)

    return IC * (IC * thetas + omegas * np.sqrt(1 - IC**2) * z)


def get_volas_cross_sec(returns, weights, permcos):
    """
    Parameters
    ----------
    returns : DataFrame
        Returns
    weights : DataFrame
        Weights
    permcos : dict
        Permcos

    Returns
    -------
    array
        Cross-sectional volatilities
    """

    volas = pd.Series(index=list(permcos.keys()), dtype=float)

    for t in tqdm(permcos.keys()):
        permcos_t = permcos[t]
        returns_t = returns.loc[t, permcos_t].values
        weights_t = weights.loc[t, permcos_t].values

        volas.loc[t] = std(returns_t, weights_t)

    return volas


def get_volas_cross_sec2(omegas, weights, permcos):
    """
    Parameters
    ----------
    omegas : DataFrame
        Idiosyncratic volatilities
    weights : DataFrame
        Weights
    permcos : dict
        Permcos

    Returns
    -------
    array
        Cross-sectional volatilities
    """

    volas = pd.Series(index=list(permcos.keys()), dtype=float)

    for t in tqdm(permcos.keys()):
        permcos_t = permcos[t]
        omegas_t = omegas.loc[t, permcos_t].values
        weights_t = weights.loc[t, permcos_t].values

        volas.loc[t] = ((omegas_t**2 * weights_t).sum() / weights_t.sum()) ** 0.5

    return volas


def get_acf(alphas, weights, lags=21):
    """
    Parameters
    ----------
    alphas : DataFrame
        Alphas
    weights : DataFrame
        Weights
    lags : int
        Number of lags

    Returns
    -------
    Series
        Autocorrelations
    """

    _weights = weights * alphas.notna()

    acfs = pd.Series(index=range(lags + 1), dtype=float)

    for lag in tqdm(range(lags + 1)):
        alphas_shifted = alphas.shift(lag)

        cov = (alphas * alphas_shifted * _weights).sum().sum() / _weights.sum().sum()
        var1 = (alphas**2 * _weights).sum().sum() / _weights.sum().sum()
        var2 = (alphas_shifted**2 * _weights).sum().sum() / _weights.sum().sum()

        acfs.loc[lag] = cov / (var1 * var2) ** 0.5

    return acfs
