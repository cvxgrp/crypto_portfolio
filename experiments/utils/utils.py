import numpy as np
import pandas as pd
from scipy.stats import norm


def plot_gaussian(x, ax, legend=False):
    """
    Plots a 1D Gaussian distribution fit to the data x.
    """

    mu, sigma = norm.fit(x)

    x = np.linspace(np.min(x), np.max(x), 100)
    y = norm.pdf(x, mu, sigma)

    ax.plot(x, y, label="Gaussian fit")
    if legend:
        ax.legend()

    return ax


def plot_histogram(x, ax, bins=50, gaussian_fit=True, legend=False):
    """
    Plots a histogram of the data x.
    """

    x.hist(bins=bins, ax=ax, density=True)
    if gaussian_fit:
        plot_gaussian(x.dropna(), ax=ax, legend=legend)


def QQ_plot(x, ax):
    """
    Plots a QQ plot of the data x.
    """

    normed_data = (x - x.mean()) / x.std()
    normed_data = normed_data.sort_values()
    normed_data = normed_data.reset_index(drop=True)

    n = len(normed_data)
    quantiles = np.arange(1, n + 1) / n
    quantiles_theoretical = norm.ppf(quantiles)

    ax.plot(normed_data, quantiles_theoretical)
    ax.plot(
        [normed_data.min(), normed_data.max()],
        [normed_data.min(), normed_data.max()],
        "k--",
    )
    ax.set_xlabel("Empirical quantiles")
    ax.set_ylabel("Theoretical quantiles")
    ax.set_title("QQ plot")

    return ax


def get_risk_allocations(weights, covariances):
    """
    Calculates the risk allocations of the portfolio.
    """

    risk_alloctions = pd.DataFrame(index=weights.index, columns=weights.columns)

    for t in weights.index:
        weights_t = weights.loc[t].values
        covariance_t = covariances[t].values

        risk_alloctions.loc[t] = (
            weights_t
            * (covariance_t @ weights_t)
            / np.dot(weights_t, covariance_t @ weights_t)
        )

    return risk_alloctions


def _from_cov_to_corr(cov):
    """
    Converts a covariance matrix to a correlation matrix.
    """

    volas = np.sqrt(np.diag(cov))

    return pd.DataFrame(
        cov / np.outer(volas, volas), index=cov.index, columns=cov.columns
    )


def from_cov_to_corr(covs):
    """
    Converts a dictionary of covariance matrices to a dictionary of correlation matrices.
    """

    return {t: _from_cov_to_corr(cov) for t, cov in covs.items()}


def get_volas(covs):
    """
    Calculates the volatilities of the assets.
    """

    asset_names = list(covs.values())[0].columns

    volas = pd.DataFrame({t: np.sqrt(np.diag(covs[t])) for t in covs.keys()}).T

    volas.index = list(covs.keys())
    volas.columns = asset_names

    return volas * np.sqrt(250)
