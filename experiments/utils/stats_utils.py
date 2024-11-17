import numpy as np


def var(x, weights):
    """
    Computes the (weighted) variance of an array.
    Assumes zero mean.
    """

    return (x**2 * weights).sum() / weights.sum()


def cov(x, y, weights):
    """
    Computes the (weighted) covariance of two arrays.
    Assumes zero mean.
    """

    return (x * y * weights).sum() / weights.sum()


def std(x, weights):
    """
    Computes the (weighted) standard deviation of an array.
    Assumes zero mean.
    """

    return np.sqrt(var(x, weights))


def corr(x, y, weights):
    """
    Computes the (weighted) correlation of two arrays.
    Assumes zero mean.
    """

    return cov(x, y, weights) / (std(x, weights) * std(y, weights))


def beta(x, y, weights):
    """
    Computes the (weighted) beta of x with respect to y.
    Assumes zero mean.
    """

    return cov(x, y, weights) / var(y, weights)


def alpha(x, y, weights):
    """
    Computes the (weighted) alpha of x with respect to y.
    Assumes zero mean.
    """

    return x.mean() - beta(x, y, weights) * y.mean()
