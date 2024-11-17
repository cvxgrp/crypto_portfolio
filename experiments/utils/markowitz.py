from collections import namedtuple

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

MarkowitzProblem = namedtuple(
    "MarkowitzProblem",
    [
        "problem",
        "weights",
        "cash",
        "sqrtgamma_chol",
        "gamma_kappa_trade",
        "gamma_kappa_wold",
        "ret",
    ],
)
MarkowitzProblemBasic = namedtuple(
    "MarkowitzProblemBasic", ["problem", "weights", "cash", "sqrtgamma_chol", "ret"]
)


def get_markowitz_problem(n_assets):
    weights = cp.Variable(n_assets)
    cash = cp.Variable()
    ret = cp.Parameter(n_assets, name="ret")
    sqrtgamma_chol = cp.Parameter((n_assets, n_assets), name="sqrtgamma_chol")
    gamma_kappa_trade = cp.Parameter(n_assets, nonneg=True, name="gamma_kappa_trade")
    gamma_kappa_wold = cp.Parameter(n_assets, name="gamma_kappa_wold")

    risk = cp.sum_squares(sqrtgamma_chol.T @ weights)
    turnover = cp.norm1(cp.multiply(gamma_kappa_trade, weights) - gamma_kappa_wold)

    objective = cp.Maximize(ret @ weights * 250 - risk * 250 - turnover * 250)
    constraints = [
        cp.sum(weights) + cash == 1,
        weights <= 0.1,
        weights >= -0.1,
        cp.norm1(weights) + cp.abs(cash) <= 1.6,
    ]
    # constraints = []

    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    return MarkowitzProblem(
        problem=problem,
        weights=weights,
        cash=cash,
        sqrtgamma_chol=sqrtgamma_chol,
        gamma_kappa_trade=gamma_kappa_trade,
        gamma_kappa_wold=gamma_kappa_wold,
        ret=ret,
    )


def get_markowitz_basic(n_assets):
    weights = cp.Variable(n_assets)
    cash = cp.Variable()
    ret = cp.Parameter(n_assets, name="ret")
    sqrtgamma_chol = cp.Parameter((n_assets, n_assets), name="sqrtgamma_chol")

    risk = cp.sum_squares(sqrtgamma_chol.T @ weights)

    objective = cp.Maximize(ret @ weights * 250 - risk * 250)
    constraints = [cp.sum(weights) + cash == 1]

    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    return MarkowitzProblemBasic(
        problem=problem,
        weights=weights,
        cash=cash,
        sqrtgamma_chol=sqrtgamma_chol,
        ret=ret,
    )


def run_backtest_backprop(
    means, spreads, choleskies, omega_risk, omega_trade, requires_grad=False
):
    """
    Parameters
    ----------
    means : np.ndarray
        (n_samples, n_assets)
    choleskies : dict
        keys are the sample indices, values are the Cholesky factors of the
        covariance matrix
    omega : torch.Tensor of shape ()
        risk aversion parameter gamma_risk=exp(omega)

    Returns
    -------
    np.ndarray
        (n_samples, n_assets)
    """

    gamma_risk = 10**omega_risk
    gamma_trade = 10**omega_trade
    weights_t = (
        torch.ones(means.shape[1], requires_grad=True, dtype=torch.float64)
        / means.shape[1]
    )
    cash_t = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)

    _, n_assets = means.shape
    markowitz_problem = get_markowitz_problem(n_assets)
    layer = CvxpyLayer(
        markowitz_problem.problem,
        parameters=[
            markowitz_problem.ret,
            markowitz_problem.sqrtgamma_chol,
            markowitz_problem.gamma_kappa_trade,
            markowitz_problem.gamma_kappa_wold,
        ],
        variables=[markowitz_problem.weights, markowitz_problem.cash],
    )

    weights = []
    cash = []

    # for t in tqdm(means.index):
    for t in means.index:
        mean_t = torch.tensor(means.loc[t].values, requires_grad=True)
        kappa_t = torch.tensor(spreads.loc[t].values, requires_grad=True)
        chol_t = torch.tensor(choleskies[t].values, requires_grad=True)
        gamma_kappa_trade = gamma_trade * kappa_t
        gamma_kappa_wold = gamma_trade * kappa_t * weights_t
        sqrtgamma_chol_t = gamma_risk.sqrt() * chol_t

        (
            weights_t,
            cash_t,
        ) = layer(
            mean_t,
            sqrtgamma_chol_t,
            gamma_kappa_trade,
            gamma_kappa_wold,
            solver_args={"solve_method": "SCS", "eps": 1e-12},
        )
        weights.append(weights_t.reshape(1, -1))
        cash.append(cash_t)

    if requires_grad:
        return Backtest(
            weights=torch.vstack(weights),
            cash=torch.tensor(cash, dtype=torch.float64),
            omega_risk=omega_risk,
            omega_trade=omega_trade,
        )
    else:
        return Backtest(
            weights=torch.vstack(weights).detach(),
            cash=torch.tensor(cash, dtype=torch.float64).detach(),
            omega_risk=omega_risk,
            omega_trade=omega_trade,
        )


def run_backtest_basic_backprop(means, choleskies, omega, requires_grad=False):
    """
    Parameters
    ----------
    means : np.ndarray
        (n_samples, n_assets)
    choleskies : dict
        keys are the sample indices, values are the Cholesky factors of the
        covariance matrix
    omega : torch.Tensor of shape ()
        risk aversion parameter gamma_risk=exp(omega)

    Returns
    -------
    np.ndarray
        (n_samples, n_assets)
    """

    gamma = 10**omega
    weights_t = (
        torch.ones(means.shape[1], requires_grad=True, dtype=torch.float64)
        / means.shape[1]
    )
    cash_t = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)

    _, n_assets = means.shape
    markowitz_problem = get_markowitz_basic(n_assets)
    layer = CvxpyLayer(
        markowitz_problem.problem,
        parameters=[markowitz_problem.ret, markowitz_problem.sqrtgamma_chol],
        variables=[markowitz_problem.weights, markowitz_problem.cash],
    )

    weights = []
    cash = []

    # for t in tqdm(means.index):
    for t in means.index:
        mean_t = torch.tensor(means.loc[t].values, requires_grad=True)
        chol_t = torch.tensor(choleskies[t].values, requires_grad=True)
        sqrtgamma_chol_t = gamma.sqrt() * chol_t

        (
            weights_t,
            cash_t,
        ) = layer(mean_t, sqrtgamma_chol_t)
        weights.append(weights_t.reshape(1, -1))
        cash.append(cash_t)

    if requires_grad:
        return BacktestBasic(
            weights=torch.vstack(weights),
            cash=torch.tensor(cash, dtype=torch.float64),
            omega=omega,
        )
    else:
        return BacktestBasic(
            weights=torch.vstack(weights).detach(),
            cash=torch.tensor(cash, dtype=torch.float64).detach(),
            omega=omega,
        )


def run_backtest_wrapper(params):
    return run_backtest(*params)


BacktestBasic = namedtuple("BacktestBasic", ["weights", "cash", "gamma"])


def run_backtest_basic(means, choleskies, gamma):
    """
    Parameters
    ----------
    means : np.ndarray
        (n_samples, n_assets)
    choleskies : dict
        keys are the sample indices, values are the Cholesky factors of the
        covariance matrix
    omega : torch.Tensor of shape ()
        risk aversion parameter gamma_risk=exp(omega)

    Returns
    -------
    np.ndarray
        (n_samples, n_assets)
    """

    weights_t = np.ones(means.shape[1]) / means.shape[1]
    cash_t = np.array(0.0)

    _, n_assets = means.shape
    markowitz_problem = get_markowitz_basic(n_assets)

    weights = []
    cash = []

    from tqdm import tqdm

    for t in tqdm(means.index):
        # for t in means.index:
        mean_t = means.loc[t].values
        chol_t = choleskies[t].values
        sqrtgamma_chol_t = gamma**0.5 * chol_t

        markowitz_problem.ret.value = mean_t
        markowitz_problem.sqrtgamma_chol.value = sqrtgamma_chol_t

        markowitz_problem.problem.solve(solver="CLARABEL")

        (
            weights_t,
            cash_t,
        ) = (
            markowitz_problem.weights.value,
            markowitz_problem.cash.value,
        )
        weights.append(weights_t.reshape(1, -1))
        cash.append(cash_t)

    return BacktestBasic(weights=np.vstack(weights), cash=np.array(cash), gamma=gamma)


Backtest = namedtuple("Backtest", ["weights", "cash", "gamma_risk", "gamma_trade"])


def run_backtest(means, spreads, choleskies, gamma_risk, gamma_trade):
    """
    Parameters
    ----------
    means : np.ndarray
        (n_samples, n_assets)
    choleskies : dict
        keys are the sample indices, values are the Cholesky factors of the
        covariance matrix
    omega : torch.Tensor of shape ()
        risk aversion parameter gamma_risk=exp(omega)

    Returns
    -------
    np.ndarray
        (n_samples, n_assets)
    """

    weights_t = np.ones(means.shape[1]) / means.shape[1]
    cash_t = np.array(0.0)

    _, n_assets = means.shape
    markowitz_problem = get_markowitz_problem(n_assets)

    weights = []
    cash = []
    from tqdm import tqdm

    for t in tqdm(means.index):
        # for t in means.index:
        mean_t = means.loc[t].values
        kappa_t = spreads.loc[t].values
        chol_t = choleskies[t].values
        gamma_kappa_trade = gamma_trade * kappa_t
        gamma_kappa_wold = gamma_trade * kappa_t * weights_t
        sqrtgamma_chol_t = gamma_risk**0.5 * chol_t

        markowitz_problem.ret.value = mean_t
        markowitz_problem.sqrtgamma_chol.value = sqrtgamma_chol_t
        markowitz_problem.gamma_kappa_trade.value = gamma_kappa_trade
        markowitz_problem.gamma_kappa_wold.value = gamma_kappa_wold

        markowitz_problem.problem.solve(solver="CLARABEL")

        weights_t, cash_t = (
            markowitz_problem.weights.value,
            markowitz_problem.cash.value,
        )

        weights.append(weights_t.reshape(1, -1))
        cash.append(cash_t)

    return Backtest(
        weights=np.vstack(weights),
        cash=np.array(cash),
        gamma_risk=gamma_risk,
        gamma_trade=gamma_trade,
    )


def run_backtest_wrapper_clarabel(params):
    return run_backtest(*params)
