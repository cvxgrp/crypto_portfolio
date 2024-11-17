from collections import namedtuple
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class ProblemData:
    n_assets: int
    n_crypto: int
    gamma: float
    w_min: float
    w_max: float
    risk_limit: float
    leverage_limit: float
    asset_names: list
    method: str
    risk_allocations: np.ndarray
    dilute: bool


Markowitz = namedtuple("Markowitz", ["weights", "cash", "risk", "chol", "problem"])
RiskParity = namedtuple("RiskParity", ["x", "risk", "chol", "problem"])


def _get_problem(problem_data):
    chol = cp.Parameter((problem_data.n_assets, problem_data.n_assets))

    if problem_data.method == "markowitz":
        weights = cp.Variable(problem_data.n_assets)
        cash = cp.Variable()

        objective = cp.Maximize(
            problem_data.gamma * weights[: problem_data.n_crypto].sum()
            + weights[problem_data.n_crypto :].sum()
        )

        risk = cp.norm(chol.T @ weights, 2) * np.sqrt(250)
        constraints = [
            cp.sum(weights) + cash == 1,
            weights >= problem_data.w_min,
            weights <= problem_data.w_max,
            risk <= problem_data.risk_limit,
            cp.norm(weights, 1) <= problem_data.leverage_limit,
        ]

        problem = cp.Problem(objective, constraints)

        return Markowitz(weights, cash, risk, chol, problem)

    elif problem_data.method == "risk_parity":
        x = cp.Variable(problem_data.n_assets)
        risk = cp.norm(chol.T @ x, 2) * np.sqrt(250)

        objective = 0.5 * cp.square(risk) - problem_data.risk_allocations @ cp.log(
            x
        )  # + problem_data.gamma * cash

        problem = cp.Problem(cp.Minimize(objective))

        return RiskParity(x, risk, chol, problem)


Solution = namedtuple("Solution", ["weights", "cash", "risk"])
Backtest = namedtuple("Backtest", ["weights", "cash", "risks"])


def estimate_risk(weights, returns, halflife=10):
    portfolio_returns = returns @ weights

    return portfolio_returns.ewm(halflife=halflife).std().iloc[-1]


def _dilute(weights, risk, problem_data):
    theta1 = problem_data.risk_limit / risk
    if problem_data.n_crypto > 0:
        theta2 = problem_data.w_max / weights[: problem_data.n_crypto].sum()
    else:
        theta2 = theta1
    theta3 = min(1, 1 / weights.sum())

    weights *= min(theta1, theta2, theta3)

    return weights


def _solve(chol, problem_data, problem, returns=None):
    """
    param chol: Cholesky decomposition of the covariance matrix
    param problem_data: dataclass containing the problem data
    """

    if problem_data.method == "markowitz":
        pass

    elif problem_data.method == "risk_parity":
        problem.chol.value = chol
        problem.problem.solve(solver="MOSEK")

        weights = problem.x.value  # / problem.x.value.sum()
        risk = np.linalg.norm(chol.T @ weights, 2) * np.sqrt(250)

        risk = estimate_risk(weights, returns, halflife=10) * np.sqrt(250)

        if problem_data.dilute and (
            risk > problem_data.risk_limit
            or weights[: problem_data.n_crypto].sum() > problem_data.w_max
        ):
            # theta1 = problem_data.risk_limit / risk
            # theta2 = problem_data.w_max / weights[:problem_data.n_crypto].sum()
            # weights *= min(theta1, theta2)

            weights = _dilute(weights, risk, problem_data)

            risk = np.linalg.norm(chol.T @ weights, 2) * np.sqrt(250)

        cash = 1 - weights.sum()

        return Solution(pd.Series(weights, index=problem_data.asset_names), cash, risk)


def run_backtest(choleskies, problem_data, returns=None):
    """
    param choleskies: dictionary of Cholesky decompositions of the covariance
    matrices
    param problem_data: dataclass containing the problem data
    """

    times = list(choleskies.keys())

    weights = pd.DataFrame(
        index=times, columns=choleskies[times[0]].columns, dtype=float
    )
    cash = pd.Series(index=times, dtype=float)
    risks = pd.Series(index=times, dtype=float)

    problem = _get_problem(problem_data)

    for t in times:
        solution_t = _solve(
            choleskies[t].values,
            problem_data,
            problem,
            returns.loc[:t] if returns is not None else None,
        )
        weights.loc[t] = solution_t.weights
        cash.loc[t] = solution_t.cash
        risks.loc[t] = solution_t.risk

    return Backtest(weights, cash, risks)


def run_backtest_eq_weight(choleskies, problem_data):
    """
    param choleskies: dictionary of Cholesky decompositions of the covariance
    matrices
    param problem_data: dataclass containing the problem data
    """

    times = list(choleskies.keys())

    weights = pd.DataFrame(
        index=times, columns=choleskies[times[0]].columns, dtype=float
    )
    cash = pd.Series(index=times, dtype=float)
    risks = pd.Series(index=times, dtype=float)

    eq_weight = [0.05, 0.05]
    if problem_data.n_assets > 2:
        eq_weight += [0.9 / (problem_data.n_assets - 2)] * (problem_data.n_assets - 2)
    eq_weight = np.array(eq_weight)

    for t in times:
        weights_t = eq_weight.copy()
        risk_t = np.linalg.norm(choleskies[t].values.T @ weights_t, 2) * np.sqrt(250)
        weights_t = _dilute(weights_t, risk_t, problem_data)
        cash_t = 1 - weights_t.sum()

        weights.loc[t] = weights_t
        cash.loc[t] = cash_t
        risks.loc[t] = np.linalg.norm(choleskies[t].values.T @ weights_t, 2) * np.sqrt(
            250
        )

    return Backtest(weights, cash, risks)
