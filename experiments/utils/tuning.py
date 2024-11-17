import numpy as np
from metrics import Metrics
from backtest import ProblemData, run_backtest


def tune_risk_allocations(returns, choleskies, risk_allocations, max_iter=100):
    _risk_allocations_curr = risk_allocations.copy()

    def _get_sharpe(allocations):
        problem_data = ProblemData(
                                    n_assets=returns.shape[1],
                                    n_crypto=2,
                                    gamma=None,
                                    w_min=None,
                                    w_max=0.1,
                                    risk_limit=0.1,
                                    leverage_limit=None,
                                    asset_names=returns.columns,
                                    method='risk_parity',
                                    risk_allocations=allocations,
                                    dilute=True,
                                )
        backtest = run_backtest(choleskies, problem_data)
        metrics = Metrics(backtest.weights, backtest.cash, returns.shift(-1))

        return metrics.sharpe

    n_allocations = len(risk_allocations)

    sharpe_max = _get_sharpe(_risk_allocations_curr)
    print("Current sharpe: ", sharpe_max)

    consecutive = 0
    iter_number = 0
    while True and iter_number < max_iter:
        i = np.random.randint(n_allocations)

        _risk_allocations_new = _risk_allocations_curr.copy()
        _risk_allocations_new[i] *= 1.25
        _risk_allocations_new /= _risk_allocations_new.sum()

        sharpe_new = _get_sharpe(_risk_allocations_new)
        if sharpe_new > sharpe_max:
            sharpe_max = sharpe_new
            _risk_allocations_curr = _risk_allocations_new.copy()
            consecutive = 0
            print("New sharpe: ", sharpe_new)
        
        else:
            _risk_allocations_new = _risk_allocations_curr.copy()
            _risk_allocations_new[i] *= 0.8
            _risk_allocations_new /= _risk_allocations_new.sum()

            sharpe_new = _get_sharpe(_risk_allocations_new)
            if sharpe_new > sharpe_max:
                sharpe_max = sharpe_new
                _risk_allocations_curr = _risk_allocations_new.copy()
                consecutive = 0
                print("New sharpe: ", sharpe_new)

        consecutive += 1
        print("\nConsecutive: ", consecutive)
        if consecutive == n_allocations:
            break

    return _risk_allocations_curr
        
         