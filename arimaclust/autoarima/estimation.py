from typing import List, Tuple
import pmdarima as pm
import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
from numpy.typing import ArrayLike


def to_arma_process(ar_params: ArrayLike, ma_params: ArrayLike) -> sm.tsa.ArmaProcess:
    """
    Get ARMA process from given parameters

    Parameters
    ----------
    ar_params : ArrayLike
        autoregressive parameters
    ma_params : ArrayLike
        moving average parameters

    Returns
    -------
    process : sm.tsa.ArmaProcess
        output process
    """
    if isinstance(ar_params, list):
        ar_params = np.array(ar_params)
    if isinstance(ma_params, list):
        ar_params = np.array(ma_params)

    ar = np.r_[1, -ar_params]  # add zero-lag and negate
    ma = np.r_[1, ma_params]  # add zero-lag
    # create model
    process = sm.tsa.ArmaProcess(ar, ma)
    return process


def fit_arma(x: np.ndarray, max_orders: Tuple[int, int], **kwargs):
    if 'information_criterion' not in kwargs:
        information_criterion = 'bic'
    else:
        information_criterion = kwargs['information_criterion']

    max_p, max_q = max_orders

    x_pm_model = pm.auto_arima(x, start_p=1, start_q=1, max_p=max_p, max_q=max_q, d_max=2,
                               error_action='ignore', stepwise=True,
                               suppress_warnings=True,
                               seasonal=False, information_criterion=information_criterion)
    x_process = to_arma_process(ar_params=x_pm_model.arparams(), ma_params=x_pm_model.maparams())
    return x_process


def estimate_processes(X: List[np.ndarray], max_orders: Tuple[int, int], underlying_model: str = 'ARMA', **kwargs):
    if 'n_workers' not in kwargs:
        n_workers = 8
    else:
        n_workers = kwargs['n_workers']

    if underlying_model == 'ARMA':
        fit_func = fit_arma
    else:
        raise NotImplementedError('Other underlying processes not implemented yet')

    results = Parallel(n_jobs=n_workers)(delayed(fit_func)(x, max_orders, **kwargs) for x in X)
    return results
