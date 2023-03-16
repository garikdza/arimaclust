from typing import List, Tuple
import pmdarima as pm
import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from arimaclust.utils import timeit


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




def fit_arma(x: np.ndarray, **kwargs):
    x_pm_model = pm.auto_arima(x, error_action='ignore',
                               stepwise=True,
                               suppress_warnings=True,
                               **kwargs
                               )
    x_process = to_arma_process(ar_params=x_pm_model.arparams(), ma_params=x_pm_model.maparams())
    return x_process

@timeit
def estimate_processes(X: List[np.ndarray], underlying_process: str = 'ARMA', **kwargs):
    if 'n_workers' not in kwargs:
        n_workers = 8
    else:
        n_workers = kwargs['n_workers']

    if underlying_process == 'ARMA':
        fit_func = fit_arma
    else:
        raise NotImplementedError('Other underlying processes not implemented yet')

    results = Parallel(n_jobs=n_workers)(delayed(fit_func)(x, **kwargs) for x in X)
    return results
