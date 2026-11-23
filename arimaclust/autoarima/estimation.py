import numpy as np
import pmdarima as pm
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
    ar_params = np.asarray(ar_params, dtype=float)
    ma_params = np.asarray(ma_params, dtype=float)

    ar = np.r_[1, -ar_params]  # add zero-lag and negate
    ma = np.r_[1, ma_params]  # add zero-lag
    return sm.tsa.ArmaProcess(ar, ma)


def fit_arma(x: np.ndarray, information_criterion: str = 'bic', **kwargs) -> sm.tsa.ArmaProcess:
    """
    Select and fit an ARIMA model with ``pmdarima.auto_arima`` and return its ARMA part.

    BIC is the default because consistent order selection (and hence the consistency
    guarantees of the clustering algorithm) requires it; AIC is not consistent.
    Remaining ``kwargs`` (``max_p``, ``max_q``, ``d``, ...) go to ``auto_arima`` and
    override its fixed options (``error_action``, ``stepwise``, ``suppress_warnings``).
    """
    options = {'error_action': 'ignore', 'stepwise': True, 'suppress_warnings': True, **kwargs}
    model = pm.auto_arima(x, information_criterion=information_criterion, **options)
    return to_arma_process(ar_params=model.arparams(), ma_params=model.maparams())


@timeit
def estimate_processes(X: list[np.ndarray],
                       underlying_process: str = 'ARMA',
                       n_workers: int | None = 8,
                       **kwargs) -> list[sm.tsa.ArmaProcess]:
    """
    Fit one process per series in parallel.

    Parameters
    ----------
    X : list[np.ndarray]
        1-D series, possibly of different lengths
    underlying_process : str
        model family to fit; only ``'ARMA'`` is implemented
    n_workers : int
        number of joblib workers (``None`` runs serially)

    Returns
    -------
    processes : list[sm.tsa.ArmaProcess]
        fitted processes, in input order
    """
    if underlying_process == 'ARMA':
        fit_func = fit_arma
    else:
        raise NotImplementedError('Other underlying processes not implemented yet')

    return Parallel(n_jobs=n_workers)(delayed(fit_func)(x, **kwargs) for x in X)
