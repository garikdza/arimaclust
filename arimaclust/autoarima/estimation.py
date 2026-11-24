from collections.abc import Hashable, Sequence

import numpy as np
import pmdarima as pm
import statsmodels.api as sm
from joblib import Parallel, delayed
from numpy.typing import ArrayLike

from arimaclust.utils import timeit


def to_arma_process(ar_params: ArrayLike, ma_params: ArrayLike, d: int = 0) -> sm.tsa.ArmaProcess:
    """
    Build the lag polynomials of an ARIMA(p, d, q) model from its parameters.

    Parameters
    ----------
    ar_params : ArrayLike
        autoregressive parameters phi_1..phi_p, in the sign of x_t = sum phi_k x_{t-k} + ...
    ma_params : ArrayLike
        moving average parameters theta_1..theta_q, in the sign of ... + e_t + sum theta_k e_{t-k}
    d : int
        differencing order, kept in the AR polynomial as phi(B) (1 - B)^d

    Returns
    -------
    process : sm.tsa.ArmaProcess
        process with AR polynomial phi(B) (1 - B)^d and MA polynomial theta(B); it is not stationary when d >= 1
    """
    ar = np.r_[1, -np.asarray(ar_params, dtype=float)]  # add zero-lag and negate
    for _ in range(d):
        ar = np.convolve(ar, [1, -1])
    ma = np.r_[1, np.asarray(ma_params, dtype=float)]  # add zero-lag
    return sm.tsa.ArmaProcess(ar, ma)


def fit_arima(x: np.ndarray, information_criterion: str = 'bic', **kwargs) -> tuple[sm.tsa.ArmaProcess, int]:
    """
    Select and fit an ARIMA model with ``pmdarima.auto_arima``.

    BIC is the default because consistent order selection (and hence the consistency
    guarantees of the clustering algorithm) requires it; AIC is not consistent.
    Remaining ``kwargs`` (``max_p``, ``max_q``, ``d``, ...) go to ``auto_arima`` and
    override its fixed options (``error_action``, ``stepwise``, ``suppress_warnings``).

    Returns
    -------
    arma, d : tuple[sm.tsa.ArmaProcess, int]
        the fitted ARMA part, and the differencing order used
    """
    options = {'error_action': 'ignore', 'stepwise': True, 'suppress_warnings': True, **kwargs}
    model = pm.auto_arima(x, information_criterion=information_criterion, **options)
    return to_arma_process(ar_params=model.arparams(), ma_params=model.maparams()), int(model.order[1])


def _fit_named(x: np.ndarray, name: Hashable, **kwargs) -> tuple[sm.tsa.ArmaProcess, int]:
    """``fit_arima``, with a failure re-raised under the series' name."""
    try:
        return fit_arima(x, **kwargs)
    except Exception as error:
        raise ValueError(f'auto_arima could not fit series {name!r}: {error}') from error


@timeit
def estimate_processes(X: list[np.ndarray], n_jobs: int | None = None, names: Sequence[Hashable] | None = None,
                       **kwargs) -> list[tuple[sm.tsa.ArmaProcess, int]]:
    """
    Fit one ARIMA model per series in parallel.

    Parameters
    ----------
    X : list[np.ndarray]
        1-D series, possibly of different lengths
    n_jobs : int | None
        number of joblib workers; ``None`` runs serially
    names : Sequence[Hashable] | None
        name of each series in error messages; positions by default
    **kwargs
        passed to ``fit_arima`` and on to ``auto_arima``

    Returns
    -------
    fits : list[tuple[sm.tsa.ArmaProcess, int]]
        the fitted ARMA part and differencing order of each series, in input order
    """
    names = range(len(X)) if names is None else names
    return Parallel(n_jobs=n_jobs)(delayed(_fit_named)(x, name, **kwargs) for x, name in zip(X, names, strict=True))
