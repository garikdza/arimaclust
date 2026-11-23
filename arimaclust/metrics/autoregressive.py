from typing import NamedTuple

import numpy as np
import pmdarima as pm
import statsmodels.api as sm
from scipy.linalg import toeplitz
from statsmodels.tsa.stattools import acovf


def arma2ar(model: tuple | pm.arima.arima.ARIMA | sm.tsa.ArmaProcess, tau: int = 50) -> np.ndarray:
    """
    Truncated AR(inf) coefficients of an invertible ARMA model.

    ``model`` can be an ``(ar_params, ma_params)`` tuple, a fitted pmdarima ARIMA,
    or a statsmodels ``ArmaProcess``.
    """
    if isinstance(model, tuple):
        if len(model) != 2:
            raise ValueError('Please provide tuple of AR and MA params')
        ar_params, ma_params = model
    elif isinstance(model, pm.arima.arima.ARIMA):
        ar_params, ma_params = model.arparams(), model.maparams()
    elif isinstance(model, sm.tsa.ArmaProcess):
        ar_params = ma_params = None
    else:
        raise TypeError('Argument model must be tuple of parameters, pmdarima ARIMA, or sm.ArmaProcess')

    if ar_params is None:
        process = model
    else:
        ar = np.r_[1, -np.asarray(ar_params, dtype=float)]  # add zero-lag and negate
        ma = np.r_[1, np.asarray(ma_params, dtype=float)]  # add zero-lag
        process = sm.tsa.ArmaProcess(ar, ma)

    if not process.isinvertible:
        raise ValueError('Provided model is not invertible')
    return process.arma2ar(lags=tau)


def piccolo(x_process: sm.tsa.ArmaProcess, y_process: sm.tsa.ArmaProcess, tau: int = 50) -> float:
    """
    Piccolo (1990) AR distance: Euclidean distance between the first ``tau`` AR(inf) coefficients
    of the two models.

    Parameters
    ----------
    x_process, y_process : sm.tsa.ArmaProcess
        invertible ARMA models
    tau : int
        number of AR(inf) coefficients pi_1..pi_tau compared

    Returns
    -------
    distance : float
        Piccolo distance
    """
    if not (isinstance(x_process, sm.tsa.ArmaProcess) and isinstance(y_process, sm.tsa.ArmaProcess)):
        raise TypeError('Both processes must be statsmodels ArmaProcess instances')

    return float(np.linalg.norm(_ar_coefficients(x_process, tau=tau) - _ar_coefficients(y_process, tau=tau)))


def _ar_coefficients(process: sm.tsa.ArmaProcess, tau: int) -> np.ndarray:
    """First ``tau`` AR(inf) coefficients pi_1..pi_tau, in the sign of x_t = sum pi_k x_{t-k} + e_t."""
    return -arma2ar(model=process, tau=tau + 1)[1:]


def cepstral_coefficients(process: sm.tsa.ArmaProcess, tau: int = 50) -> np.ndarray:
    """
    LPC cepstral coefficients of an invertible ARMA model, computed from its AR(inf) coefficients.

    Parameters
    ----------
    process : sm.tsa.ArmaProcess
        invertible ARMA model
    tau : int
        number of AR(inf) coefficients used, and of cepstral coefficients returned

    Returns
    -------
    coefficients : np.ndarray
        cepstral coefficients c_1..c_tau, with c_n = pi_n + sum_{m=1}^{n-1} (m / n) c_m pi_{n-m}
    """
    pi = _ar_coefficients(process, tau=tau)
    coefficients = np.zeros(tau)
    for n in range(1, tau + 1):
        m = np.arange(1, n)
        coefficients[n - 1] = pi[n - 1] + (m / n * coefficients[:n - 1]) @ pi[:n - 1][::-1]
    return coefficients


def cepstral(x_process: sm.tsa.ArmaProcess, y_process: sm.tsa.ArmaProcess, tau: int = 50) -> float:
    """
    LPC cepstral distance (Kalpakis et al., 2001; TSclust's ``diss.AR.LPC.CEPS``): Euclidean distance
    between the first ``tau`` cepstral coefficients of the two models.

    Parameters
    ----------
    x_process, y_process : sm.tsa.ArmaProcess
        invertible ARMA models
    tau : int
        number of cepstral coefficients compared

    Returns
    -------
    distance : float
        cepstral distance
    """
    if not (isinstance(x_process, sm.tsa.ArmaProcess) and isinstance(y_process, sm.tsa.ArmaProcess)):
        raise TypeError('Both processes must be statsmodels ArmaProcess instances')

    return float(np.linalg.norm(cepstral_coefficients(x_process, tau=tau) - cepstral_coefficients(y_process, tau=tau)))


class SeriesModel(NamedTuple):
    """A series together with the ARMA model fitted to it, the input of ``mahalanobis``."""

    series: np.ndarray
    process: sm.tsa.ArmaProcess


def _ar_estimate(item: SeriesModel, tau: int) -> tuple[np.ndarray, np.ndarray]:
    """AR(inf) coefficients of ``item``'s model and their asymptotic covariance sigma^2 R^-1 / T."""
    pi = _ar_coefficients(item.process, tau=tau)
    gamma = acovf(item.series, adjusted=False, demean=True, fft=True, nlag=tau)
    a = np.r_[1, -pi]
    autocovariances = toeplitz(gamma)
    sigma2 = a @ autocovariances @ a
    return pi, sigma2 * np.linalg.inv(autocovariances[:tau, :tau]) / len(item.series)


def mahalanobis(x: SeriesModel, y: SeriesModel, tau: int = 50) -> float:
    """
    Mahalanobis AR distance (Maharaj, 1996; TSclust's ``diss.AR.MAH`` for independent series): the Wald
    statistic D = d' (sigma_x^2 R_x^-1 / T_x + sigma_y^2 R_y^-1 / T_y)^-1 d of the difference d between
    the first ``tau`` AR(inf) coefficients of the two models.

    R is the Toeplitz matrix of the series' biased, demeaned sample autocovariances at lags 0..tau-1,
    T its length, and sigma^2 = a' Gamma a, with a = (1, -pi_1, ..., -pi_tau) and Gamma the same Toeplitz
    matrix at lags 0..tau, the sample one-step prediction-error variance of the truncated AR filter.
    The coefficients come from the fitted ARMA model rather than from an AR fit of order ``tau``. Each
    series is weighted by its own length, so D is symmetric; TSclust instead scales by sqrt(T_x), so with
    equal lengths and the same coefficients and variances D is TSclust's value times sqrt(T).

    Parameters
    ----------
    x, y : SeriesModel
        series with their invertible ARMA models; each series should be much longer than ``tau``
    tau : int
        number of AR(inf) coefficients compared, the order of the test

    Returns
    -------
    distance : float
        Mahalanobis distance
    """
    if not (isinstance(x, SeriesModel) and isinstance(y, SeriesModel)):
        raise TypeError('Both arguments must be SeriesModel instances')

    x_pi, x_cov = _ar_estimate(x, tau=tau)
    y_pi, y_cov = _ar_estimate(y, tau=tau)
    diff = x_pi - y_pi
    return float(diff @ np.linalg.solve(x_cov + y_cov, diff))
