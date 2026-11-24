import numbers
from typing import NamedTuple

import numpy as np
import pmdarima as pm
import statsmodels.api as sm
from scipy.linalg import toeplitz
from statsmodels.tsa.stattools import acovf

from arimaclust.autoarima.estimation import to_arma_process


def check_tau(tau: object) -> None:
    """Raise a ``ValueError`` unless ``tau``, the number of AR(inf) coefficients compared, is an integer >= 1."""
    if not isinstance(tau, numbers.Integral) or isinstance(tau, bool) or tau < 1:
        raise ValueError(f'tau must be an integer >= 1, got {tau!r}')


def arma2ar(model: tuple | pm.arima.arima.ARIMA | sm.tsa.ArmaProcess, tau: int = 50) -> np.ndarray:
    """
    Truncated AR(inf) lag polynomial of an invertible ARIMA model.

    A differencing order is part of the AR polynomial, phi(B) (1 - B)^d: a ``pmdarima`` model keeps it, an
    ``ArmaProcess`` built by ``to_arma_process(..., d=d)`` already holds it, and a tuple has none.

    Parameters
    ----------
    model : tuple | pm.arima.arima.ARIMA | sm.tsa.ArmaProcess
        ``(ar_params, ma_params)`` tuple, fitted pmdarima ARIMA, or statsmodels ``ArmaProcess``
    tau : int
        number of lags returned, an integer >= 1

    Returns
    -------
    coefficients : np.ndarray
        lag polynomial coefficients at lags 0..tau-1, that is (1, -pi_1, ..., -pi_{tau-1})
    """
    check_tau(tau)
    if isinstance(model, tuple):
        if len(model) != 2:
            raise ValueError('Please provide tuple of AR and MA params')
        process = to_arma_process(*model)
    elif isinstance(model, pm.arima.arima.ARIMA):
        process = to_arma_process(model.arparams(), model.maparams(), d=model.order[1])
    elif isinstance(model, sm.tsa.ArmaProcess):
        process = model
    else:
        raise TypeError('Argument model must be tuple of parameters, pmdarima ARIMA, or sm.ArmaProcess')

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
        invertible ARIMA models
    tau : int
        number of AR(inf) coefficients pi_1..pi_tau compared, an integer >= 1

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
    check_tau(tau)  # before tau + 1, which would let tau=0 through
    return -arma2ar(model=process, tau=tau + 1)[1:]


def cepstral_coefficients(process: sm.tsa.ArmaProcess, tau: int = 50) -> np.ndarray:
    """
    LPC cepstral coefficients of an invertible ARIMA model, computed from its AR(inf) coefficients.

    Parameters
    ----------
    process : sm.tsa.ArmaProcess
        invertible ARIMA model
    tau : int
        number of AR(inf) coefficients pi_1..pi_tau used, and of cepstral coefficients returned, an integer >= 1

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
        invertible ARIMA models
    tau : int
        number of cepstral coefficients compared, computed from pi_1..pi_tau, an integer >= 1

    Returns
    -------
    distance : float
        cepstral distance
    """
    if not (isinstance(x_process, sm.tsa.ArmaProcess) and isinstance(y_process, sm.tsa.ArmaProcess)):
        raise TypeError('Both processes must be statsmodels ArmaProcess instances')

    return float(np.linalg.norm(cepstral_coefficients(x_process, tau=tau) - cepstral_coefficients(y_process, tau=tau)))


class SeriesModel(NamedTuple):
    """A series together with the ARMA model fitted to it, the input of ``mahalanobis``; no differencing order."""

    series: np.ndarray
    process: sm.tsa.ArmaProcess


def _ar_estimate(item: SeriesModel, tau: int) -> tuple[np.ndarray, np.ndarray]:
    """AR(inf) coefficients of ``item``'s model and their asymptotic covariance sigma^2 R^-1 / T."""
    pi = _ar_coefficients(item.process, tau=tau)
    if len(item.series) <= tau:
        raise ValueError(f'SeriesModel series has {len(item.series)} observations; '
                         f'mahalanobis needs more than tau={tau}')
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
        series with their invertible ARMA models; each series must be longer than ``tau``, and should be
        much longer
    tau : int
        number of AR(inf) coefficients pi_1..pi_tau compared, the order of the test, an integer >= 1

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
