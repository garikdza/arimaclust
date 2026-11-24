"""Distances between two fitted ARIMA models, all built on their AR(inf) coefficients."""
import numpy as np
from scipy.linalg import toeplitz
from statsmodels.tsa.stattools import acovf

from arimaclust.modeling import ARIMA


def _check_models(x: object, y: object) -> None:
    if not (isinstance(x, ARIMA) and isinstance(y, ARIMA)):
        raise TypeError(f'Both models must be ARIMA instances, got {type(x).__name__} and {type(y).__name__}')


def piccolo(x: ARIMA, y: ARIMA, tau: int = 50) -> float:
    """
    Piccolo (1990) AR distance: Euclidean distance between the first ``tau`` AR(inf) coefficients
    of the two models.

    Parameters
    ----------
    x, y : ARIMA
        invertible ARIMA models
    tau : int
        number of AR(inf) coefficients pi_1..pi_tau compared, an integer >= 1

    Returns
    -------
    distance : float
        Piccolo distance
    """
    _check_models(x, y)
    return float(np.linalg.norm(x.pi(tau) - y.pi(tau)))


def cepstral_coefficients(model: ARIMA, tau: int = 50) -> np.ndarray:
    """
    LPC cepstral coefficients of an invertible ARIMA model, computed from its AR(inf) coefficients.

    Parameters
    ----------
    model : ARIMA
        invertible ARIMA model
    tau : int
        number of AR(inf) coefficients pi_1..pi_tau used, and of cepstral coefficients returned, an integer >= 1

    Returns
    -------
    coefficients : np.ndarray
        cepstral coefficients c_1..c_tau, with c_n = pi_n + sum_{m=1}^{n-1} (m / n) c_m pi_{n-m}
    """
    pi = model.pi(tau)
    coefficients = np.zeros(tau)
    for n in range(1, tau + 1):
        m = np.arange(1, n)
        coefficients[n - 1] = pi[n - 1] + (m / n * coefficients[:n - 1]) @ pi[:n - 1][::-1]
    return coefficients


def cepstral(x: ARIMA, y: ARIMA, tau: int = 50) -> float:
    """
    LPC cepstral distance (Kalpakis et al., 2001; TSclust's ``diss.AR.LPC.CEPS``): Euclidean distance
    between the first ``tau`` cepstral coefficients of the two models.

    Parameters
    ----------
    x, y : ARIMA
        invertible ARIMA models
    tau : int
        number of cepstral coefficients compared, computed from pi_1..pi_tau, an integer >= 1

    Returns
    -------
    distance : float
        cepstral distance
    """
    _check_models(x, y)
    return float(np.linalg.norm(cepstral_coefficients(x, tau=tau) - cepstral_coefficients(y, tau=tau)))


def _ar_estimate(model: ARIMA, tau: int) -> tuple[np.ndarray, np.ndarray]:
    """AR(inf) coefficients of the differenced model and their asymptotic covariance sigma^2 R^-1 / T."""
    if model.series is None:
        raise ValueError('mahalanobis needs the series each model was fitted to; this ARIMA has none')
    arma = model.differenced()
    pi = arma.pi(tau)
    if len(arma.series) <= tau:
        raise ValueError(f'The series has {len(arma.series)} observations after differencing; '
                         f'mahalanobis needs more than tau={tau}')
    gamma = acovf(arma.series, adjusted=False, demean=True, fft=True, nlag=tau)
    a = np.r_[1, -pi]
    autocovariances = toeplitz(gamma)
    sigma2 = a @ autocovariances @ a
    return pi, sigma2 * np.linalg.inv(autocovariances[:tau, :tau]) / len(arma.series)


def mahalanobis(x: ARIMA, y: ARIMA, tau: int = 50) -> float:
    """
    Mahalanobis AR distance (Maharaj, 1996; TSclust's ``diss.AR.MAH`` for independent series): the Wald
    statistic D = d' (sigma_x^2 R_x^-1 / T_x + sigma_y^2 R_y^-1 / T_y)^-1 d of the difference d between
    the first ``tau`` AR(inf) coefficients of the two models' ARMA parts.

    Each model is taken with its series differenced ``d`` times. R is the Toeplitz matrix of that series'
    biased, demeaned sample autocovariances at lags 0..tau-1, T its length, and sigma^2 = a' Gamma a, with
    a = (1, -pi_1, ..., -pi_tau) and Gamma the same Toeplitz matrix at lags 0..tau, the sample one-step
    prediction-error variance of the truncated AR filter. The coefficients come from the fitted ARMA model
    rather than from an AR fit of order ``tau``. Each series is weighted by its own length, so D is symmetric;
    TSclust instead scales by sqrt(T_x), so with equal lengths and the same coefficients and variances D is
    TSclust's value times sqrt(T).

    Parameters
    ----------
    x, y : ARIMA
        invertible ARIMA models with their series; each differenced series must be longer than ``tau``, and
        should be much longer
    tau : int
        number of AR(inf) coefficients pi_1..pi_tau compared, the order of the test, an integer >= 1

    Returns
    -------
    distance : float
        Mahalanobis distance
    """
    _check_models(x, y)
    x_pi, x_cov = _ar_estimate(x, tau=tau)
    y_pi, y_cov = _ar_estimate(y, tau=tau)
    diff = x_pi - y_pi
    return float(diff @ np.linalg.solve(x_cov + y_cov, diff))


# the metrics ARMADistance accepts, by name
METRICS = {'piccolo': piccolo, 'cepstral': cepstral, 'mahalanobis': mahalanobis}
