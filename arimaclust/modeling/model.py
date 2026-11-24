from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm

from arimaclust.utils import check_tau


@dataclass(frozen=True, eq=False, repr=False)
class FittedARIMA:
    """
    An ARIMA(p, d, q) model, phi(B) (1 - B)^d x_t = theta(B) e_t, optionally with the series it was fitted to.

    Parameters
    ----------
    ar : ArrayLike
        autoregressive parameters phi_1..phi_p, in the sign of x_t = sum phi_k x_{t-k} + ...
    ma : ArrayLike
        moving average parameters theta_1..theta_q, in the sign of ... + e_t + sum theta_k e_{t-k}
    d : int
        differencing order
    series : ArrayLike | None
        the series the model was fitted to, undifferenced; ``mahalanobis`` needs it
    """

    ar: np.ndarray
    ma: np.ndarray
    d: int = 0
    series: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'ar', np.asarray(self.ar, dtype=float).reshape(-1))
        object.__setattr__(self, 'ma', np.asarray(self.ma, dtype=float).reshape(-1))
        if not isinstance(self.d, int | np.integer) or isinstance(self.d, bool) or self.d < 0:
            raise ValueError(f'd must be an integer >= 0, got {self.d!r}')
        object.__setattr__(self, 'd', int(self.d))
        if self.series is not None:
            object.__setattr__(self, 'series', np.asarray(self.series, dtype=float))

    def __repr__(self) -> str:
        return f'FittedARIMA(ar={self.ar.tolist()}, ma={self.ma.tolist()}, d={self.d})'

    @property
    def order(self) -> tuple[int, int, int]:
        """``(p, d, q)``."""
        return len(self.ar), self.d, len(self.ma)

    @property
    def arma_process(self) -> sm.tsa.ArmaProcess:
        """The ARMA part, without the differencing, as a statsmodels ``ArmaProcess``."""
        return sm.tsa.ArmaProcess(np.r_[1, -self.ar], np.r_[1, self.ma])

    @property
    def ar_polynomial(self) -> np.ndarray:
        """Coefficients of phi(B) (1 - B)^d at lags 0, 1, ..., p + d."""
        polynomial = np.r_[1, -self.ar]
        for _ in range(self.d):
            polynomial = np.convolve(polynomial, [1, -1])
        return polynomial

    @property
    def isinvertible(self) -> bool:
        """Whether every root of theta(B) lies outside the unit circle, which the AR(inf) representation needs."""
        return bool(self.arma_process.isinvertible)

    def pi(self, tau: int = 50) -> np.ndarray:
        """
        The first ``tau`` AR(inf) coefficients pi_1..pi_tau of phi(B) (1 - B)^d / theta(B), in the sign of
        x_t = sum pi_k x_{t-k} + e_t.

        Parameters
        ----------
        tau : int
            number of coefficients, an integer >= 1

        Returns
        -------
        pi : np.ndarray
            the coefficients, of shape ``(tau,)``
        """
        check_tau(tau)
        if not self.isinvertible:
            raise ValueError(f'{self!r} is not invertible, so it has no AR(inf) representation')
        lag_polynomial = sm.tsa.ArmaProcess(self.ar_polynomial, np.r_[1, self.ma]).arma2ar(lags=tau + 1)
        return -lag_polynomial[1:]

    def differenced(self) -> 'FittedARIMA':
        """The ARMA(p, q) model of the series differenced ``d`` times, with that series."""
        series = None if self.series is None else np.diff(self.series, n=self.d)
        return FittedARIMA(self.ar, self.ma, 0, series)
