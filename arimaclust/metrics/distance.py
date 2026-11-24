from collections.abc import Hashable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from arimaclust.metrics.autoregressive import METRICS
from arimaclust.metrics.matrix import dist_matrix_parallel
from arimaclust.modeling import ARIMA
from arimaclust.utils import check_tau, timeit

RESERVED_ARIMA_KWARGS = ('d', 'max_p', 'max_q')


def _split_series(X: ArrayLike | pd.DataFrame) -> tuple[list[np.ndarray], list]:
    """
    Split ``X`` into validated 1-D series.

    Parameters
    ----------
    X : ArrayLike | pd.DataFrame
        list or tuple of 1-D series, 2D array (one series per row), or DataFrame
        (one series per column, NaNs dropped)

    Returns
    -------
    series, names : tuple[list[np.ndarray], list]
        the series, and the name of each in error messages: its DataFrame column, or its position
    """
    if isinstance(X, pd.DataFrame):
        names = X.columns.tolist()
        series = [
            check_array(X[col].dropna().to_numpy(), ensure_2d=False, ensure_all_finite=False, ensure_min_samples=0)
            for col in names
        ]
    else:
        if isinstance(X, list | tuple):
            series = [check_array(x, ensure_2d=False, ensure_all_finite=False, ensure_min_samples=0) for x in X]
        else:
            series = list(check_array(X, ensure_all_finite=False))
        names = list(range(len(series)))

    for name, x in zip(names, series, strict=True):
        if x.ndim != 1:
            raise ValueError(f'Series {name!r} must be 1-D, got an array of shape {x.shape}')
        if x.size == 0:
            raise ValueError(f'Series {name!r} is empty; DataFrame columns are empty when all their values are NaN')
        if not np.isfinite(x).all():
            raise ValueError(
                f'Series {name!r} contains NaN or infinite values; only DataFrame columns have their NaNs dropped'
            )
    if len(series) < 2:
        raise ValueError(f'ARMADistance needs at least 2 series, got n_samples={len(series)}')
    return series, names


def _fit_named(x: np.ndarray, name: Hashable, **kwargs) -> ARIMA:
    """``ARIMA.fit``, with a failure re-raised under the series' name."""
    try:
        return ARIMA.fit(x, **kwargs)
    except Exception as error:
        raise ValueError(f'auto_arima could not fit series {name!r}: {error}') from error


@timeit
def _fit_all(series: list[np.ndarray], names: list[Hashable], n_jobs: int | None, **kwargs) -> list[ARIMA]:
    """Fit one model per series in parallel, in input order."""
    jobs = (delayed(_fit_named)(x, name, **kwargs) for x, name in zip(series, names, strict=True))
    return Parallel(n_jobs=n_jobs)(jobs)


class ARMADistance(TransformerMixin, BaseEstimator):
    """
    Pairwise distances between ARIMA models fitted to each series.

    Each series gets an ARIMA model selected with ``pmdarima.auto_arima`` (BIC by default),
    and the models are compared through their first ``tau`` AR(inf) coefficients, the expansion of
    phi(B) (1 - B)^d / theta(B). ``auto_arima`` searches the orders stepwise by default: about three times
    faster than the full search the consistency theorem assumes, and in our tests it rarely missed the BIC
    minimum. ``arima_kwargs={'stepwise': False, 'max_order': None}`` runs the full search.

    There is no ``transform``: comparing new series needs their models fitted together with the others,
    so ``fit`` and ``fit_transform`` work on the whole collection at once.

    Parameters
    ----------
    max_p : int
        maximum AR order
    max_q : int
        maximum MA order
    d : int | None
        differencing order, kept in each model's AR(inf) coefficients; ``None`` lets ``auto_arima``
        choose it per series, which ``'mahalanobis'`` does not allow
    metric : str
        ``'piccolo'`` (Piccolo distance), ``'cepstral'`` (LPC cepstral distance) or ``'mahalanobis'``
        (Maharaj's Mahalanobis distance, which also weights the coefficients by each series)
    tau : int
        number of AR(inf) coefficients pi_1..pi_tau compared, an integer >= 1; for ``'cepstral'`` the number
        of cepstral coefficients, for ``'mahalanobis'`` the AR order of the test, and series must be longer
        than it after differencing
    arima_kwargs : dict | None
        extra keywords for ``auto_arima`` (for example ``information_criterion``); they override
        the fit defaults but must not set ``max_p``, ``max_q`` or ``d``
    n_jobs : int | None
        number of parallel jobs for the model fits and the pairwise distances; ``None`` runs both serially

    Attributes
    ----------
    models_ : list[ARIMA] | dict[Hashable, ARIMA]
        fitted model per series, holding its series, as ``{column: model}`` for a DataFrame
    dist_matrix_ : np.ndarray | pd.DataFrame
        symmetric distance matrix, labelled by the columns for a DataFrame
    """

    def __init__(
        self,
        max_p: int = 3,
        max_q: int = 3,
        d: int | None = 0,
        metric: str = 'piccolo',
        tau: int = 50,
        arima_kwargs: dict | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.max_p = max_p
        self.max_q = max_q
        self.d = d
        self.metric = metric
        self.tau = tau
        self.arima_kwargs = arima_kwargs
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike | pd.DataFrame, y: None = None) -> 'ARMADistance':
        """
        Fit one ARMA model per series and compute their pairwise distances.

        Parameters
        ----------
        X : ArrayLike | pd.DataFrame
            list or tuple of 1-D series (lengths may differ), 2D array with one series per row,
            or DataFrame with one series per column (NaNs dropped)
        y : None
            ignored

        Returns
        -------
        self : ARMADistance
            fitted estimator
        """
        arima_kwargs = {} if self.arima_kwargs is None else self.arima_kwargs
        reserved = sorted(set(arima_kwargs) & set(RESERVED_ARIMA_KWARGS))
        if reserved:
            raise ValueError(
                f"arima_kwargs must not set {', '.join(map(repr, reserved))}; use ARMADistance's own parameter"
            )
        if self.metric not in METRICS:
            raise ValueError(f'metric must be one of {", ".join(map(repr, METRICS))}, got {self.metric!r}')
        if self.metric == 'mahalanobis' and self.d is None:
            raise ValueError(
                "metric='mahalanobis' needs an explicit differencing order d, got d=None; "
                'it compares the series differenced d times'
            )
        check_tau(self.tau)
        series, names = _split_series(X)
        if self.metric == 'mahalanobis':
            # checked before fitting, so a bad series fails fast and by name
            for name, x in zip(names, (np.diff(x, n=self.d) for x in series), strict=True):
                if len(x) <= self.tau:
                    raise ValueError(
                        f'Series {name!r} has {len(x)} observations after differencing; '
                        f"metric='mahalanobis' needs more than tau={self.tau}"
                    )
                if np.ptp(x) == 0:
                    raise ValueError(
                        f'Series {name!r} is constant after differencing; '
                        "metric='mahalanobis' needs a non-constant series"
                    )

        fit_kwargs = {'start_p': min(2, self.max_p), 'start_q': min(2, self.max_q), **arima_kwargs}
        models = _fit_all(series, names, n_jobs=self.n_jobs, max_p=self.max_p, max_q=self.max_q, d=self.d, **fit_kwargs)
        for name, model in zip(names, models, strict=True):
            if not model.isinvertible:
                raise ValueError(
                    f'Series {name!r} has a non-invertible fitted model {model!r}; '
                    'the AR(inf) representation every metric uses needs invertible models'
                )

        self.models_ = dict(zip(names, models, strict=True)) if isinstance(X, pd.DataFrame) else models
        self.dist_matrix_ = dist_matrix_parallel(
            self.models_, metric=METRICS[self.metric], n_jobs=self.n_jobs, tau=self.tau
        )
        return self

    def fit_transform(self, X: ArrayLike | pd.DataFrame, y: None = None) -> np.ndarray | pd.DataFrame:
        """
        Fit the models and return the distance matrix.

        Parameters
        ----------
        X : ArrayLike | pd.DataFrame
            series, as in ``fit``
        y : None
            ignored

        Returns
        -------
        dist_matrix : np.ndarray | pd.DataFrame
            symmetric distance matrix, labelled by the columns for a DataFrame
        """
        return self.fit(X).dist_matrix_
