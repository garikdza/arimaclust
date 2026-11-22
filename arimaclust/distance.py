import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from arimaclust.autoarima.estimation import estimate_processes
from arimaclust.metrics.matrix import dist_matrix_parallel

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
        series = [check_array(X[col].dropna().to_numpy(), ensure_2d=False, ensure_all_finite=False,
                              ensure_min_samples=0) for col in names]
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
            raise ValueError(f'Series {name!r} contains NaN or infinite values; '
                             'only DataFrame columns have their NaNs dropped')
    if len(series) < 2:
        raise ValueError(f'ARMADistance needs at least 2 series, got n_samples={len(series)}')
    return series, names


class ARMADistance(TransformerMixin, BaseEstimator):
    """
    Pairwise Piccolo distances between ARMA models fitted to each series.

    Each series gets an ARMA model selected with ``pmdarima.auto_arima`` (BIC by default),
    and the models are compared through their AR(inf) coefficients truncated at ``tau`` lags.

    Parameters
    ----------
    max_p : int
        maximum AR order
    max_q : int
        maximum MA order
    d : int
        differencing order; it only affects fitting, the models keep their AR and MA parts
    tau : int
        number of AR(inf) lags compared by the Piccolo distance
    arima_kwargs : dict | None
        extra keywords for ``auto_arima`` (for example ``information_criterion``); they override
        the fit defaults but must not set ``max_p``, ``max_q`` or ``d``
    n_jobs : int | None
        number of series fitted in parallel; ``None`` fits them serially

    Attributes
    ----------
    processes_ : list[sm.tsa.ArmaProcess] | dict[Hashable, sm.tsa.ArmaProcess]
        fitted model per series, as ``{column: process}`` for a DataFrame
    dist_matrix_ : np.ndarray | pd.DataFrame
        symmetric distance matrix, labelled by the columns for a DataFrame
    """

    def __init__(self, max_p: int = 3, max_q: int = 3, d: int = 0, tau: int = 10, arima_kwargs: dict | None = None,
                 n_jobs: int | None = None) -> None:
        self.max_p = max_p
        self.max_q = max_q
        self.d = d
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
            raise ValueError(f"arima_kwargs must not set {', '.join(map(repr, reserved))}; "
                             "use ARMADistance's own parameter")
        series, names = _split_series(X)

        fit_kwargs = {'start_p': min(2, self.max_p), 'start_q': min(2, self.max_q), **arima_kwargs}
        processes = estimate_processes(series, n_workers=self.n_jobs, max_p=self.max_p, max_q=self.max_q, d=self.d,
                                       **fit_kwargs)
        for name, process in zip(names, processes, strict=True):
            if not process.isinvertible:
                raise ValueError(f'Series {name!r} has a non-invertible fitted ARMA model '
                                 f'(AR coefficients {process.arcoefs.tolist()}, '
                                 f'MA coefficients {process.macoefs.tolist()}); '
                                 'the Piccolo distance needs invertible models')

        self.processes_ = dict(zip(names, processes, strict=True)) if isinstance(X, pd.DataFrame) else processes
        self.dist_matrix_ = dist_matrix_parallel(self.processes_, metric_name='piccolo', tau=self.tau)
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
