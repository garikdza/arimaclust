from collections.abc import Hashable, Sequence

import numpy as np
import pmdarima as pm
from joblib import Parallel, delayed

from arimaclust.modeling.model import FittedARIMA
from arimaclust.utils import timeit


def fit_arima(x: np.ndarray, information_criterion: str = 'bic', **kwargs) -> FittedARIMA:
    """
    Select and fit an ARIMA model to ``x`` with ``pmdarima.auto_arima``.

    BIC is the default because consistent order selection (and hence the consistency
    guarantees of the clustering algorithm) requires it; AIC is not consistent.
    Remaining ``kwargs`` (``max_p``, ``max_q``, ``d``, ...) go to ``auto_arima`` and
    override its fixed options (``error_action``, ``stepwise``, ``suppress_warnings``).

    Returns
    -------
    model : FittedARIMA
        the selected model, holding ``x``
    """
    options = {'error_action': 'ignore', 'stepwise': True, 'suppress_warnings': True, **kwargs}
    model = pm.auto_arima(x, information_criterion=information_criterion, **options)
    return FittedARIMA(model.arparams(), model.maparams(), int(model.order[1]), x)


def _fit_named(x: np.ndarray, name: Hashable, **kwargs) -> FittedARIMA:
    """``fit_arima``, with a failure re-raised under the series' name."""
    try:
        return fit_arima(x, **kwargs)
    except Exception as error:
        raise ValueError(f'auto_arima could not fit series {name!r}: {error}') from error


@timeit
def fit_arimas(X: list[np.ndarray], n_jobs: int | None = None, names: Sequence[Hashable] | None = None,
               **kwargs) -> list[FittedARIMA]:
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
    models : list[FittedARIMA]
        the fitted model of each series, in input order
    """
    names = range(len(X)) if names is None else names
    return Parallel(n_jobs=n_jobs)(delayed(_fit_named)(x, name, **kwargs) for x, name in zip(X, names, strict=True))
