from collections.abc import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from arimaclust.utils import timeit


@timeit
def dist_matrix_parallel(
    items: list | dict, metric: Callable[..., float], n_jobs: int | None = None, **kwargs
) -> np.ndarray | pd.DataFrame:
    """
    Symmetric pairwise distance matrix, with the pairs computed in parallel.

    Parameters
    ----------
    items : list | dict
        the models the metric compares
    metric : Callable[..., float]
        distance between two items, for example a value of ``METRICS``
    n_jobs : int | None
        number of joblib workers; ``None`` runs serially
    **kwargs
        passed to the metric (for example ``tau``)

    Returns
    -------
    dist_matrix : np.ndarray | pd.DataFrame
        symmetric distance matrix with a zero diagonal; a DataFrame labelled by the keys for a dict
    """
    data = list(items.values()) if isinstance(items, dict) else list(items)
    n = len(data)
    rows, cols = np.tril_indices(n, k=-1)
    out_mat = np.zeros((n, n))
    out_mat[rows, cols] = Parallel(n_jobs=n_jobs)(
        delayed(metric)(data[i], data[j], **kwargs) for i, j in zip(rows, cols, strict=True)
    )
    result = out_mat + out_mat.T

    if isinstance(items, dict):
        keys = list(items.keys())
        return pd.DataFrame(data=result, index=keys, columns=keys)
    return result
