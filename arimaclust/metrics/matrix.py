import importlib

import numpy as np
import pandas as pd

from arimaclust.utils import timeit


@timeit
def dist_matrix_parallel(processes: list | dict, metric_name: str, **kwargs) -> np.ndarray | pd.DataFrame:
    """
    Symmetric pairwise distance matrix between fitted processes.

    ``metric_name`` is looked up in ``arimaclust.metrics``; extra kwargs (e.g. ``tau``) go
    to the metric. A dict of processes gives a DataFrame labelled by its keys, a list
    gives an ndarray.
    """
    metrics_module = importlib.import_module('arimaclust.metrics')
    metric = getattr(metrics_module, metric_name)

    data = list(processes.values()) if isinstance(processes, dict) else list(processes)
    n = len(data)
    out_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            out_mat[i, j] = metric(data[i], data[j], **kwargs)
    result = out_mat + out_mat.T

    if isinstance(processes, dict):
        keys = list(processes.keys())
        return pd.DataFrame(data=result, index=keys, columns=keys)
    return result
