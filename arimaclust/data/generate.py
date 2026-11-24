from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from arimaclust.autoarima.estimation import to_arma_process

# observations drawn and discarded before each series, so it starts close to the stationary distribution
BURNIN = 100


def generate_arma_sample(ar_params: ArrayLike, ma_params: ArrayLike, length: int, burnin: int = BURNIN,
                         random_state: int | np.random.Generator | None = None) -> np.ndarray:
    """
    Draw one series from a stationary ARMA process with standard Gaussian noise.

    Parameters
    ----------
    ar_params : ArrayLike
        autoregressive parameters phi_1..phi_p, in the sign of x_t = sum phi_k x_{t-k} + ...
    ma_params : ArrayLike
        moving average parameters theta_1..theta_q, in the sign of ... + e_t + sum theta_k e_{t-k}
    length : int
        number of observations returned
    burnin : int
        number of observations drawn first and discarded, so the series starts close to the stationary
        distribution rather than at 0
    random_state : int | np.random.Generator | None
        seed or generator of the noise; ``None`` draws fresh entropy

    Returns
    -------
    x : np.ndarray
        the series, of shape ``(length,)``
    """
    rng = np.random.default_rng(random_state)
    return to_arma_process(ar_params, ma_params).generate_sample(nsample=length, burnin=burnin,
                                                                   distrvs=rng.standard_normal)


def generate_clusters(params: Sequence[tuple[ArrayLike, ArrayLike]], length: int, n_per_cluster: int | Sequence[int],
                      random_state: int | np.random.Generator | None = None) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Draw a labelled clustering data set: each cluster is a group of series from one ARMA process.

    Parameters
    ----------
    params : Sequence[tuple[ArrayLike, ArrayLike]]
        ``(ar_params, ma_params)`` of each cluster's process, as in ``generate_arma_sample``
    length : int
        number of observations in every series
    n_per_cluster : int | Sequence[int]
        number of series in each cluster, the same for all or one count per cluster
    random_state : int | np.random.Generator | None
        seed or generator of all the draws; ``None`` draws fresh entropy

    Returns
    -------
    X, y : tuple[list[np.ndarray], np.ndarray]
        the series, grouped by cluster in the order of ``params``, and the cluster of each
    """
    if isinstance(n_per_cluster, int | np.integer):
        counts = [int(n_per_cluster)] * len(params)
    elif isinstance(n_per_cluster, Sequence) and not isinstance(n_per_cluster, str):
        counts = list(n_per_cluster)
        if len(counts) != len(params):
            raise ValueError(f'n_per_cluster has {len(counts)} counts for {len(params)} clusters')
    else:
        raise TypeError(f'n_per_cluster must be an int or a sequence of ints, got {n_per_cluster!r}')

    rng = np.random.default_rng(random_state)
    X = [generate_arma_sample(ar, ma, length, random_state=rng) for (ar, ma), count in zip(params, counts, strict=True)
         for _ in range(count)]
    y = np.repeat(np.arange(len(params)), counts)
    return X, y
