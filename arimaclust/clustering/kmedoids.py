import numbers
import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import ConvergenceWarning

from arimaclust.clustering._input import DistanceInputMixin
from arimaclust.metrics import ARMADistance


def _farthest_start(D: np.ndarray, n_clusters: int) -> list[int]:
    """
    Farthest-point medoids from sample 0, stopping early when every sample is at distance 0 from one.

    Parameters
    ----------
    D : np.ndarray
        distance matrix, ``D[j, m]`` the distance from sample ``j`` to medoid ``m``
    n_clusters : int
        number of medoids wanted

    Returns
    -------
    medoids : list[int]
        sample index of each medoid, in selection order
    """
    medoids = [0]
    while len(medoids) < n_clusters:
        d_to_nearest = D[:, medoids].min(axis=1)
        farthest = int(np.argmax(d_to_nearest))
        if d_to_nearest[farthest] <= 0:
            break
        medoids.append(farthest)
    return medoids


def _build_start(D: np.ndarray, n_clusters: int) -> list[int]:
    """
    PAM BUILD medoids, stopping early when no further medoid lowers the total distance.

    Parameters
    ----------
    D : np.ndarray
        distance matrix, ``D[j, m]`` the distance from sample ``j`` to medoid ``m``
    n_clusters : int
        number of medoids wanted

    Returns
    -------
    medoids : list[int]
        sample index of each medoid, in selection order
    """
    medoids = [int(np.argmin(D.sum(axis=0)))]
    while len(medoids) < n_clusters:
        d_to_nearest = D[:, medoids].min(axis=1)
        gain = np.maximum(d_to_nearest[:, None] - D, 0).sum(axis=0)
        gain[medoids] = -np.inf
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            break
        medoids.append(best)
    return medoids


def _best_swap(D: np.ndarray, medoids: list[int]) -> tuple[float, int, int]:
    """
    Find the exchange of a medoid for a non-medoid that lowers the total distance most.

    On a tie, the lowest incoming sample index wins, then the lowest outgoing medoid sample index.

    Parameters
    ----------
    D : np.ndarray
        distance matrix, ``D[j, m]`` the distance from sample ``j`` to medoid ``m``
    medoids : list[int]
        current medoids

    Returns
    -------
    delta : float
        change in total distance made by the exchange, negative when it improves
    incoming : int
        sample index of the non-medoid brought in
    position : int
        position in ``medoids`` of the medoid taken out
    """
    # an infinite last column gives every sample a second-nearest distance, infinite with a single medoid
    dm = np.column_stack([D[:, medoids], np.full(len(D), np.inf)])
    order = np.argsort(dm, axis=1, kind='stable')
    near, second = np.take_along_axis(dm, order[:, :2], axis=1).T
    # delta[h, i]: change in total distance when the medoid at position i is exchanged for sample h
    delta = np.empty((len(D), len(medoids)))
    for i in range(len(medoids)):
        without_i = np.where(order[:, 0] == i, second, near)
        delta[:, i] = (np.minimum(D, without_i[:, None]) - near[:, None]).sum(axis=0)
    delta[medoids] = np.inf
    by_index = np.argsort(medoids, kind='stable')
    incoming, i = np.unravel_index(np.argmin(delta[:, by_index]), delta.shape)
    position = by_index[i]
    return float(delta[incoming, position]), int(incoming), int(position)


def _swap(D: np.ndarray, medoids: list[int], max_iter: int) -> tuple[list[int], int, bool]:
    """
    PAM swaps: in each iteration, make the exchange of a medoid for a non-medoid that lowers the total distance
    most, until none lowers it.

    Parameters
    ----------
    D : np.ndarray
        distance matrix, ``D[j, m]`` the distance from sample ``j`` to medoid ``m``
    medoids : list[int]
        starting medoids, changed in place: each swap replaces one at its position, so the order is kept
    max_iter : int
        largest number of iterations

    Returns
    -------
    medoids : list[int]
        medoids reached
    n_iter : int
        number of iterations run, counting the last one that finds no improving swap
    converged : bool
        whether no swap lowers the total distance any further
    """
    for n_iter in range(1, max_iter + 1):
        delta, incoming, position = _best_swap(D, medoids)
        if not delta < 0:
            return medoids, n_iter, True
        medoids[position] = incoming
    delta, _, _ = _best_swap(D, medoids)
    return medoids, max_iter, not delta < 0


class KMedoids(DistanceInputMixin, ClusterMixin, BaseEstimator):
    """
    k-medoids by PAM (Kaufman & Rousseeuw, *Finding Groups in Data*, 1990): a start of ``n_clusters`` medoids,
    then swaps of a medoid for a non-medoid until none lowers the total distance of the samples to their nearest
    medoid; each sample then goes to its nearest medoid.

    ``init='build'`` starts from PAM BUILD, which takes the sample with the lowest total distance to the others,
    then adds the sample that lowers the total distance most. ``init='farthest'`` starts from farthest-point
    selection at sample 0, the centres of ``ConsistentClustering``. Each swap iteration makes the exchange that
    lowers the total distance most. Ties go to the lowest sample index in the start and in the swaps (the incoming
    sample first, then the outgoing medoid); a sample at equal distance from two medoids goes to the one listed
    first in ``medoids_``.

    The swaps often place the medoids better than farthest-point selection on real data, but ``KMedoids`` has no
    consistency guarantee; ``ConsistentClustering`` is the algorithm with the consistency proof.

    When the samples form fewer than ``n_clusters`` distinct groups, that is when every sample is at distance 0
    from a medoid already chosen, ``fit`` warns (``UserWarning``) and continues with the medoids found. When
    ``max_iter`` iterations leave an improving swap, it warns (``ConvergenceWarning``) and keeps the medoids reached.

    Parameters
    ----------
    n_clusters : int
        number of clusters
    distance : ARMADistance | str | None
        ``ARMADistance`` fitted on the raw series (``None`` uses ``ARMADistance()``), or
        ``'precomputed'`` when ``X`` is a square distance matrix
    init : str
        start, ``'build'`` (PAM BUILD) or ``'farthest'`` (farthest-point selection from sample 0)
    max_iter : int
        largest number of swap iterations

    Attributes
    ----------
    labels_ : np.ndarray
        cluster per sample, its position in ``medoids_``
    medoids_ : np.ndarray
        sample index of each medoid, in selection order (a swap replaces a medoid in place); fewer than
        ``n_clusters`` when the samples form fewer distinct groups
    inertia_ : float
        sum over samples of the distance to their medoid
    n_iter_ : int
        number of swap iterations run, counting the last one that finds no improving swap; ``max_iter`` when
        the cap is reached
    distance_ : ARMADistance | None
        fitted copy of ``distance``, or ``None`` when ``distance='precomputed'``
    """

    def __init__(
        self,
        n_clusters: int = 2,
        distance: ARMADistance | str | None = None,
        init: str = 'build',
        max_iter: int = 300,
    ) -> None:
        self.n_clusters = n_clusters
        self.distance = distance
        self.init = init
        self.max_iter = max_iter

    def fit(self, X: ArrayLike | pd.DataFrame, y: None = None) -> 'KMedoids':
        """
        Cluster the series (or the precomputed distance matrix) ``X``.

        Parameters
        ----------
        X : ArrayLike | pd.DataFrame
            series as accepted by ``ARMADistance``, or a square distance matrix when
            ``distance='precomputed'``
        y : None
            ignored

        Returns
        -------
        self : KMedoids
            fitted estimator
        """
        for name in ('n_clusters', 'max_iter'):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be an integer >= 1, got {value!r}')
        if not isinstance(self.init, str) or self.init not in ('build', 'farthest'):
            raise ValueError(f"init must be 'build' or 'farthest', got {self.init!r}")
        D = np.asarray(self._distance_matrix(X), dtype=float)
        n = D.shape[0]
        if self.n_clusters > n:
            raise ValueError(f'n_clusters={self.n_clusters} is larger than the number of samples, n_samples={n}')

        start = _build_start if self.init == 'build' else _farthest_start
        medoids = start(D, self.n_clusters)
        if len(medoids) < self.n_clusters:
            warnings.warn(
                f'n_clusters={self.n_clusters}, but every sample is at distance 0 from one of the '
                f'first {len(medoids)} medoids, so only {len(medoids)} clusters were found',
                UserWarning,
                stacklevel=2,
            )
        medoids, self.n_iter_, converged = _swap(D, medoids, self.max_iter)
        if not converged:
            warnings.warn(
                f'KMedoids stopped after max_iter={self.max_iter} iterations with an improving swap left; '
                'increase max_iter',
                ConvergenceWarning,
                stacklevel=2,
            )

        dm = D[:, medoids]
        self.medoids_ = np.array(medoids)
        self.labels_ = dm.argmin(axis=1)
        self.inertia_ = float(dm.min(axis=1).sum())
        return self
