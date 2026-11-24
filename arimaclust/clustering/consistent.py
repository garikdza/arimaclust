import numbers
import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClusterMixin

from arimaclust.clustering._input import DistanceInputMixin
from arimaclust.metrics import ARMADistance


class ConsistentClustering(DistanceInputMixin, ClusterMixin, BaseEstimator):
    """
    Farthest-point initialisation followed by nearest-centre assignment, with a known
    number of clusters (Khaleghi et al., JMLR 2016, applied to the Piccolo distance;
    weak consistency for ARMA processes: Adamyan, J. Contemp. Math. Anal. 2023).

    When the samples form fewer than ``n_clusters`` distinct groups, that is when a new centre would be at
    distance 0 from the centres already chosen, ``fit`` warns and returns the clusters found so far; this
    happens when short series all get the same fitted model.

    Parameters
    ----------
    n_clusters : int
        number of clusters
    distance : ARMADistance | str | None
        ``ARMADistance`` fitted on the raw series (``None`` uses ``ARMADistance()``), or
        ``'precomputed'`` when ``X`` is a square distance matrix

    Attributes
    ----------
    labels_ : np.ndarray
        cluster per sample
    centers_ : np.ndarray
        sample index of each cluster centre, in selection order; fewer than ``n_clusters`` when the samples
        form fewer distinct groups
    distance_ : ARMADistance | None
        fitted copy of ``distance``, or ``None`` when ``distance='precomputed'``
    """

    def __init__(self, n_clusters: int = 2, distance: ARMADistance | str | None = None) -> None:
        self.n_clusters = n_clusters
        self.distance = distance

    def fit(self, X: ArrayLike | pd.DataFrame, y: None = None) -> 'ConsistentClustering':
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
        self : ConsistentClustering
            fitted estimator
        """
        if (not isinstance(self.n_clusters, numbers.Integral) or isinstance(self.n_clusters, bool)
                or self.n_clusters < 1):
            raise ValueError(f'n_clusters must be an integer >= 1, got {self.n_clusters!r}')
        dm = self._distance_matrix(X)
        n = dm.shape[0]
        if self.n_clusters > n:
            raise ValueError(f'n_clusters={self.n_clusters} is larger than the number of samples, n_samples={n}')

        centers = [0]
        for _ in range(1, self.n_clusters):
            d_to_nearest_center = dm[:, centers].min(axis=1)
            farthest = int(np.argmax(d_to_nearest_center))
            if d_to_nearest_center[farthest] <= 0:
                warnings.warn(f'n_clusters={self.n_clusters}, but every sample is at distance 0 from one of the '
                              f'first {len(centers)} centres, so only {len(centers)} clusters were found',
                              UserWarning, stacklevel=2)
                break
            centers.append(farthest)

        self.centers_ = np.array(centers)
        self.labels_ = dm[:, centers].argmin(axis=1)
        return self
