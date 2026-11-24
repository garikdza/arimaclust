import numbers

import pandas as pd
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator, ClusterMixin

from arimaclust.distance import ARMADistance
from arimaclust.models.distance_input import DistanceInputMixin


class ThresholdClustering(DistanceInputMixin, ClusterMixin, BaseEstimator):
    """
    Threshold single-linkage: samples within distance ``alpha`` are linked, and clusters are
    the connected components of that graph (isolated samples form their own cluster).
    The number of clusters is not needed.

    Parameters
    ----------
    alpha : float | None
        linking threshold, a number >= 0; required, since distances have no natural scale
    distance : ARMADistance | str | None
        ``ARMADistance`` fitted on the raw series (``None`` uses ``ARMADistance()``), or
        ``'precomputed'`` when ``X`` is a square distance matrix

    Attributes
    ----------
    labels_ : np.ndarray
        cluster per sample
    distance_ : ARMADistance | None
        fitted copy of ``distance``, or ``None`` when ``distance='precomputed'``
    """

    def __init__(self, alpha: float | None = None, distance: ARMADistance | str | None = None) -> None:
        self.alpha = alpha
        self.distance = distance

    def fit(self, X: ArrayLike | pd.DataFrame, y: None = None) -> 'ThresholdClustering':
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
        self : ThresholdClustering
            fitted estimator
        """
        if self.alpha is None:
            raise ValueError('alpha must be set: the linking threshold has no default')
        if not isinstance(self.alpha, numbers.Real) or isinstance(self.alpha, bool) or not self.alpha >= 0:
            raise ValueError(f'alpha must be a number >= 0, got {self.alpha!r}')
        dm = self._distance_matrix(X)
        adjacency = csr_matrix(dm <= self.alpha)
        _, self.labels_ = connected_components(adjacency, directed=False)
        return self
