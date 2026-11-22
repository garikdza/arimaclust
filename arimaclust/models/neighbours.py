import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from arimaclust.models.base import ClusteringModel


class NearestNeighbourModel(ClusteringModel):
    """
    Threshold single-linkage: samples closer than ``alpha`` are linked, and clusters are
    the connected components of that graph (isolated samples form their own cluster).
    The number of clusters is not needed.
    """

    def __init__(self, alpha, metric='piccolo', underlying_process='ARMA', process_args=None, metric_args=None):
        super().__init__(metric=metric, underlying_process=underlying_process, metric_args=metric_args,
                         process_args=process_args)
        self.alpha = alpha

    def fit(self, X, **kwargs):
        self.dist_matrix = self.get_distance_matrix(X)
        dm = self.dist_matrix.values if isinstance(self.dist_matrix, pd.DataFrame) else np.asarray(self.dist_matrix)
        adjacency = csr_matrix(dm <= self.alpha)
        _, self.labels_ = connected_components(adjacency, directed=False)
        return self
