import numpy as np
import pandas as pd

from arimaclust.models.base import ClusteringModel


class ConsistentARMAClustering(ClusteringModel):
    """
    Farthest-point initialisation followed by nearest-centre assignment, with a known
    number of clusters ``k`` (Khaleghi et al., JMLR 2016, applied to the Piccolo distance;
    weak consistency for ARMA processes: Adamyan, J. Contemp. Math. Anal. 2023).
    """

    def __init__(self, k, metric='piccolo', underlying_process='ARMA', process_args=None, metric_args=None):
        super().__init__(metric=metric, underlying_process=underlying_process, metric_args=metric_args,
                         process_args=process_args)
        self.k = k
        self.centroids_ = None

    def fit(self, X, **kwargs):
        self.dist_matrix = self.get_distance_matrix(X)
        dm = self.dist_matrix.values if isinstance(self.dist_matrix, pd.DataFrame) else np.asarray(self.dist_matrix)
        n = dm.shape[0]
        if self.k > n:
            raise ValueError(f'k={self.k} is larger than the number of samples ({n})')

        centroids = [0]
        for _ in range(1, self.k):
            d_to_nearest_centroid = dm[:, centroids].min(axis=1)
            centroids.append(int(np.argmax(d_to_nearest_centroid)))

        self.centroids_ = np.array(centroids)
        self.labels_ = dm[:, centroids].argmin(axis=1)
        return self
