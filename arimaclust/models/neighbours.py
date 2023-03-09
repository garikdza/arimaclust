import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
from arimaclust.metrics.matrix import dist_matrix_parallel


class NearestNeighbourModel:

    def __init__(self, alpha=0.5, metric='piccolo', underlying_model='ARMA', tau=100, max_orders=(3, 3)):
        if metric == 'piccolo':
            metric_args = {'max_orders': max_orders,
                           'tau': tau
                           }
        elif metric == 'precomputed':
            metric_args = None
        else:
            raise ValueError
        if underlying_model != 'ARMA':
            raise ValueError
        self.metric = metric
        self.metric_args = metric_args
        self.alpha = alpha
        self.underlying_model = underlying_model
        self.tau = tau
        self.labels_ = None
        self.D = None
        self.max_orders = max_orders

    @staticmethod
    def _check_array(X):
        pass

    def compute_distance_matrix(self, X):
        self._check_array(X=X)
        if self.metric == 'precomputed':
            # check symeric
            self.D = X
        else:
            self.D = dist_matrix_parallel(X=X, metric_name=self.metric, **self.metric_args)

    def d_c(self, c1, c2):
        dists = []
        for i in c1:
            for j in c2:
                d = self.D[i][j]
                dists.append(d)

        return min(dists)

    def fit(self, X):

        # compute dist matrix
        self.compute_distance_matrix(X=X)
        n = len(X)

        # create separate clusters

        # Create an empty list to store the clusters
        clusters = []

        # Iterate over all pairs of points
        for i in range(n):
            single = True
            for j in range(i + 1, n):
                # If the distance between the points is smaller than alpha

                if self.D[i, j] <= self.alpha:
                    # Check if the points are already in a cluster
                    merged = False
                    for cluster in clusters:
                        if i in cluster or j in cluster:
                            # Merge the points into the existing cluster
                            cluster.add(i)
                            cluster.add(j)
                            merged = True
                            break
                    if not merged:
                        # Create a new cluster for the points
                        clusters.append({i, j})
                    single = False
            if single:
                clusters.append({i})
        labels = np.ones(n)*(-1)
        for i, cluster in enumerate(clusters):
            labels[list(cluster)] = i
        self.labels_ = labels


if __name__ == '__main__':
    model = NearestNeighbourModel(alpha=0.2, metric='precomputed'
                                  )
    # D = pairwise_distances(X)
    dist_matrix = np.load('../experiments/exchange/matrix.npy')
    model.fit(X=dist_matrix)

    # plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
    # plt.show()
