import numpy as np
from arimaclust.models.base import ClusteringModel


class NearestNeighbourModel(ClusteringModel):

    def __init__(self, alpha, metric='piccolo', underlying_process='ARMA', process_args=None, metric_args=None):
        super().__init__(metric=metric, underlying_process=underlying_process, metric_args=metric_args,
                         process_args=process_args)

        self.metric = metric
        self.alpha = alpha
        self.underlying_process = underlying_process
        if metric_args is None:
            self.metric_args = {'tau': 10
                                }
        if process_args is None:
            self.process_args = {'max_orders': (3, 3),
                                 'd': 0
                                 }

    def d_c(self, c1, c2):
        dists = []
        for i in c1:
            for j in c2:
                d = self.dist_matrix[i][j]
                dists.append(d)

        return min(dists)

    def fit(self, X, **kwargs):

        # compute dist matrix
        self.dist_matrix = self.get_distance_matrix(X=X, metric=self.metric, underlying_process=self.underlying_process,
                                                    metric_args=self.metric_args,
                                                    process_args=self.process_args)
        n = len(X)

        # Create an empty list to store the clusters
        clusters = []

        # Iterate over all pairs of points
        for i in range(n):
            single = True
            if i + 1 != n:
                for j in range(i + 1, n):
                    # If the distance between the points is smaller than alpha
                    if self.dist_matrix[i, j] <= self.alpha:
                        # Check if the points are already in a cluster
                        single = False
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
                if single:
                    already_merged = True
                    for cluster in clusters:
                        if i in cluster:
                            already_merged = False
                            break
                    if already_merged:
                        clusters.append({i})

        labels = np.ones(n) * (-1)
        for i, cluster in enumerate(clusters):
            labels[list(cluster)] = i
        self.labels_ = labels


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.metrics.pairwise import euclidean_distances
    import matplotlib.pyplot as plt

    X, Y = make_blobs(n_samples=2000, cluster_std=1.5, centers=2)
    d = euclidean_distances(X)
    model = NearestNeighbourModel(alpha=4.5, metric='precomputed'
                                  )
    # D = pairwise_distances(X)
    # dist_matrix = np.load('../experiments/exchange/matrix.npy')
    model.fit(X=d)
    plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
    plt.show()
