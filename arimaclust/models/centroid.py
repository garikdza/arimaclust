import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from arimaclust.metrics.matrix import dist_matrix_parallel
from arimaclust.data.generate import generate_arma_sample
from arimaclust.models.base import ClusteringModel


class ConsistentARMAClustering(ClusteringModel):

    def __init__(self, k, metric='piccolo', underlying_process='ARMA', process_args=None, metric_args=None):
        super().__init__(metric=metric, underlying_process=underlying_process, metric_args=metric_args,
                         process_args=process_args)

        self.metric = metric
        self.k = k
        self.underlying_process = underlying_process
        if metric_args is None:
            self.metric_args = {'tau': 10
                                }
        if process_args is None:
            self.process_args = {'max_p': 3,
                                 'max_q': 3,
                                 'd': 0
                                 }

    def fit(self, X, **kwargs):
        self.dist_matrix = self.get_distance_matrix(X=X, metric=self.metric, underlying_process=self.underlying_process,
                                                    metric_args=self.metric_args,
                                                    process_args=self.process_args)

        c_inds = [0]
        for k in range(1, self.k):
            d_for_sample = []
            for i, x in enumerate(X):
                d_from_centroids = []
                for j in range(k):
                    d_from_centroids.append(self.dist_matrix[i, c_inds[j]])
                d_for_sample.append(np.min(d_from_centroids).item())

            c_k = np.argmax(d_for_sample)

            c_inds.append(c_k)

        labels = []
        for i, x in enumerate(X):
            d_from_centroids = []
            for c_k in c_inds:
                d_from_centroids.append(self.dist_matrix[i, c_k])

            labels.append(np.argmin(d_from_centroids))
        self.labels_ = np.array(labels)
        return self


if __name__ == '__main__':
    kappa = 2

    X = []

    ar_params = np.array([0.3, -0.2])
    ma_params = np.array([-0.5])

    for i in range(5):
        x = generate_arma_sample(ar_params=ar_params, ma_params=ma_params, t=100)
        X.append(x)

    ar_params = np.array([0.7, -0.4, 0.1])
    ma_params = np.array([0.5, -0.2])

    for i in range(5):
        x = generate_arma_sample(ar_params=ar_params, ma_params=ma_params, t=100)
        X.append(x)

    print(len(X))

    clustering_model = ConsistentARMAClustering(k=kappa, process_args={'d': None, 'max_p': 3, 'max_q': 2},
                                                metric_args={'tau': 10})
    clustering_model.fit(X)
    print(clustering_model.labels_)
    print(clustering_model.processes_)
