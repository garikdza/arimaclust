import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from arimaclust.metrics.matrix import dist_matrix_parallel
from arimaclust.data.generate import generate_arma_sample


class ConsistentARMAClustering:

    def __init__(self, k, metric='piccolo', underlying_model='ARMA', tau=100, max_orders=(3, 3)):
        if metric == 'piccolo':
            metric_args = {'max_orders': max_orders,
                           'tau': tau
                           }
        else:
            raise ValueError
        if underlying_model != 'ARMA':
            raise ValueError
        self.metric = metric
        self.metric_args = metric_args
        self.k = k
        self.underlying_model = underlying_model
        self.tau = tau
        self.labels = None
        self.D = None
        self.max_orders = max_orders

    @staticmethod
    def _check_array(X):
        pass

    def compute_distance_matrix(self, X):
        self._check_array(X=X)
        self.D = dist_matrix_parallel(X=X, metric_name=self.metric, **self.metric_args)

    def d(self, x, y):
        return compute_metric(X=[x, y], metric_name=self.metric, **self.metric_args)

    def arma_to_ar(self, x):
        pass

    def fit(self, X):

        self.compute_distance_matrix(X=X)

        c_inds = [0]
        for k in range(1, self.k):
            d_for_sample = []
            for i, x in enumerate(X):
                d_from_centroids = []
                for j in range(k):
                    d_from_centroids.append(self.D[i, c_inds[j]])
                d_for_sample.append(np.min(d_from_centroids).item())

            c_k = np.argmax(d_for_sample)

            c_inds.append(c_k)

        labels = []
        for i, x in enumerate(X):
            d_from_centroids = []
            for c_k in c_inds:
                d_from_centroids.append(self.D[i, c_k])

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

    clustering_model = ConsistentARMAClustering(k=kappa)
    clustering_model.fit(X)
    print(clustering_model.labels_)
