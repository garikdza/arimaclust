import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class ConsistentARMAClustering:

    def __init__(self, k, metric='piccolo', underlying_model='ARMA', tau=100, max_orders=(3, 3)):
        if metric != 'piccolo':
            raise ValueError
        if underlying_model != 'ARMA':
            raise ValueError
        self.metric = metric
        self.k = k
        self.underlying_model = underlying_model
        self.tau = tau
        self.labels = None
        self.max_orders = max_orders

    def d(self, x, y):
        # model_x = self.fit_single_arma(x)
        # model_y = self.fit_single_arma(y)
        # x_ar = self.arma_to_ar(model_x)
        # y_ar = self.arma_to_ar(model_y)
        return np.linalg.norm(x - y)

    def arma_to_ar(self, x):
        pass

    def fit(self, X):

        centroids = [X[0]]
        for k in range(1, self.k):
            d_for_sample = []
            for x in X:
                d_from_centroids = []
                for j in range(k):
                    d_from_centroids.append(self.d(x=x, y=centroids[j]))
                d_for_sample.append(min(d_from_centroids))
            c_k = np.argmax(d_for_sample)

            centroids.append(X[c_k])

        labels = []
        for x in X:
            d_from_centroids = []
            for x_c in centroids:
                d_from_centroids.append(self.d(x, x_c))

            labels.append(np.argmin(d_from_centroids))
        self.labels_ = labels
        return self


kappa = 3
X, y = make_blobs(n_samples=100, centers=kappa, cluster_std=0.5, n_features=2)
clustering_model = ConsistentARMAClustering(k=kappa)
clustering_model.fit(X)
labels = clustering_model.labels_
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
