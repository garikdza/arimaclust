from arimaclust.metrics.matrix import dist_matrix_parallel


class ClusteringModel:

    def __init__(self, metric, underlying_model='ARMA', **kwargs):
        self.metric = metric
        self.underlying_model = underlying_model
        self.dist_matrix = None

    @staticmethod
    def _check_array(X):
        pass

    @staticmethod
    def _check_distance_matrix(X):
        pass

    def get_distance_matrix(self, X, metric, **kwargs):
        if metric == 'precomputed':
            # check symmetric
            self._check_distance_matrix(X=X)
            self.dist_matrix = X
        else:
            self._check_array(X=X)
            self.dist_matrix = dist_matrix_parallel(X=X, metric_name=metric, **kwargs)

    def fit(self, X, **kwargs):
        pass
