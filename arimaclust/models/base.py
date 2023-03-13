from arimaclust.metrics.matrix import dist_matrix_parallel
from arimaclust.autoarima.estimation import estimate_processes


class ClusteringModel:

    def __init__(self, metric, metric_args, process_args, underlying_process='ARMA'):
        self.metric = metric
        self.metric_args = metric_args
        self.underlying_process = underlying_process
        self.process_args = process_args
        self.dist_matrix = None
        self.processes_ = None
        self.labels_ = None

    @staticmethod
    def _check_array(X):
        pass

    @staticmethod
    def _check_distance_matrix(X):
        pass

    @staticmethod
    def estimate_processes(X, underlying_process, **kwargs):
        return estimate_processes(X=X, underlying_model=underlying_process, **kwargs)

    def get_distance_matrix(self, X, metric, underlying_process, metric_args, process_args):
        if metric == 'precomputed':
            # check symmetric
            self._check_distance_matrix(X=X)
            return X
        else:
            self._check_array(X=X)
            self.processes_ = self.estimate_processes(X=X, underlying_process=underlying_process, **process_args)
            return dist_matrix_parallel(processes=self.processes_, metric_name=metric, **metric_args)

    def fit(self, X, **kwargs):
        pass
