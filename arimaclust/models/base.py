from arimaclust.autoarima.estimation import estimate_processes
from arimaclust.metrics.matrix import dist_matrix_parallel

DEFAULT_METRIC_ARGS = {'tau': 10}
DEFAULT_PROCESS_ARGS = {'max_p': 3, 'max_q': 3, 'd': 0}


class ClusteringModel:
    """
    Base class: estimate one process per series, build a distance matrix, cluster on it.

    ``metric='precomputed'`` treats ``X`` in ``fit`` as a ready distance matrix.
    """

    def __init__(self, metric='piccolo', metric_args=None, process_args=None, underlying_process='ARMA'):
        self.metric = metric
        self.metric_args = dict(DEFAULT_METRIC_ARGS if metric_args is None else metric_args)
        self.process_args = dict(DEFAULT_PROCESS_ARGS if process_args is None else process_args)
        self.underlying_process = underlying_process
        self.dist_matrix = None
        self.processes_ = None
        self.labels_ = None

    def get_distance_matrix(self, X):
        if self.metric == 'precomputed':
            return X
        self.processes_ = estimate_processes(X=X, underlying_process=self.underlying_process, **self.process_args)
        return dist_matrix_parallel(processes=self.processes_, metric_name=self.metric, **self.metric_args)

    def fit(self, X, **kwargs):
        raise NotImplementedError

    def fit_predict(self, X, **kwargs):
        return self.fit(X, **kwargs).labels_
