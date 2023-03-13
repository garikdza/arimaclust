from functools import partial
import numpy as np
import importlib
import statsmodels.api as sm
from arimaclust.utils import timeit


@timeit
def dist_matrix_parallel(processes, metric_name, **kwargs):
    # X here is a list of processes

    # get metric func
    metrics_module = importlib.import_module('arimaclust.metrics')
    metric = partial(getattr(metrics_module, metric_name), **kwargs)

    n = len(processes)
    out_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(i):
            out_mat[i][j] = metric(processes[i], processes[j], **kwargs)
    return out_mat + out_mat.T


if __name__ == "__main__":
    t = 1000
    arparams = np.array([0.3, -0.2])
    maparams = np.array([-0.5])
    ar = np.r_[1, -arparams]  # add zero-lag and negate
    ma = np.r_[1, maparams]  # add zero-lag
    x_process = sm.tsa.ArmaProcess(ar, ma)
    x = x_process.generate_sample(nsample=t)

    arparams = np.array([0.25, -0.15])
    maparams = np.array([-0.5])
    ar = np.r_[1, -arparams]  # add zero-lag and negate
    ma = np.r_[1, maparams]  # add zero-lag
    y_process = sm.tsa.ArmaProcess(ar, ma)
    y = y_process.generate_sample(nsample=t)

    arparams = np.array([0.8, -0.5, 0.2])
    maparams = np.array([0.6, -0.3, 0.1])
    ar = np.r_[1, -arparams]  # add zero-lag and negate
    ma = np.r_[1, maparams]  # add zero-lag
    z_process = sm.tsa.ArmaProcess(ar, ma)
    z = z_process.generate_sample(nsample=t)

    arparams = np.array([0.7, -0.4, 0.1])
    maparams = np.array([0.5, -0.2])
    ar = np.r_[1, -arparams]  # add zero-lag and negate
    ma = np.r_[1, maparams]  # add zero-lag
    k_process = sm.tsa.ArmaProcess(ar, ma)
    k = k_process.generate_sample(nsample=t)

    # m = compute_metric(X=[x] * 100, metric_name='piccolo', x_max_orders=(2, 2), y_max_orders=(2, 2), tau=100)
    # m = sklearn_pair(X=[x] * 100, metric_name='piccolo', x_max_orders=(2, 2), y_max_orders=(2, 2), tau=100)
    m = dist_matrix_parallel(X=[x, y, z, k], metric_name='piccolo', max_orders=(2, 2), tau=100)
