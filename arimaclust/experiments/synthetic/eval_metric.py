import matplotlib.pyplot as plt
import numpy as np
from arimaclust.metrics.autoregressive import piccolo
from arimaclust.metrics.matrix import dist_matrix_parallel
from arimaclust.autoarima.estimation import estimate_processes
from arimaclust.data.examples import EXAMPLE1,EXAMPLE2
from arimaclust.data.generate import generate_arma_sample, to_arma_process

t = 800
estimation_args = {'max_p': 3,
                   'max_q': 3,
                   'd': 0,
                   'trace': False
                   }

params1, params2 = EXAMPLE2
p1 = to_arma_process(*params1)
p2 = to_arma_process(*params2)
dist = piccolo(p1, p2)
print('Process1', p1)
print('Process2', p2)
print('True distance', dist)

dists = []
tau_range = [2, 3, 5, 10, 15, 20, 25, 30, 35, 40]
for tau in tau_range:
    dists_per_tau = []
    for i in range(10):
        x = generate_arma_sample(*params1, t=t)
        y = generate_arma_sample(*params2, t=t)
        processes = estimate_processes(X=[x, y], **estimation_args)
        metric_args = {'tau': tau}
        dm = dist_matrix_parallel(processes, metric_name='piccolo', **metric_args)
        dist_hat = dm[0][1]
        dists_per_tau.append(dist_hat)
    dists.append(dists_per_tau)

m = np.mean(dists, axis=1)
e = np.std(dists, axis=1)

plt.errorbar(tau_range, m, e, linestyle='None', marker='^')
plt.axhline(y=dist, color='r', linestyle='--')
plt.show()
