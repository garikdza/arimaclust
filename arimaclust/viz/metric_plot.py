import numpy as np
import matplotlib.pyplot as plt


def plot_metric_vs_time(t_range=None, n_runs=10):
    if t_range is None:
        t_range = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800]

    results = []
    for t in t_range:
        print(f'========{t}==========')
        res_per_t = []
        for _ in range(n_runs):
            print(f'________{_}_________')

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

            dist_per_run = piccolo(x=x, y=y, x_max_orders=(2, 2), y_max_orders=(2, 2))
            res_per_t.append(dist_per_run)
        results.append(res_per_t)

    m = np.mean(results, axis=1)

    e = np.std(results, axis=1)

    plt.errorbar(t_range, m, e, linestyle='None', marker='^')
    plt.savefig('res.png')
    plt.show()
