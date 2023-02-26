from typing import Union, Tuple
import numpy as np
import pmdarima as pm
import statsmodels.api as sm
import matplotlib.pyplot as plt


def arma2ar(model: Union[Tuple, pm.arima.arima.ARIMA, sm.tsa.ArmaProcess], tau: int = 100):
    if isinstance(model, tuple):
        if len(model) != 2:
            raise ValueError('Please provide tuple of AR and MA params')
        ar_params, ma_params = model
        ar_params = np.array(ar_params)
        ma_params = np.array(ma_params)
        ar = np.r_[1, -ar_params]  # add zero-lag and negate
        ma = np.r_[1, ma_params]  # add zero-lag
        process = sm.tsa.ArmaProcess(ar, ma)

    elif isinstance(model, pm.arima.arima.ARIMA):
        ar_params, ma_params = model.arparams(), model.maparams()
        ar = np.r_[1, -ar_params]  # add zero-lag and negate
        ma = np.r_[1, ma_params]  # add zero-lag
        process = sm.tsa.ArmaProcess(ar, ma)
    elif isinstance(model, sm.tsa.ArmaProcess):
        process = model
    else:
        raise ValueError('Argument model must be tuple of parameters, pmdarima ARIMA, or sm.ARMA model')

    if not process.isinvertible:
        raise ValueError('Provided model is not invertible')
    return process.arma2ar(lags=tau)


def piccolo(x_process, y_process, tau=100, **kwargs):
    if (not isinstance(x_process, sm.tsa.ArmaProcess)) and (not isinstance(y_process, sm.tsa.ArmaProcess)):
        raise ValueError('Unknown types for processes')

    x_ar = arma2ar(model=x_process, tau=tau)
    y_ar = arma2ar(model=y_process, tau=tau)

    return np.linalg.norm(x_ar - y_ar)


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


if __name__ == '__main__':
    # pi1 = arma2ar(model=([0.3, -0.2], [-0.5]), tau=100)
    # pi2 = arma2ar(model=([0.25, -0.15], [-0.5]), tau=100)
    # np.linalg.norm(pi1 - pi2)
    # pi1 = ar_inf_coeffs(phi=[0.3, -0.2], theta=[-0.5], tau=100)
    # pi2 = ar_inf_coeffs(phi=[0.25, -0.15], theta=[-0.5], tau=100)
    # print(np.linalg.norm(pi1 - pi2))
    plot_metric_vs_time(t_range=[1000, 1500])
