from typing import Union, Tuple
import numpy as np
import pmdarima as pm
import statsmodels.api as sm
import matplotlib.pyplot as plt


def arma2ar(model: Union[Tuple, pm.arima.arima.ARIMA], tau: int = 100):
    if isinstance(model, tuple):
        if len(model) != 2:
            raise ValueError('Please provide tuple of AR and MA params')
        ar_params, ma_params = model
        ar_params = np.array(ar_params)
        ma_params = np.array(ma_params)

    elif isinstance(model, pm.arima.arima.ARIMA):
        ar_params, ma_params = model.arparams(), model.maparams()
    else:
        raise ValueError('Argument model must be tuple of parameters or pmdarima ARIMA model')

    ar = np.r_[1, -ar_params]  # add zero-lag and negate
    ma = np.r_[1, ma_params]  # add zero-lag
    process = sm.tsa.ArmaProcess(ar, ma)
    if not process.isinvertible:
        raise ValueError('Provided model is not invertible')
    return process.arma2ar(lags=tau)


def piccolo(x, y, x_max_orders, y_max_orders, tau=100, **kwargs):
    if 'information_criterion' not in kwargs:
        information_criterion = 'bic'
    else:
        information_criterion = kwargs['information_criterion']

    x_max_p, x_max_q = x_max_orders
    y_max_p, y_max_q = y_max_orders

    x_model = pm.auto_arima(x, start_p=1, start_q=1, max_p=x_max_p, max_q=x_max_q, d=0,
                            error_action='ignore', stepwise=True,
                            suppress_warnings=True,
                            seasonal=False, information_criterion=information_criterion)

    x_ar = arma2ar(model=x_model, tau=tau)

    y_model = pm.auto_arima(y, start_p=1, start_q=1, max_p=y_max_p, max_q=y_max_q,
                            error_action='ignore', d=0, stepwise=True,
                            suppress_warnings=True,
                            seasonal=False, information_criterion=information_criterion)
    y_ar = arma2ar(model=y_model, tau=tau)
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
