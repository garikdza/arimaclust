import numpy as np
import pmdarima as pm
import statsmodels.api as sm


def arma2ar(model: tuple | pm.arima.arima.ARIMA | sm.tsa.ArmaProcess, tau: int = 100) -> np.ndarray:
    """
    Truncated AR(inf) coefficients of an invertible ARMA model.

    ``model`` can be an ``(ar_params, ma_params)`` tuple, a fitted pmdarima ARIMA,
    or a statsmodels ``ArmaProcess``.
    """
    if isinstance(model, tuple):
        if len(model) != 2:
            raise ValueError('Please provide tuple of AR and MA params')
        ar_params, ma_params = model
    elif isinstance(model, pm.arima.arima.ARIMA):
        ar_params, ma_params = model.arparams(), model.maparams()
    elif isinstance(model, sm.tsa.ArmaProcess):
        ar_params = ma_params = None
    else:
        raise TypeError('Argument model must be tuple of parameters, pmdarima ARIMA, or sm.ArmaProcess')

    if ar_params is None:
        process = model
    else:
        ar = np.r_[1, -np.asarray(ar_params, dtype=float)]  # add zero-lag and negate
        ma = np.r_[1, np.asarray(ma_params, dtype=float)]  # add zero-lag
        process = sm.tsa.ArmaProcess(ar, ma)

    if not process.isinvertible:
        raise ValueError('Provided model is not invertible')
    return process.arma2ar(lags=tau)


def piccolo(x_process: sm.tsa.ArmaProcess, y_process: sm.tsa.ArmaProcess, tau: int = 100) -> float:
    """
    Piccolo (1990) AR distance: Euclidean distance between the AR(inf) coefficients,
    truncated at ``tau`` lags.
    """
    if not (isinstance(x_process, sm.tsa.ArmaProcess) and isinstance(y_process, sm.tsa.ArmaProcess)):
        raise TypeError('Both processes must be statsmodels ArmaProcess instances')

    x_ar = arma2ar(model=x_process, tau=tau)
    y_ar = arma2ar(model=y_process, tau=tau)
    return float(np.linalg.norm(x_ar - y_ar))
