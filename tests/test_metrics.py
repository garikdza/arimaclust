import numpy as np
import pandas as pd
import pmdarima as pm
import pytest
import statsmodels.api as sm

from arimaclust.autoarima.estimation import to_arma_process
from arimaclust.data.examples import EXAMPLE1, EXAMPLE2
from arimaclust.data.generate import generate_arma_sample
from arimaclust.metrics.autoregressive import (
    SeriesModel,
    arma2ar,
    cepstral,
    cepstral_coefficients,
    mahalanobis,
    piccolo,
)
from arimaclust.metrics.matrix import dist_matrix_parallel

X_SERIES = np.array([1., 2., 3., 4.])
Y_SERIES = np.array([1., -1., 1., -1., 1., -1.])


@pytest.mark.parametrize('model', [
    pytest.param(([0.5], [0.3]), id='tuple'),
    pytest.param(to_arma_process([0.5], [0.3]), id='arma-process'),
])
def test_arma2ar_input_types_match_hand_values(model: tuple | sm.tsa.ArmaProcess):
    # pi(B) = (1 - 0.5B) / (1 + 0.3B) = (1 - 0.5B)(1 - 0.3B + 0.09B^2 - 0.027B^3 ...) = 1 - 0.8B + 0.24B^2 - 0.072B^3,
    # the lag polynomial of pi_k = (phi + theta)(-theta)^(k-1) = 0.8, -0.24, 0.072
    np.testing.assert_allclose(arma2ar(model, tau=4), [1, -0.8, 0.24, -0.072], rtol=1e-12)


def test_arma2ar_fitted_pmdarima_model():
    np.random.seed(0)
    model = pm.ARIMA(order=(1, 0, 1)).fit(generate_arma_sample([0.5], [0.3], 300))
    phi, theta = model.arparams()[0], model.maparams()[0]
    ar = arma2ar(model, tau=4)
    # the same ARMA(1,1) closed form as above, with the fitted phi and theta
    np.testing.assert_allclose(ar, np.r_[1, -(phi + theta) * (-theta) ** np.arange(3)], rtol=1e-12)
    np.testing.assert_allclose(ar, arma2ar(to_arma_process(model.arparams(), model.maparams()), tau=4), rtol=1e-12)


@pytest.mark.parametrize(('model', 'error', 'match'), [
    pytest.param(([0.5], [0.3], [0.1]), ValueError, 'tuple of AR and MA params', id='tuple-length'),
    pytest.param([[0.5], [0.3]], TypeError, 'must be tuple of parameters', id='list'),
    # MA root 1/2.0 lies inside the unit circle
    pytest.param(([0.5], [2.0]), ValueError, 'not invertible', id='non-invertible'),
])
def test_arma2ar_bad_input_raises(model: tuple | list, error: type[Exception], match: str):
    with pytest.raises(error, match=match):
        arma2ar(model)


@pytest.mark.parametrize(('ar_params', 'ma_params', 'expected'), [
    # pi = (0.8, -0.24, 0.072); c1 = 0.8; c2 = -0.24 + 1/2*0.8*0.8 = 0.08;
    # c3 = 0.072 + 1/3*0.8*(-0.24) + 2/3*0.08*0.8 = 0.152/3, which is (phi^n - (-theta)^n)/n
    pytest.param([0.5], [0.3], [0.8, 0.08, 0.152 / 3], id='arma11'),
    # c1 = 0.5; c2 = -0.2 + 1/2*0.5*0.5 = -0.075; c3 = 1/3*0.5*(-0.2) + 2/3*(-0.075)*0.5 = -0.175/3
    pytest.param([0.5, -0.2], [], [0.5, -0.075, -0.175 / 3], id='ar2'),
])
def test_cepstral_coefficients_match_hand_values(ar_params: list, ma_params: list, expected: list):
    np.testing.assert_allclose(cepstral_coefficients(to_arma_process(ar_params, ma_params), tau=3), expected,
                               rtol=1e-12)


def test_cepstral_ar1_closed_form():
    # an AR(1) has cepstral coefficients phi^n / n
    expected = np.sqrt(sum((0.6 ** n / n - (-0.3) ** n / n) ** 2 for n in range(1, 6)))
    assert cepstral(to_arma_process([0.6], []), to_arma_process([-0.3], []), tau=5) == pytest.approx(expected,
                                                                                                    rel=1e-10)


def test_cepstral_zero_on_self_and_symmetric():
    p, q = (to_arma_process(*params) for params in EXAMPLE1)
    assert cepstral(p, p) == 0
    assert cepstral(p, q) > 0
    assert cepstral(p, q) == cepstral(q, p)


def test_mahalanobis_tau1_hand_value():
    # biased demeaned autocovariances: gamma_x = (5/4, 5/16), gamma_y = (1, -5/6)
    # sigma^2 = gamma_0 (1 + phi^2) - 2 phi gamma_1: sigma_x^2 = 6/5, sigma_y^2 = 5/12
    # V = sigma_x^2 / (gamma_x0 * 4) + sigma_y^2 / (gamma_y0 * 6) = 6/25 + 5/72 = 557/1800
    # D = (0.4 - (-0.5))^2 / V = 1458/557
    x = SeriesModel(X_SERIES, to_arma_process([0.4], []))
    y = SeriesModel(Y_SERIES, to_arma_process([-0.5], []))
    assert mahalanobis(x, y, tau=1) == pytest.approx(1458 / 557, rel=1e-12)


def test_mahalanobis_tau2_hand_value():
    # gamma_x = (5/4, 5/16, -3/8), gamma_y = (1, -5/6, 2/3); y's AR(1) is read as pi = (-0.5, 0)
    # sigma_x^2 = a' Gamma_3 a with a = (1, -0.5, 0.2):
    #   1.25 * 1.29 + 2 (-0.5 * 5/16 + 0.2 * (-3/8) - 0.1 * 5/16) = 87/80
    # sigma_y^2 = 5/12; R^-1 = [[g0, -g1], [-g1, g0]] / (g0^2 - g1^2)
    # V = sigma_x^2 Rx^-1 / 4 + sigma_y^2 Ry^-1 / 6 = [[1263/2750, 542/4125], [542/4125, 1263/2750]]
    # d = (1, -0.2), D = d' V^-1 d = 656124/239663
    x = SeriesModel(X_SERIES, to_arma_process([0.5, -0.2], []))
    y = SeriesModel(Y_SERIES, to_arma_process([-0.5], []))
    assert mahalanobis(x, y, tau=2) == pytest.approx(656124 / 239663, rel=1e-12)


def test_mahalanobis_zero_for_equal_coefficients_and_symmetric():
    process = to_arma_process([0.4], [])
    assert mahalanobis(SeriesModel(X_SERIES, process), SeriesModel(Y_SERIES, process), tau=1) == 0

    x = SeriesModel(X_SERIES, to_arma_process([0.5, -0.2], []))
    y = SeriesModel(Y_SERIES, to_arma_process([-0.5], []))
    assert mahalanobis(x, y, tau=2) >= 0
    assert mahalanobis(x, y, tau=2) == pytest.approx(mahalanobis(y, x, tau=2), rel=1e-12)


def test_mahalanobis_rejects_bare_processes():
    process = to_arma_process([0.4], [])
    with pytest.raises(TypeError, match='SeriesModel'):
        mahalanobis(process, SeriesModel(X_SERIES, process), tau=1)


@pytest.mark.parametrize('as_dict', [pytest.param(False, id='list'), pytest.param(True, id='dict')])
def test_dist_matrix_entries_and_parallel_equality(as_dict: bool):
    p1, p2 = (to_arma_process(*params) for params in EXAMPLE2)
    items = [p1, p2, p1, to_arma_process(*EXAMPLE1[1])]
    data = dict(zip('abcd', items, strict=True)) if as_dict else items

    serial = dist_matrix_parallel(data, metric_name='piccolo', tau=100)
    parallel = dist_matrix_parallel(data, metric_name='piccolo', n_jobs=2, tau=100)
    if as_dict:
        for D in (serial, parallel):
            assert isinstance(D, pd.DataFrame)
            assert list(D.index) == list(D.columns) == list('abcd')
        serial, parallel = serial.to_numpy(), parallel.to_numpy()
    np.testing.assert_array_equal(serial, parallel)

    assert serial[1, 0] == serial[2, 1] == pytest.approx(0.0577350269, rel=1e-6)
    assert serial[2, 0] == 0
    np.testing.assert_array_equal(np.diag(serial), 0)
    np.testing.assert_array_equal(serial, serial.T)
    for i in range(4):
        for j in range(i):
            assert serial[i, j] == piccolo(items[i], items[j], tau=100)
