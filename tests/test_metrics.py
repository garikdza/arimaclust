import re
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from arimaclust.metrics import cepstral, cepstral_coefficients, dist_matrix_parallel, mahalanobis, piccolo
from arimaclust.modeling import ARIMA
from synthetic import EXAMPLE1, EXAMPLE2

X_SERIES = np.array([1., 2., 3., 4.])
Y_SERIES = np.array([1., -1., 1., -1., 1., -1.])
P, Q = ARIMA([0.4], []), ARIMA([-0.5], [])
X_ITEM, Y_ITEM = ARIMA([0.4], [], series=X_SERIES), ARIMA([-0.5], [], series=Y_SERIES)


@pytest.mark.parametrize('tau', [0, -1, 2.5, True, '3', None], ids=repr)
@pytest.mark.parametrize(('func', 'args'), [
    pytest.param(piccolo, (P, Q), id='piccolo'),
    pytest.param(cepstral, (P, Q), id='cepstral'),
    pytest.param(cepstral_coefficients, (P,), id='cepstral-coefficients'),
    pytest.param(mahalanobis, (X_ITEM, Y_ITEM), id='mahalanobis'),
])
def test_bad_tau_raises(func: Callable, args: tuple, tau: object):
    # the repr shows the value as given, not tau + 1 (True as True, 2.5 as 2.5)
    with pytest.raises(ValueError, match=rf'tau must be an integer >= 1, got {re.escape(repr(tau))}'):
        func(*args, tau=tau)


def test_numpy_integer_tau_matches_int():
    assert piccolo(P, Q, np.int64(3)) == piccolo(P, Q, 3)
    assert cepstral(P, Q, np.int64(3)) == cepstral(P, Q, 3)
    assert mahalanobis(X_ITEM, Y_ITEM, np.int64(3)) == mahalanobis(X_ITEM, Y_ITEM, 3)


def test_mahalanobis_series_not_longer_than_tau_raises():
    with pytest.raises(ValueError, match='has 4 observations after differencing.*tau=4'):
        mahalanobis(X_ITEM, Y_ITEM, tau=4)
    # n = tau + 1 is the shortest series that works
    assert np.isfinite(mahalanobis(X_ITEM, Y_ITEM, tau=3))


@pytest.mark.parametrize(('x_params', 'y_params', 'tau', 'expected'), [
    # an AR(1) has pi_1 = phi and pi_j = 0 for j > 1
    pytest.param(([0.5], []), ([0.3], []), 1, 0.2, id='ar1-tau1'),
    # an MA(1) has pi_j = -(-theta)^j: pi = (0.9, -0.81, ...) and (0.5, -0.25, ...)
    pytest.param(([], [0.9]), ([], [0.5]), 2, np.hypot(0.4, 0.56), id='ma1-tau2'),
    pytest.param(([], [0.9]), ([], [0.5]), 10,
                 np.sqrt(sum(((-0.9) ** j - (-0.5) ** j) ** 2 for j in range(1, 11))), id='ma1-tau10'),
])
def test_piccolo_matches_closed_form(x_params: tuple, y_params: tuple, tau: int, expected: float):
    distance = piccolo(ARIMA(*x_params), ARIMA(*y_params), tau=tau)
    assert distance == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(('ar_params', 'ma_params', 'expected'), [
    # pi = (0.8, -0.24, 0.072); c1 = 0.8; c2 = -0.24 + 1/2*0.8*0.8 = 0.08;
    # c3 = 0.072 + 1/3*0.8*(-0.24) + 2/3*0.08*0.8 = 0.152/3, which is (phi^n - (-theta)^n)/n
    pytest.param([0.5], [0.3], [0.8, 0.08, 0.152 / 3], id='arma11'),
    # c1 = 0.5; c2 = -0.2 + 1/2*0.5*0.5 = -0.075; c3 = 1/3*0.5*(-0.2) + 2/3*(-0.075)*0.5 = -0.175/3
    pytest.param([0.5, -0.2], [], [0.5, -0.075, -0.175 / 3], id='ar2'),
])
def test_cepstral_coefficients_match_hand_values(ar_params: list, ma_params: list, expected: list):
    np.testing.assert_allclose(cepstral_coefficients(ARIMA(ar_params, ma_params), tau=3), expected,
                               rtol=1e-12)


def test_cepstral_ar1_closed_form():
    # an AR(1) has cepstral coefficients phi^n / n
    expected = np.sqrt(sum((0.6 ** n / n - (-0.3) ** n / n) ** 2 for n in range(1, 6)))
    assert cepstral(ARIMA([0.6], []), ARIMA([-0.3], []), tau=5) == pytest.approx(expected,
                                                                                                    rel=1e-10)


def test_cepstral_zero_on_self_and_symmetric():
    p, q = (ARIMA(*params) for params in EXAMPLE1)
    assert cepstral(p, p) == 0
    assert cepstral(p, q) > 0
    assert cepstral(p, q) == cepstral(q, p)


def test_mahalanobis_tau1_hand_value():
    # biased demeaned autocovariances: gamma_x = (5/4, 5/16), gamma_y = (1, -5/6)
    # sigma^2 = gamma_0 (1 + phi^2) - 2 phi gamma_1: sigma_x^2 = 6/5, sigma_y^2 = 5/12
    # V = sigma_x^2 / (gamma_x0 * 4) + sigma_y^2 / (gamma_y0 * 6) = 6/25 + 5/72 = 557/1800
    # D = (0.4 - (-0.5))^2 / V = 1458/557
    x = ARIMA([0.4], [], series=X_SERIES)
    y = ARIMA([-0.5], [], series=Y_SERIES)
    assert mahalanobis(x, y, tau=1) == pytest.approx(1458 / 557, rel=1e-12)


def test_mahalanobis_tau2_hand_value():
    # gamma_x = (5/4, 5/16, -3/8), gamma_y = (1, -5/6, 2/3); y's AR(1) is read as pi = (-0.5, 0)
    # sigma_x^2 = a' Gamma_3 a with a = (1, -0.5, 0.2):
    #   1.25 * 1.29 + 2 (-0.5 * 5/16 + 0.2 * (-3/8) - 0.1 * 5/16) = 87/80
    # sigma_y^2 = 5/12; R^-1 = [[g0, -g1], [-g1, g0]] / (g0^2 - g1^2)
    # V = sigma_x^2 Rx^-1 / 4 + sigma_y^2 Ry^-1 / 6 = [[1263/2750, 542/4125], [542/4125, 1263/2750]]
    # d = (1, -0.2), D = d' V^-1 d = 656124/239663
    x = ARIMA([0.5, -0.2], [], series=X_SERIES)
    y = ARIMA([-0.5], [], series=Y_SERIES)
    assert mahalanobis(x, y, tau=2) == pytest.approx(656124 / 239663, rel=1e-12)


def test_mahalanobis_zero_for_equal_coefficients_and_symmetric():
    assert mahalanobis(ARIMA([0.4], [], series=X_SERIES), ARIMA([0.4], [], series=Y_SERIES), tau=1) == 0

    x = ARIMA([0.5, -0.2], [], series=X_SERIES)
    y = ARIMA([-0.5], [], series=Y_SERIES)
    assert mahalanobis(x, y, tau=2) >= 0
    assert mahalanobis(x, y, tau=2) == pytest.approx(mahalanobis(y, x, tau=2), rel=1e-12)


def test_mahalanobis_needs_the_series():
    with pytest.raises(ValueError, match='needs the series'):
        mahalanobis(ARIMA([0.4], []), X_ITEM, tau=1)


def test_mahalanobis_differences_each_series_by_its_own_order():
    # an ARIMA(1,1,0) with a cumulated series equals the ARMA(1,0) with the original series
    integrated = ARIMA([0.4], [], d=1, series=np.r_[0, np.cumsum(X_SERIES)])
    assert mahalanobis(integrated, Y_ITEM, tau=1) == mahalanobis(X_ITEM, Y_ITEM, tau=1)


@pytest.mark.parametrize('metric', [piccolo, cepstral, mahalanobis])
def test_metrics_reject_other_types(metric: Callable):
    with pytest.raises(TypeError, match='ARIMA instances, got ArmaProcess'):
        metric(P.arma_process, X_ITEM, tau=1)


@pytest.mark.parametrize('as_dict', [pytest.param(False, id='list'), pytest.param(True, id='dict')])
def test_dist_matrix_entries_and_parallel_equality(as_dict: bool):
    p1, p2 = (ARIMA(*params) for params in EXAMPLE2)
    items = [p1, p2, p1, ARIMA(*EXAMPLE1[1])]
    data = dict(zip('abcd', items, strict=True)) if as_dict else items

    serial = dist_matrix_parallel(data, metric=piccolo, tau=100)
    parallel = dist_matrix_parallel(data, metric=piccolo, n_jobs=2, tau=100)
    if as_dict:
        for D in (serial, parallel):
            assert isinstance(D, pd.DataFrame)
            assert list(D.index) == list(D.columns) == list('abcd')
        serial, parallel = serial.to_numpy(), parallel.to_numpy()
    np.testing.assert_array_equal(serial, parallel)

    # EXAMPLE2 shares theta(B) = 1 - 0.5B, so the pi difference is -0.05B(1 - B) / (1 - 0.5B):
    # -0.05 at lag 1 and 0.025 * 0.5^(k-2) at lag k >= 2, with squares summing to 0.0025 + 0.000625 * 4/3 = 1/300
    # (the tail beyond tau=100 is about 1e-60)
    assert serial[1, 0] == serial[2, 1] == pytest.approx(np.sqrt(1 / 300), rel=1e-12)
    assert serial[2, 0] == 0
    np.testing.assert_array_equal(np.diag(serial), 0)
    np.testing.assert_array_equal(serial, serial.T)
    for i in range(4):
        for j in range(i):
            assert serial[i, j] == piccolo(items[i], items[j], tau=100)
