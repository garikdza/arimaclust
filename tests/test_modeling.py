import re
from types import SimpleNamespace

import numpy as np
import pmdarima
import pytest

from arimaclust.data import generate_arma_sample
from arimaclust.modeling import FittedARIMA, fit_arima, fit_arimas


def test_order_and_polynomials_follow_the_sign_conventions():
    model = FittedARIMA([0.5, -0.2], [0.3])
    assert model.order == (2, 0, 1)
    # AR params enter the lag polynomial negated, 1 - 0.5B + 0.2B^2; MA params keep their sign, 1 + 0.3B
    np.testing.assert_allclose(model.ar_polynomial, [1, -0.5, 0.2])
    np.testing.assert_allclose(model.arma_process.ma, [1, 0.3])
    np.testing.assert_allclose(model.arma_process.arcoefs, [0.5, -0.2])


def test_differencing_is_kept_in_the_ar_polynomial():
    # (1 - 0.5B)(1 - B)^2 = 1 - 2.5B + 2B^2 - 0.5B^3; the ARMA part keeps only 1 - 0.5B
    model = FittedARIMA([0.5], [0.3], d=2)
    assert model.order == (1, 2, 1)
    np.testing.assert_allclose(model.ar_polynomial, [1, -2.5, 2, -0.5])
    np.testing.assert_allclose(model.arma_process.ar, [1, -0.5])


def test_pi_matches_hand_values():
    # pi(B) = (1 - 0.5B) / (1 + 0.3B), so pi_k = (phi + theta)(-theta)^(k-1) = 0.8, -0.24, 0.072
    np.testing.assert_allclose(FittedARIMA([0.5], [0.3]).pi(3), [0.8, -0.24, 0.072], rtol=1e-12)


def test_pi_of_differenced_model_includes_the_unit_root():
    # (1 - 0.5B)(1 - B) = 1 - 1.5B + 0.5B^2: pi = (1.5, -0.5, 0, ...); a random walk has pi = (1, 0, ...)
    np.testing.assert_allclose(FittedARIMA([0.5], [], d=1).pi(4), [1.5, -0.5, 0, 0], atol=1e-12)
    np.testing.assert_allclose(FittedARIMA([], [], d=1).pi(3), [1, 0, 0], atol=1e-12)


def test_pi_rejects_a_non_invertible_model():
    # MA root 1/2.0 lies inside the unit circle
    model = FittedARIMA([0.5], [2.0])
    assert not model.isinvertible
    with pytest.raises(ValueError, match='not invertible'):
        model.pi(5)


@pytest.mark.parametrize('tau', [0, -1, 2.5, True, '3', None], ids=repr)
def test_pi_rejects_bad_tau(tau: object):
    with pytest.raises(ValueError, match=rf'tau must be an integer >= 1, got {re.escape(repr(tau))}'):
        FittedARIMA([0.5], [0.3]).pi(tau)


def test_numpy_integer_tau_matches_int():
    model = FittedARIMA([0.5], [0.3])
    np.testing.assert_array_equal(model.pi(np.int64(3)), model.pi(3))


@pytest.mark.parametrize('d', [-1, 1.5, True, None], ids=repr)
def test_rejects_bad_differencing_order(d: object):
    with pytest.raises(ValueError, match='d must be an integer >= 0'):
        FittedARIMA([0.5], [], d=d)


def test_differenced_gives_the_arma_part_with_the_differenced_series():
    series = np.cumsum(np.arange(6.0))
    arma = FittedARIMA([0.5], [0.3], d=1, series=series).differenced()
    assert arma.order == (1, 0, 1)
    np.testing.assert_array_equal(arma.series, np.diff(series))
    assert FittedARIMA([0.5], [], d=1).differenced().series is None


def test_fit_arima_keeps_order_parameters_and_series():
    x = generate_arma_sample([0.5], [0.3], 300, random_state=0)
    model = fit_arima(x, max_p=2, max_q=2, d=0)
    fitted = pmdarima.auto_arima(x, max_p=2, max_q=2, d=0, information_criterion='bic', error_action='ignore',
                                 stepwise=True, suppress_warnings=True)
    assert model.order == fitted.order
    np.testing.assert_allclose(model.ar, fitted.arparams())
    np.testing.assert_allclose(model.ma, fitted.maparams())
    np.testing.assert_array_equal(model.series, x)


def test_fit_arimas_selects_differencing_per_series():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.standard_normal(400))
    noise = generate_arma_sample([], [], 400, random_state=rng)
    walk_model, noise_model = fit_arimas([walk, noise], d=None, max_p=2, max_q=2)
    assert (walk_model.d, noise_model.d) == (1, 0)


def test_failed_fit_names_series(monkeypatch: pytest.MonkeyPatch):
    def fail_on_short(x: np.ndarray, **kwargs) -> SimpleNamespace:
        if len(x) == 20:
            raise ValueError('Could not successfully fit a viable ARIMA model')
        return SimpleNamespace(arparams=lambda: np.array([]), maparams=lambda: np.array([]), order=(0, 0, 0))

    monkeypatch.setattr(pmdarima, 'auto_arima', fail_on_short)
    with pytest.raises(ValueError, match=r"could not fit series 'b': Could not successfully fit"):
        fit_arimas([np.zeros(30), np.zeros(20)], names=['a', 'b'])


def test_repr_shows_parameters_not_the_series():
    assert repr(FittedARIMA([0.5], [2.0], d=1, series=np.zeros(3))) == 'FittedARIMA(ar=[0.5], ma=[2.0], d=1)'
