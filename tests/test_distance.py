import re

import numpy as np
import pandas as pd
import pmdarima
import pytest
from numpy.typing import ArrayLike
from sklearn.base import clone

import arimaclust
from arimaclust import ARMADistance, ConsistentClustering, ThresholdClustering
from arimaclust.data import generate_arma_sample, generate_clusters
from arimaclust.metrics import METRICS, mahalanobis
from arimaclust.modeling import FittedARIMA

# fixed white noise for parameters built at collection time
NOISE = np.random.default_rng(0).standard_normal(80)


def test_top_level_exports():
    assert sorted(arimaclust.__all__) == ['ARMADistance', 'ConsistentClustering', 'ThresholdClustering']
    assert arimaclust.ARMADistance is ARMADistance
    assert arimaclust.ConsistentClustering is ConsistentClustering
    assert arimaclust.ThresholdClustering is ThresholdClustering


def _three_series() -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    return [generate_arma_sample([0.8], [], 120, random_state=rng),
            generate_arma_sample([0.8], [], 100, random_state=rng),
            generate_arma_sample([-0.5], [], 90, random_state=rng)]


def test_arma_distance_default_params_and_clone():
    assert ARMADistance().get_params() == {'max_p': 3, 'max_q': 3, 'd': 0, 'metric': 'piccolo', 'tau': 50,
                                           'arima_kwargs': None, 'n_jobs': None}
    cloned = clone(ARMADistance(arima_kwargs={'stepwise': False}))
    assert cloned.get_params()['arima_kwargs'] == {'stepwise': False}


def test_ragged_list_gives_symmetric_matrix_in_input_order():
    distance = ARMADistance()
    D = distance.fit_transform(_three_series())
    assert D.shape == (3, 3)
    np.testing.assert_array_equal(D, D.T)
    np.testing.assert_array_equal(np.diag(D), 0)
    assert D[0, 1] < D[0, 2]  # the two AR(0.8) series are the close pair
    assert len(distance.models_) == 3
    assert all(isinstance(m, FittedARIMA) for m in distance.models_)
    for model, x in zip(distance.models_, _three_series(), strict=True):
        np.testing.assert_array_equal(model.series, x)


def test_array_rows_match_list_of_series():
    rows, _ = generate_clusters([([0.8], []), ([0.8], []), ([-0.5], [])], 120, 1, random_state=2)
    np.testing.assert_array_equal(ARMADistance().fit_transform(np.vstack(rows)), ARMADistance().fit_transform(rows))


def test_dataframe_columns_are_series_with_trailing_nans_dropped():
    series = _three_series()
    df = pd.DataFrame({name: pd.Series(x) for name, x in zip('abc', series, strict=True)})
    assert df['c'].isna().sum() == 30  # padded to the longest series

    distance = ARMADistance()
    D = distance.fit_transform(df)
    assert list(D.index) == list(D.columns) == ['a', 'b', 'c']
    assert list(distance.models_) == ['a', 'b', 'c']
    np.testing.assert_array_equal(D.to_numpy(), ARMADistance().fit_transform(series))


@pytest.mark.parametrize('metric', METRICS)
def test_metric_gives_symmetric_matrix_with_close_pair(metric: str):
    D = ARMADistance(metric=metric).fit_transform(_three_series())
    assert D.shape == (3, 3)
    np.testing.assert_array_equal(D, D.T)
    np.testing.assert_array_equal(np.diag(D), 0)
    assert D[0, 1] < D[0, 2]  # the two AR(0.8) series are the close pair


def test_mahalanobis_dataframe_keeps_labels_and_pairs_series():
    series = _three_series()
    df = pd.DataFrame({name: pd.Series(x) for name, x in zip('abc', series, strict=True)})
    D = ARMADistance(metric='mahalanobis').fit_transform(df)
    assert list(D.index) == list(D.columns) == ['a', 'b', 'c']
    np.testing.assert_array_equal(D.to_numpy(), ARMADistance(metric='mahalanobis').fit_transform(series))


@pytest.mark.parametrize('metric', ['nonexistent', 'arma2ar'])
def test_unknown_metric_raises_with_valid_names(metric: str):
    with pytest.raises(ValueError, match=f"'piccolo', 'cepstral', 'mahalanobis'.*got '{metric}'"):
        ARMADistance(metric=metric).fit(_three_series())


def test_mahalanobis_without_differencing_order_raises():
    with pytest.raises(ValueError, match="metric='mahalanobis' needs an explicit differencing order d"):
        ARMADistance(metric='mahalanobis', d=None).fit(_three_series())


@pytest.mark.parametrize(('X', 'match'), [
    pytest.param([NOISE, NOISE[:50]], 'Series 1 has 50 observations .*tau=50', id='list'),
    pytest.param(pd.DataFrame({'a': pd.Series(NOISE), 'b': pd.Series(NOISE[:50])}),
                 "Series 'b' has 50 observations", id='dataframe'),
    pytest.param([NOISE, np.ones(80)], 'Series 1 is constant', id='constant'),
])
def test_mahalanobis_short_or_constant_series_names_series(X: list | pd.DataFrame, match: str):
    with pytest.raises(ValueError, match=match):
        ARMADistance(metric='mahalanobis', tau=50).fit(X)


def test_mahalanobis_checks_differenced_series():
    # a linear trend is long enough but constant once differenced
    with pytest.raises(ValueError, match='Series 1 is constant after differencing'):
        ARMADistance(metric='mahalanobis', d=1, tau=10).fit([np.random.randn(80), np.arange(80.0)])


@pytest.mark.parametrize('tau', [0, -1, 2.5, True, '3', None], ids=repr)
@pytest.mark.parametrize('metric', METRICS)
def test_bad_tau_raises_before_series_checks(metric: str, tau: object):
    # a single series would itself fail validation, so the tau error shows tau is checked first
    with pytest.raises(ValueError, match=rf'tau must be an integer >= 1, got {re.escape(repr(tau))}'):
        ARMADistance(metric=metric, tau=tau).fit([NOISE])


@pytest.mark.parametrize('metric', METRICS)
def test_parallel_fit_matches_serial(metric: str):
    X = _three_series()
    np.testing.assert_array_equal(ARMADistance(metric=metric, n_jobs=2).fit_transform(X),
                                  ARMADistance(metric=metric).fit_transform(X))


def _orders(distance: ARMADistance) -> list[tuple[int, int]]:
    return [(p, q) for p, _, q in (model.order for model in distance.models_)]


def _two_processes_sample() -> list[np.ndarray]:
    # a draw on which AIC picks a larger model than BIC for the first series
    rng = np.random.default_rng(5)
    return [generate_arma_sample([0.8], [], 300, random_state=rng),
            generate_arma_sample([-0.6], [0.3], 250, random_state=rng)]


def test_arima_kwargs_reach_auto_arima():
    X = _two_processes_sample()
    assert _orders(ARMADistance().fit(X)) == [(1, 0), (1, 0)]  # BIC by default
    assert _orders(ARMADistance(arima_kwargs={'information_criterion': 'aic'}).fit(X))[0] == (2, 3)


def test_arima_kwargs_can_override_fit_options():
    distance = ARMADistance(max_p=1, max_q=1, arima_kwargs={'stepwise': False}).fit(_three_series())
    assert distance.dist_matrix_.shape == (3, 3)


def test_arima_kwargs_rejects_constructor_keys():
    with pytest.raises(ValueError, match="'d', 'max_p'"):
        ARMADistance(arima_kwargs={'max_p': 1, 'd': 1, 'stepwise': False}).fit(_three_series())


def test_auto_selected_differencing_order_fits():
    # d=None lets auto_arima choose the order; only metric='mahalanobis' differences the series itself
    assert ARMADistance(d=None).fit(_three_series()).dist_matrix_.shape == (3, 3)


@pytest.mark.parametrize('metric', ['piccolo', 'cepstral'])
def test_differencing_order_separates_random_walk_from_noise(metric: str):
    # both fit as ARMA(0,0), once differenced and once not: the (1 - B) factor kept in the AR polynomial
    # gives the walk pi = (1, 0, 0, ...), so the two are 1 apart instead of 0 (cepstral: c_n = 1/n)
    rng = np.random.default_rng(0)
    noise, walk = rng.standard_normal(500), np.cumsum(rng.standard_normal(500))
    distance = ARMADistance(d=None, metric=metric).fit([noise, walk])
    assert [model.d for model in distance.models_] == [0, 1]
    expected = 1.0 if metric == 'piccolo' else np.sqrt(sum(1 / n ** 2 for n in range(1, 51)))
    assert distance.dist_matrix_[0, 1] == pytest.approx(expected)


def test_fixed_differencing_order_is_kept_in_models():
    X = [np.cumsum(x) for x in _three_series()]
    distance = ARMADistance(d=1).fit(X)
    # a (1 - B) factor puts a root of the AR polynomial at B = 1, so its coefficients sum to 0
    for model in distance.models_:
        assert model.d == 1
        assert model.ar_polynomial.sum() == pytest.approx(0, abs=1e-12)


def test_mahalanobis_with_differencing_matches_the_metric_on_the_models():
    X = [np.cumsum(x) for x in _three_series()]
    distance = ARMADistance(metric='mahalanobis', d=1, tau=10).fit(X)
    assert all(model.d == 1 for model in distance.models_)
    a, b, c = distance.models_
    np.testing.assert_allclose(distance.dist_matrix_[0, 1], mahalanobis(a, b, tau=10), rtol=1e-12)
    np.testing.assert_allclose(distance.dist_matrix_[0, 2], mahalanobis(a, c, tau=10), rtol=1e-12)


def test_max_orders_below_two_fit():
    distance = ARMADistance(max_p=1, max_q=0).fit(_three_series())
    assert all(p <= 1 and q == 0 for p, q in _orders(distance))


class _FakeArima:
    order = (1, 0, 1)

    def __init__(self, ma: float) -> None:
        self.ma = ma

    def arparams(self) -> np.ndarray:
        return np.array([0.5])

    def maparams(self) -> np.ndarray:
        return np.array([self.ma])


@pytest.mark.parametrize(('X', 'label'), [
    pytest.param([np.zeros(50), np.zeros(60)], '1', id='list'),
    pytest.param(pd.DataFrame({'a': pd.Series(np.zeros(50)), 'b': pd.Series(np.zeros(60))}), "'b'", id='dataframe'),
])
def test_non_invertible_fit_names_series(monkeypatch: pytest.MonkeyPatch, X: list | pd.DataFrame, label: str):
    # an MA coefficient of 2 puts the MA root inside the unit circle
    monkeypatch.setattr(pmdarima, 'auto_arima', lambda x, **kwargs: _FakeArima(ma=2.0 if len(x) == 60 else 0.3))
    with pytest.raises(ValueError, match=rf'Series {label} has a non-invertible .*0\.5.*2\.0'):
        ARMADistance().fit_transform(X)


@pytest.mark.parametrize(('X', 'match'), [
    pytest.param([np.zeros(50)], 'at least 2 series', id='a'),
    pytest.param(np.array([np.zeros(50), np.r_[np.zeros(49), np.nan]]), 'Series 1 contains NaN', id='b-array'),
    pytest.param([np.zeros(50), np.r_[np.zeros(20), np.nan, np.zeros(29)]], 'Series 1 contains NaN', id='b-list'),
    pytest.param([np.zeros(50), np.zeros((5, 10))], 'must be 1-D', id='e'),
    pytest.param([np.zeros(50), np.array([])], 'Series 1 is empty', id='empty-list'),
    pytest.param(pd.DataFrame({'a': np.zeros(50), 'b': np.nan}), "Series 'b' is empty", id='empty-dataframe'),
])
def test_bad_series_input_raises(X: ArrayLike, match: str):
    with pytest.raises(ValueError, match=match):
        ARMADistance().fit(X)
