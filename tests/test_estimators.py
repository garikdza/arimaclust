import re

import numpy as np
import pandas as pd
import pmdarima
import pytest
import statsmodels.api as sm
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.metrics import adjusted_mutual_info_score

import arimaclust
from arimaclust.data.generate import generate_arma_sample, generate_clusters
from arimaclust.distance import METRICS, ARMADistance
from arimaclust.models.centroid import ConsistentClustering
from arimaclust.models.neighbours import ThresholdClustering

WELL_SEPARATED = [([0.8], []), ([-0.6], [0.3])]
# fixed white noise for parameters built at collection time
NOISE = np.random.default_rng(0).standard_normal(80)


def test_top_level_exports():
    assert sorted(arimaclust.__all__) == ['ARMADistance', 'ConsistentClustering', 'ThresholdClustering']
    assert arimaclust.ARMADistance is ARMADistance
    assert arimaclust.ConsistentClustering is ConsistentClustering
    assert arimaclust.ThresholdClustering is ThresholdClustering


def _three_series() -> list[np.ndarray]:
    np.random.seed(1)
    return [generate_arma_sample([0.8], [], 120),
            generate_arma_sample([0.8], [], 100),
            generate_arma_sample([-0.5], [], 90)]


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
    assert len(distance.processes_) == 3
    assert all(isinstance(p, sm.tsa.ArmaProcess) for p in distance.processes_)


def test_array_rows_match_list_of_series():
    np.random.seed(2)
    rows = [generate_arma_sample([0.8], [], 120), generate_arma_sample([0.8], [], 120),
            generate_arma_sample([-0.5], [], 120)]
    np.testing.assert_array_equal(ARMADistance().fit_transform(np.vstack(rows)), ARMADistance().fit_transform(rows))


def test_dataframe_columns_are_series_with_trailing_nans_dropped():
    series = _three_series()
    df = pd.DataFrame({name: pd.Series(x) for name, x in zip('abc', series, strict=True)})
    assert df['c'].isna().sum() == 30  # padded to the longest series

    distance = ARMADistance()
    D = distance.fit_transform(df)
    assert list(D.index) == list(D.columns) == ['a', 'b', 'c']
    assert list(distance.processes_) == ['a', 'b', 'c']
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
    return [(len(p.arcoefs), len(p.macoefs)) for p in distance.processes_]


def _two_processes_sample() -> list[np.ndarray]:
    np.random.seed(0)
    return [generate_arma_sample([0.8], [], 300), generate_arma_sample([-0.6], [0.3], 250)]


def test_arima_kwargs_reach_auto_arima():
    X = _two_processes_sample()
    assert _orders(ARMADistance().fit(X)) == [(1, 0), (1, 0)]  # BIC by default
    assert _orders(ARMADistance(arima_kwargs={'information_criterion': 'aic'}).fit(X))[0] == (2, 1)


def test_arima_kwargs_can_override_fit_options():
    distance = ARMADistance(max_p=1, max_q=1, arima_kwargs={'stepwise': False}).fit(_three_series())
    assert distance.dist_matrix_.shape == (3, 3)


def test_arima_kwargs_rejects_constructor_keys():
    with pytest.raises(ValueError, match="'d', 'max_p'"):
        ARMADistance(arima_kwargs={'max_p': 1, 'd': 1, 'stepwise': False}).fit(_three_series())


def test_auto_selected_differencing_order_fits():
    # d=None lets auto_arima choose the order; only metric='mahalanobis' differences the series itself
    assert ARMADistance(d=None).fit(_three_series()).dist_matrix_.shape == (3, 3)


def test_max_orders_below_two_fit():
    distance = ARMADistance(max_p=1, max_q=0).fit(_three_series())
    assert all(p <= 1 and q == 0 for p, q in _orders(distance))


class _FakeArima:
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


def test_consistent_clustering_farthest_point_on_hand_matrix():
    # start at 0; 3 is farthest (11); nearest-centre distances are then (1, 1, 5.2) so 4 comes next
    pos = np.array([0, 1, 10, 11, 5.2])
    D = np.abs(pos[:, None] - pos[None, :])
    model = ConsistentClustering(n_clusters=3, distance='precomputed').fit(D)
    np.testing.assert_array_equal(model.centers_, [0, 3, 4])
    np.testing.assert_array_equal(model.labels_, [0, 0, 1, 1, 2])


def test_threshold_clustering_links_chains_on_precomputed():
    # 0-1 and 1-2 are close, 3 is far: one chain cluster plus a singleton
    d = np.array([[0, 1, 5, 9], [1, 0, 1, 9], [5, 1, 0, 9], [9, 9, 9, 0]], dtype=float)
    labels = ThresholdClustering(alpha=1.5, distance='precomputed').fit_predict(d)
    assert labels[0] == labels[1] == labels[2] != labels[3]


def test_consistent_clustering_recovers_well_separated_clusters():
    np.random.seed(0)
    X, y = generate_clusters(WELL_SEPARATED, t_min=500, samples_per_cluster=5)
    model = ConsistentClustering(n_clusters=2, distance=ARMADistance(max_p=2, max_q=2, d=0, tau=20)).fit(X)
    assert adjusted_mutual_info_score(y, model.labels_) == pytest.approx(1.0)
    assert isinstance(model.distance_, ARMADistance)
    assert model.distance_.dist_matrix_.shape == (10, 10)
    assert len(model.distance_.processes_) == 10
    assert not hasattr(model.distance, 'processes_')


@pytest.mark.slow
def test_consistent_clustering_recovery_improves_with_length():
    lengths, seeds = (20, 50, 200, 1000), range(10)
    distance = ARMADistance(max_p=2, max_q=2, d=0, tau=10, n_jobs=-1)
    means = []
    for length in lengths:
        scores = []
        for seed in seeds:
            np.random.seed(seed)
            X, y = generate_clusters(WELL_SEPARATED, t_min=length, samples_per_cluster=5)
            model = ConsistentClustering(n_clusters=2, distance=distance).fit(X)
            scores.append(adjusted_mutual_info_score(y, model.labels_))
        means.append(np.mean(scores))
    # observed mean AMI: 20 -> 0.367, 50 -> 0.807, 200 -> 1.0, 1000 -> 1.0
    assert np.all(np.diff(means) >= 0), means
    assert means[0] < means[-1], means
    assert means[-1] >= 0.9, means


def test_default_distance_is_fitted_copy():
    model = ConsistentClustering(n_clusters=2).fit(_three_series())
    assert model.distance is None
    assert model.distance_.get_params() == ARMADistance().get_params()
    assert model.distance_.dist_matrix_.shape == (3, 3)
    np.testing.assert_array_equal(model.labels_, [0, 0, 1])


def test_nested_distance_params_are_settable():
    model = ThresholdClustering(alpha=0.5, distance=ARMADistance(max_p=1, max_q=1))
    params = model.get_params(deep=True)
    assert {'distance__tau', 'distance__max_p', 'distance__max_q', 'distance__d', 'distance__n_jobs'} <= set(params)
    model.set_params(distance__tau=5).fit(_three_series())
    assert model.distance_.tau == 5


@pytest.mark.parametrize(('model', 'X', 'match'), [
    pytest.param(ConsistentClustering(distance='precomputed'), np.ones((3, 2)), 'square', id='c'),
    pytest.param(ConsistentClustering(n_clusters=4, distance='precomputed'), np.zeros((3, 3)),
                 'n_clusters=4.*n_samples=3', id='d'),
    pytest.param(ConsistentClustering(n_clusters=0, distance='precomputed'), np.zeros((3, 3)),
                 'n_clusters must be an integer', id='n_clusters=0'),
    pytest.param(ConsistentClustering(distance=ARMADistance(tau=0)), [NOISE], 'tau must be an integer >= 1, got 0',
                 id='tau=0'),
    pytest.param(ThresholdClustering(distance='precomputed'), np.zeros((3, 3)), 'alpha', id='alpha=None'),
    pytest.param(ThresholdClustering(alpha=1.0, distance='euclidean'), np.zeros((3, 3)), 'distance must be',
                 id='distance=euclidean'),
])
def test_bad_clusterer_input_raises(model: ConsistentClustering | ThresholdClustering, X: np.ndarray | list,
                                    match: str):
    with pytest.raises(ValueError, match=match):
        model.fit(X)
