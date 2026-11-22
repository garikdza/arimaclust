import numpy as np
import pandas as pd
import pmdarima
import pytest
import statsmodels.api as sm
from numpy.typing import ArrayLike
from sklearn.base import clone

from arimaclust.data.generate import generate_arma_sample
from arimaclust.distance import ARMADistance


def _three_series() -> list[np.ndarray]:
    np.random.seed(1)
    return [generate_arma_sample([0.8], [], 120),
            generate_arma_sample([0.8], [], 100),
            generate_arma_sample([-0.5], [], 90)]


def test_arma_distance_default_params_and_clone():
    assert ARMADistance().get_params() == {'max_p': 3, 'max_q': 3, 'd': 0, 'tau': 10, 'arima_kwargs': None,
                                           'n_jobs': None}
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
