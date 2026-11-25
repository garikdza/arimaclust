from collections.abc import Callable

import numpy as np
import pytest

from arimaclust import ARMADistance, ConsistentClustering, KMedoids, ThresholdClustering


def test_default_distance_is_fitted_copy(three_series: list[np.ndarray]):
    model = ConsistentClustering(n_clusters=2).fit(three_series)
    assert model.distance is None
    assert model.distance_.get_params() == ARMADistance().get_params()
    assert model.distance_.dist_matrix_.shape == (3, 3)
    np.testing.assert_array_equal(model.labels_, [0, 0, 1])


def test_nested_distance_params_are_settable(three_series: list[np.ndarray]):
    model = ThresholdClustering(alpha=0.5, distance=ARMADistance(max_p=1, max_q=1))
    params = model.get_params(deep=True)
    assert {'distance__tau', 'distance__max_p', 'distance__max_q', 'distance__d', 'distance__n_jobs'} <= set(params)
    model.set_params(distance__tau=5).fit(three_series)
    assert model.distance_.tau == 5


CLUSTERERS = [
    pytest.param(lambda distance: ConsistentClustering(distance=distance), id='consistent'),
    pytest.param(lambda distance: KMedoids(distance=distance), id='kmedoids'),
    pytest.param(lambda distance: ThresholdClustering(alpha=1.0, distance=distance), id='threshold'),
]


@pytest.mark.parametrize('make', CLUSTERERS)
def test_precomputed_needs_a_square_matrix(make: Callable):
    with pytest.raises(ValueError, match='square'):
        make('precomputed').fit(np.ones((3, 2)))


@pytest.mark.parametrize('make', CLUSTERERS)
def test_unknown_distance_raises(make: Callable):
    with pytest.raises(ValueError, match='distance must be'):
        make('euclidean').fit(np.zeros((3, 3)))
