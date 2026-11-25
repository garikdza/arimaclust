import numpy as np
import pytest

from arimaclust import ThresholdClustering


def test_threshold_clustering_links_chains_on_precomputed():
    # 0-1 and 1-2 are close, 3 is far: one chain cluster plus a singleton
    d = np.array([[0, 1, 5, 9], [1, 0, 1, 9], [5, 1, 0, 9], [9, 9, 9, 0]], dtype=float)
    labels = ThresholdClustering(alpha=1.5, distance='precomputed').fit_predict(d)
    assert labels[0] == labels[1] == labels[2] != labels[3]


@pytest.mark.parametrize(
    ('alpha', 'match'),
    [
        pytest.param(None, 'alpha must be set', id='alpha=None'),
        pytest.param(-0.1, 'alpha must be a number', id='alpha=-0.1'),
        pytest.param('1', 'alpha must be a number', id='alpha=str'),
        pytest.param(float('nan'), 'alpha must be a number', id='alpha=nan'),
    ],
)
def test_bad_alpha_raises(alpha: object, match: str):
    with pytest.raises(ValueError, match=match):
        ThresholdClustering(alpha=alpha, distance='precomputed').fit(np.zeros((3, 3)))
