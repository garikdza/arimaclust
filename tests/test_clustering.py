import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score

from arimaclust import ARMADistance, ConsistentClustering, ThresholdClustering
from arimaclust.data import generate_arma_sample, generate_clusters

WELL_SEPARATED = [([0.8], []), ([-0.6], [0.3])]
# fixed white noise for parameters built at collection time
NOISE = np.random.default_rng(0).standard_normal(80)


def _three_series() -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    return [generate_arma_sample([0.8], [], 120, random_state=rng),
            generate_arma_sample([0.8], [], 100, random_state=rng),
            generate_arma_sample([-0.5], [], 90, random_state=rng)]


def test_consistent_clustering_farthest_point_on_hand_matrix():
    # start at 0; 3 is farthest (11); nearest-centre distances are then (1, 1, 5.2) so 4 comes next
    pos = np.array([0, 1, 10, 11, 5.2])
    D = np.abs(pos[:, None] - pos[None, :])
    model = ConsistentClustering(n_clusters=3, distance='precomputed').fit(D)
    np.testing.assert_array_equal(model.centers_, [0, 3, 4])
    np.testing.assert_array_equal(model.labels_, [0, 0, 1, 1, 2])


def test_consistent_clustering_warns_when_too_few_distinct_groups():
    # samples 0 and 1 coincide, as do 2 and 3: two groups, three clusters asked
    D = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]], dtype=float)
    with pytest.warns(UserWarning, match='only 2 clusters were found'):
        model = ConsistentClustering(n_clusters=3, distance='precomputed').fit(D)
    np.testing.assert_array_equal(model.centers_, [0, 2])
    np.testing.assert_array_equal(model.labels_, [0, 0, 1, 1])
    np.testing.assert_array_equal(ConsistentClustering(n_clusters=2, distance='precomputed').fit(D).labels_,
                                  [0, 0, 1, 1])


def test_threshold_clustering_links_chains_on_precomputed():
    # 0-1 and 1-2 are close, 3 is far: one chain cluster plus a singleton
    d = np.array([[0, 1, 5, 9], [1, 0, 1, 9], [5, 1, 0, 9], [9, 9, 9, 0]], dtype=float)
    labels = ThresholdClustering(alpha=1.5, distance='precomputed').fit_predict(d)
    assert labels[0] == labels[1] == labels[2] != labels[3]


def test_consistent_clustering_recovers_well_separated_clusters():
    X, y = generate_clusters(WELL_SEPARATED, 500, 5, random_state=0)
    model = ConsistentClustering(n_clusters=2, distance=ARMADistance(max_p=2, max_q=2, d=0, tau=20)).fit(X)
    assert adjusted_mutual_info_score(y, model.labels_) == pytest.approx(1.0)
    assert isinstance(model.distance_, ARMADistance)
    assert model.distance_.dist_matrix_.shape == (10, 10)
    assert len(model.distance_.models_) == 10
    assert not hasattr(model.distance, 'models_')


@pytest.mark.slow
def test_consistent_clustering_recovery_improves_with_length():
    lengths, seeds = (20, 50, 200, 1000), range(10)
    distance = ARMADistance(max_p=2, max_q=2, d=0, tau=10, n_jobs=-1)
    means = []
    for length in lengths:
        scores = []
        for seed in seeds:
            X, y = generate_clusters(WELL_SEPARATED, length, 5, random_state=seed)
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
    pytest.param(ThresholdClustering(alpha=-0.1, distance='precomputed'), np.zeros((3, 3)), 'alpha must be a number',
                 id='alpha=-0.1'),
    pytest.param(ThresholdClustering(alpha='1', distance='precomputed'), np.zeros((3, 3)), 'alpha must be a number',
                 id='alpha=str'),
    pytest.param(ThresholdClustering(alpha=float('nan'), distance='precomputed'), np.zeros((3, 3)),
                 'alpha must be a number', id='alpha=nan'),
])
def test_bad_clusterer_input_raises(model: ConsistentClustering | ThresholdClustering, X: np.ndarray | list,
                                    match: str):
    with pytest.raises(ValueError, match=match):
        model.fit(X)
