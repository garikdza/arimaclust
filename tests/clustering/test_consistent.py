import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score

from arimaclust import ARMADistance, ConsistentClustering
from arimaclust.data import generate_clusters

WELL_SEPARATED = [([0.8], []), ([-0.6], [0.3])]


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
    np.testing.assert_array_equal(
        ConsistentClustering(n_clusters=2, distance='precomputed').fit(D).labels_, [0, 0, 1, 1]
    )


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


@pytest.mark.parametrize(
    ('n_clusters', 'match'),
    [
        pytest.param(4, 'n_clusters=4.*n_samples=3', id='n_clusters>n_samples'),
        pytest.param(0, 'n_clusters must be an integer', id='n_clusters=0'),
    ],
)
def test_bad_params_raise(n_clusters: int, match: str):
    with pytest.raises(ValueError, match=match):
        ConsistentClustering(n_clusters=n_clusters, distance='precomputed').fit(np.zeros((3, 3)))
