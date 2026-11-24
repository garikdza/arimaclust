import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_mutual_info_score

import arimaclust
import arimaclust.clustering
from arimaclust import ARMADistance, ConsistentClustering, KMedoids, ThresholdClustering
from arimaclust.data import generate_arma_sample, generate_clusters

WELL_SEPARATED = [([0.8], []), ([-0.6], [0.3])]
# fixed white noise for parameters built at collection time
NOISE = np.random.default_rng(0).standard_normal(80)


def _positions_matrix(*pos: float) -> np.ndarray:
    pos = np.array(pos, dtype=float)
    return np.abs(pos[:, None] - pos[None, :])


# two triples at 0, 1, 2 and 10, 11, 12: the farthest-point centres 0 and 5 cost 6, the swaps 0 -> 1 and 5 -> 4 each
# lower it to 5, and the optimum (1, 4) costing 4 needs both
TWO_TRIPLES = _positions_matrix(0, 1, 2, 10, 11, 12)


def _three_series() -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    return [
        generate_arma_sample([0.8], [], 120, random_state=rng),
        generate_arma_sample([0.8], [], 100, random_state=rng),
        generate_arma_sample([-0.5], [], 90, random_state=rng),
    ]


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


def test_kmedoids_is_exported():
    assert arimaclust.KMedoids is arimaclust.clustering.KMedoids
    assert 'KMedoids' in arimaclust.clustering.__all__


def test_kmedoids_default_params():
    assert KMedoids().get_params(deep=False) == {'n_clusters': 2, 'distance': None, 'init': 'build', 'max_iter': 300}


def test_kmedoids_build_finds_hand_optimum():
    # two triples at 0, 1, 2 and 10, 11, 12: the optimum takes each middle point, 1 and 4, costing 4. BUILD starts at 2
    # (column sums 36, 32, 30, 30, ...: the tie 2/3 goes to the lower index), then adds 4 (gain 25 against 24 for 3
    # and 5), costing 5; one swap 2 -> 1 reaches the optimum, and a second iteration finds nothing to improve
    model = KMedoids(n_clusters=2, distance='precomputed').fit(TWO_TRIPLES)
    np.testing.assert_array_equal(model.medoids_, [1, 4])
    np.testing.assert_array_equal(model.labels_, [0, 0, 0, 1, 1, 1])
    assert model.inertia_ == 4.0
    assert model.n_iter_ == 2


def test_kmedoids_farthest_start_improved_by_swap():
    # farthest-point centres 0 and 4 (positions 0 and 11) cost 0 + 1 + 2 + 1 + 0 = 4; swapping 0 -> 1 gives 3, the
    # optimum, while 4 -> 3 leaves the cost at 4 and is not taken; the second iteration finds no improving swap
    D = _positions_matrix(0, 1, 2, 10, 11)
    centers = ConsistentClustering(n_clusters=2, distance='precomputed').fit(D).centers_
    np.testing.assert_array_equal(centers, [0, 4])
    assert D[:, centers].min(axis=1).sum() == 4.0
    model = KMedoids(n_clusters=2, distance='precomputed', init='farthest').fit(D)
    np.testing.assert_array_equal(model.medoids_, [1, 4])
    np.testing.assert_array_equal(model.labels_, [0, 0, 0, 1, 1])
    assert model.inertia_ == 3.0
    assert model.n_iter_ == 2


def test_kmedoids_swap_tie_lowest_incoming_index_and_deterministic():
    fits = []
    for _ in range(2):
        with pytest.warns(ConvergenceWarning):
            fits.append(KMedoids(distance='precomputed', init='farthest', max_iter=1).fit(TWO_TRIPLES))
    # of the two equal swaps the one bringing in sample 1 wins over the one bringing in sample 4
    np.testing.assert_array_equal(fits[0].medoids_, [1, 5])
    np.testing.assert_array_equal(fits[0].medoids_, fits[1].medoids_)
    np.testing.assert_array_equal(fits[0].labels_, fits[1].labels_)


def test_kmedoids_swap_tie_lowest_outgoing_index():
    # farthest-point centres 0, 3, 1 (positions 0, 12, 10) cost 2; bringing in sample 2 (position 11) for either
    # medoid 3 or medoid 1 lowers it to 1, and medoid 1, the lower sample index, goes out although it is listed last
    D = _positions_matrix(0, 10, 11, 12, 11)
    model = KMedoids(n_clusters=3, distance='precomputed', init='farthest').fit(D)
    np.testing.assert_array_equal(model.medoids_, [0, 3, 2])
    np.testing.assert_array_equal(model.labels_, [0, 2, 2, 1, 2])
    assert model.inertia_ == 1.0
    assert model.n_iter_ == 2


def test_kmedoids_assignment_tie_goes_to_medoid_listed_first():
    # farthest-point centres 0, 3, 1 (positions 0, 12, 10) cost 2.5; bringing in sample 2 (position 11) for medoid 1
    # lowers it to 1.5 and is replaced in place, so medoids_ is [0, 3, 2]. Sample 5 (position 11.5) is 0.5 from both
    # medoid 3 and medoid 2, and goes to medoid 3, listed first in medoids_, although medoid 2 has the lower index
    D = _positions_matrix(0, 10, 11, 12, 11, 11.5, 12)
    model = KMedoids(n_clusters=3, distance='precomputed', init='farthest').fit(D)
    np.testing.assert_array_equal(model.medoids_, [0, 3, 2])
    assert model.labels_[5] == 1


def test_kmedoids_max_iter_stops_with_convergence_warning():
    with pytest.warns(ConvergenceWarning, match='max_iter=1'):
        model = KMedoids(distance='precomputed', init='farthest', max_iter=1).fit(TWO_TRIPLES)
    assert model.n_iter_ == 1
    np.testing.assert_array_equal(model.medoids_, [1, 5])
    np.testing.assert_array_equal(model.labels_, [0, 0, 0, 1, 1, 1])
    assert model.inertia_ == 5.0


def test_kmedoids_max_iter_reached_exactly_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model = KMedoids(distance='precomputed', init='farthest', max_iter=2).fit(TWO_TRIPLES)
    assert model.n_iter_ == 2
    np.testing.assert_array_equal(model.medoids_, [1, 4])
    assert model.inertia_ == 4.0
    # below the cap the same two swaps take three iterations, the last one finding nothing to improve
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert KMedoids(distance='precomputed', init='farthest').fit(TWO_TRIPLES).n_iter_ == 3


@pytest.mark.parametrize('init', ['build', 'farthest'])
def test_kmedoids_reaches_local_optimum_no_worse_than_start(init: str):
    rng = np.random.default_rng(0)
    for _ in range(20):
        n, k = int(rng.integers(6, 13)), int(rng.integers(2, 4))
        A = rng.random((n, n))
        D = A + A.T
        np.fill_diagonal(D, 0)
        model = KMedoids(n_clusters=k, distance='precomputed', init=init).fit(D)
        medoids = list(model.medoids_)
        assert model.inertia_ == pytest.approx(D[:, medoids].min(axis=1).sum())
        np.testing.assert_array_equal(model.labels_, D[:, medoids].argmin(axis=1))
        if init == 'farthest':
            centers = ConsistentClustering(n_clusters=k, distance='precomputed').fit(D).centers_
            assert model.inertia_ <= D[:, centers].min(axis=1).sum()
        # no single exchange of a medoid for a non-medoid lowers the cost
        for i in range(len(medoids)):
            for h in set(range(n)) - set(medoids):
                trial = medoids[:i] + [h] + medoids[i + 1 :]
                assert D[:, trial].min(axis=1).sum() >= model.inertia_ - 1e-12


def test_kmedoids_farthest_matches_consistent_clustering_at_local_optimum():
    # farthest-point centres are 0, 3, 4 (see the ConsistentClustering hand test), costing 1 + 1 + 0 = 2; the groups
    # {0, 1}, {10, 11} and {5.2} cannot cost less, and the swaps 0 -> 1 and 3 -> 2 leave the cost at 2, so none is taken
    D = _positions_matrix(0, 1, 10, 11, 5.2)
    consistent = ConsistentClustering(n_clusters=3, distance='precomputed').fit(D)
    model = KMedoids(n_clusters=3, distance='precomputed', init='farthest').fit(D)
    np.testing.assert_array_equal(model.medoids_, consistent.centers_)
    np.testing.assert_array_equal(model.labels_, consistent.labels_)
    assert model.inertia_ == pytest.approx(2.0)
    # one iteration, which finds no improving swap
    assert model.n_iter_ == 1


@pytest.mark.parametrize('init', ['build', 'farthest'])
def test_kmedoids_warns_when_too_few_distinct_groups(init: str):
    # samples 0 and 1 coincide, as do 2 and 3: after medoids 0 and 2 every sample is at distance 0 from one of them
    D = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]], dtype=float)
    with pytest.warns(UserWarning, match='only 2 clusters were found'):
        model = KMedoids(n_clusters=3, distance='precomputed', init=init).fit(D)
    np.testing.assert_array_equal(model.medoids_, [0, 2])
    np.testing.assert_array_equal(model.labels_, [0, 0, 1, 1])
    assert model.inertia_ == 0.0


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


def test_kmedoids_recovers_well_separated_clusters():
    X, y = generate_clusters(WELL_SEPARATED, 500, 5, random_state=0)
    model = KMedoids(n_clusters=2, distance=ARMADistance(max_p=2, max_q=2, d=0, tau=20)).fit(X)
    assert adjusted_mutual_info_score(y, model.labels_) == pytest.approx(1.0)
    assert isinstance(model.distance_, ARMADistance)
    assert model.distance_.dist_matrix_.shape == (10, 10)
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


@pytest.mark.parametrize(
    ('model', 'X', 'match'),
    [
        pytest.param(ConsistentClustering(distance='precomputed'), np.ones((3, 2)), 'square', id='c'),
        pytest.param(
            ConsistentClustering(n_clusters=4, distance='precomputed'),
            np.zeros((3, 3)),
            'n_clusters=4.*n_samples=3',
            id='d',
        ),
        pytest.param(
            ConsistentClustering(n_clusters=0, distance='precomputed'),
            np.zeros((3, 3)),
            'n_clusters must be an integer',
            id='n_clusters=0',
        ),
        pytest.param(
            ConsistentClustering(distance=ARMADistance(tau=0)),
            [NOISE],
            'tau must be an integer >= 1, got 0',
            id='tau=0',
        ),
        pytest.param(
            KMedoids(n_clusters=0, distance='precomputed'),
            np.zeros((3, 3)),
            'n_clusters must be an integer',
            id='kmedoids-n_clusters=0',
        ),
        pytest.param(
            KMedoids(n_clusters=True, distance='precomputed'),
            np.zeros((3, 3)),
            'n_clusters must be an integer',
            id='kmedoids-n_clusters=True',
        ),
        pytest.param(
            KMedoids(n_clusters=4, distance='precomputed'),
            np.zeros((3, 3)),
            'n_clusters=4.*n_samples=3',
            id='kmedoids-n_clusters>n_samples',
        ),
        pytest.param(
            KMedoids(init='random', distance='precomputed'),
            np.zeros((3, 3)),
            "init must be 'build' or 'farthest'",
            id='kmedoids-init=random',
        ),
        pytest.param(
            KMedoids(max_iter=0, distance='precomputed'),
            np.zeros((3, 3)),
            'max_iter must be an integer',
            id='kmedoids-max_iter=0',
        ),
        pytest.param(
            KMedoids(max_iter=1.5, distance='precomputed'),
            np.zeros((3, 3)),
            'max_iter must be an integer',
            id='kmedoids-max_iter=1.5',
        ),
        pytest.param(
            KMedoids(distance='euclidean'), np.zeros((3, 3)), 'distance must be', id='kmedoids-distance=euclidean'
        ),
        pytest.param(KMedoids(distance='precomputed'), np.ones((3, 2)), 'square', id='kmedoids-not-square'),
        pytest.param(ThresholdClustering(distance='precomputed'), np.zeros((3, 3)), 'alpha', id='alpha=None'),
        pytest.param(
            ThresholdClustering(alpha=1.0, distance='euclidean'),
            np.zeros((3, 3)),
            'distance must be',
            id='distance=euclidean',
        ),
        pytest.param(
            ThresholdClustering(alpha=-0.1, distance='precomputed'),
            np.zeros((3, 3)),
            'alpha must be a number',
            id='alpha=-0.1',
        ),
        pytest.param(
            ThresholdClustering(alpha='1', distance='precomputed'),
            np.zeros((3, 3)),
            'alpha must be a number',
            id='alpha=str',
        ),
        pytest.param(
            ThresholdClustering(alpha=float('nan'), distance='precomputed'),
            np.zeros((3, 3)),
            'alpha must be a number',
            id='alpha=nan',
        ),
    ],
)
def test_bad_clusterer_input_raises(
    model: ConsistentClustering | KMedoids | ThresholdClustering, X: np.ndarray | list, match: str
):
    with pytest.raises(ValueError, match=match):
        model.fit(X)
