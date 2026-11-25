import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_mutual_info_score

from arimaclust import ARMADistance, ConsistentClustering, KMedoids
from arimaclust.data import generate_clusters

WELL_SEPARATED = [([0.8], []), ([-0.6], [0.3])]


def _positions_matrix(*pos: float) -> np.ndarray:
    pos = np.array(pos, dtype=float)
    return np.abs(pos[:, None] - pos[None, :])


# two triples at 0, 1, 2 and 10, 11, 12: the farthest-point centres 0 and 5 cost 6, the swaps 0 -> 1 and 5 -> 4 each
# lower it to 5, and the optimum (1, 4) costing 4 needs both
TWO_TRIPLES = _positions_matrix(0, 1, 2, 10, 11, 12)


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


def test_kmedoids_recovers_well_separated_clusters():
    X, y = generate_clusters(WELL_SEPARATED, 500, 5, random_state=0)
    model = KMedoids(n_clusters=2, distance=ARMADistance(max_p=2, max_q=2, d=0, tau=20)).fit(X)
    assert adjusted_mutual_info_score(y, model.labels_) == pytest.approx(1.0)
    assert isinstance(model.distance_, ARMADistance)
    assert model.distance_.dist_matrix_.shape == (10, 10)
    assert not hasattr(model.distance, 'models_')


@pytest.mark.parametrize(
    ('params', 'match'),
    [
        pytest.param({'n_clusters': 0}, 'n_clusters must be an integer', id='n_clusters=0'),
        pytest.param({'n_clusters': True}, 'n_clusters must be an integer', id='n_clusters=True'),
        pytest.param({'n_clusters': 4}, 'n_clusters=4.*n_samples=3', id='n_clusters>n_samples'),
        pytest.param({'init': 'random'}, "init must be 'build' or 'farthest'", id='init=random'),
        pytest.param({'max_iter': 0}, 'max_iter must be an integer', id='max_iter=0'),
        pytest.param({'max_iter': 1.5}, 'max_iter must be an integer', id='max_iter=1.5'),
    ],
)
def test_bad_params_raise(params: dict, match: str):
    with pytest.raises(ValueError, match=match):
        KMedoids(distance='precomputed', **params).fit(np.zeros((3, 3)))
