import numpy as np
import pytest
from scipy import stats

from arimaclust.data import generate_arma_sample, generate_clusters
from synthetic import EXAMPLE1


def test_generate_clusters_per_cluster_counts():
    X, y = generate_clusters(EXAMPLE1, 30, [2, 3], random_state=0)
    np.testing.assert_array_equal(y, [0, 0, 1, 1, 1])
    assert [len(x) for x in X] == [30] * 5


def test_generate_clusters_same_count_for_every_cluster():
    X, y = generate_clusters(EXAMPLE1, 30, 2, random_state=0)
    np.testing.assert_array_equal(y, [0, 0, 1, 1])
    assert len(X) == 4


@pytest.mark.parametrize(
    ('n_per_cluster', 'error', 'match'),
    [
        pytest.param('3', TypeError, 'int or a sequence', id='string'),
        pytest.param(2.0, TypeError, 'int or a sequence', id='float'),
        pytest.param([2], ValueError, '1 counts for 2 clusters', id='wrong-length'),
    ],
)
def test_generate_clusters_rejects_bad_counts(n_per_cluster: object, error: type[Exception], match: str):
    with pytest.raises(error, match=match):
        generate_clusters(EXAMPLE1, 30, n_per_cluster)


def test_same_seed_gives_same_series_and_draws_differ():
    X, _ = generate_clusters(EXAMPLE1, 50, 2, random_state=3)
    again, _ = generate_clusters(EXAMPLE1, 50, 2, random_state=3)
    for a, b in zip(X, again, strict=True):
        np.testing.assert_array_equal(a, b)
    # one generator feeds every series, so no two series repeat the same noise
    assert len({x.tobytes() for x in X}) == len(X)


def test_generator_random_state_continues_the_stream():
    rng = np.random.default_rng(5)
    first = generate_arma_sample([0.5], [], 40, random_state=rng)
    second = generate_arma_sample([0.5], [], 40, random_state=rng)
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(first, generate_arma_sample([0.5], [], 40, random_state=5))


def test_burnin_starts_series_in_the_stationary_distribution():
    # an AR(1) with phi = 0.95 has stationary variance 1 / (1 - 0.95^2), about 10.3; without a burn-in
    # every series would start at 0, with variance 1 at t=0
    phi, n = 0.95, 2000
    rng = np.random.default_rng(0)
    first = np.array([generate_arma_sample([phi], [], 1, burnin=500, random_state=rng)[0] for _ in range(n)])
    no_burnin = np.array([generate_arma_sample([phi], [], 1, burnin=0, random_state=rng)[0] for _ in range(n)])
    stationary_var = 1 / (1 - phi**2)
    assert stats.kstest(first / np.sqrt(stationary_var), 'norm').pvalue > 0.01
    assert no_burnin.var() == pytest.approx(1, rel=0.15)
