import numpy as np
import pytest

from arimaclust.data.generate import generate_clusters
from synthetic import EXAMPLE1


def test_generate_clusters_per_cluster_counts():
    np.random.seed(0)
    X, y = generate_clusters(EXAMPLE1, t_min=30, samples_per_cluster=[2, 3])
    assert y == [0, 0, 1, 1, 1]
    assert [len(x) for x in X] == [30] * 5


def test_generate_clusters_rejects_non_int_count():
    # no match: generate_clusters raises a bare TypeError with no message
    with pytest.raises(TypeError):
        generate_clusters(EXAMPLE1, t_min=30, samples_per_cluster='3')
