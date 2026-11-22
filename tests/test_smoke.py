import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_mutual_info_score

from arimaclust.autoarima.estimation import estimate_processes, to_arma_process
from arimaclust.data.examples import EXAMPLE2
from arimaclust.data.generate import generate_clusters
from arimaclust.metrics.autoregressive import piccolo
from arimaclust.metrics.matrix import dist_matrix_parallel
from arimaclust.models.centroid import ConsistentARMAClustering
from arimaclust.models.neighbours import NearestNeighbourModel
from arimaclust.tracking.mapping import mapp_hungarian

PROCESS_ARGS = {'max_p': 2, 'max_q': 2, 'd': 0, 'n_workers': 1}
WELL_SEPARATED = [([0.8], []), ([-0.6], [0.3])]


def test_to_arma_process_accepts_lists():
    process = to_arma_process([0.5, -0.2], [0.3])
    np.testing.assert_allclose(process.arcoefs, [0.5, -0.2])
    np.testing.assert_allclose(process.macoefs, [0.3])


def test_piccolo_matches_known_value():
    p1, p2 = (to_arma_process(*params) for params in EXAMPLE2)
    assert piccolo(p1, p1) == 0
    assert piccolo(p1, p2, tau=100) == pytest.approx(0.0577350269, rel=1e-6)


def test_dataframe_input_is_column_wise():
    np.random.seed(0)
    X, _ = generate_clusters(WELL_SEPARATED, t_min=300, samples_per_cluster=1)
    df = pd.DataFrame({'a': X[0], 'b': X[1]})
    processes = estimate_processes(df, **PROCESS_ARGS)
    assert list(processes) == ['a', 'b']
    dm = dist_matrix_parallel(processes, metric_name='piccolo', tau=20)
    assert list(dm.index) == ['a', 'b'] and dm.loc['a', 'b'] > 0


def test_consistent_clustering_recovers_well_separated_clusters():
    np.random.seed(0)
    X, y = generate_clusters(WELL_SEPARATED, t_min=500, samples_per_cluster=5)
    model = ConsistentARMAClustering(k=2, process_args=PROCESS_ARGS, metric_args={'tau': 20}).fit(X)
    assert adjusted_mutual_info_score(y, model.labels_) == pytest.approx(1.0)


def test_nearest_neighbour_is_transitive_on_precomputed():
    # 0-1 and 1-2 are close, 3 is far: one chain cluster plus a singleton
    d = np.array([[0, 1, 5, 9], [1, 0, 1, 9], [5, 1, 0, 9], [9, 9, 9, 0]], dtype=float)
    labels = NearestNeighbourModel(alpha=1.5, metric='precomputed').fit_predict(d)
    assert labels[0] == labels[1] == labels[2] != labels[3]


def test_hungarian_mapping_handles_permuted_labels():
    rows, cols = mapp_hungarian(np.array([0, 0, 1, 1, 2]), np.array([2, 2, 0, 0, 1]))
    assert dict(zip(rows, cols, strict=False)) == {0: 2, 1: 0, 2: 1}
