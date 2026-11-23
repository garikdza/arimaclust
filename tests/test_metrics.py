import numpy as np
import pandas as pd
import pytest

from arimaclust.autoarima.estimation import to_arma_process
from arimaclust.data.examples import EXAMPLE1, EXAMPLE2
from arimaclust.metrics.autoregressive import piccolo
from arimaclust.metrics.matrix import dist_matrix_parallel


@pytest.mark.parametrize('as_dict', [pytest.param(False, id='list'), pytest.param(True, id='dict')])
def test_dist_matrix_entries_and_parallel_equality(as_dict: bool):
    p1, p2 = (to_arma_process(*params) for params in EXAMPLE2)
    items = [p1, p2, p1, to_arma_process(*EXAMPLE1[1])]
    data = dict(zip('abcd', items, strict=True)) if as_dict else items

    serial = dist_matrix_parallel(data, metric_name='piccolo', tau=100)
    parallel = dist_matrix_parallel(data, metric_name='piccolo', n_jobs=2, tau=100)
    if as_dict:
        for D in (serial, parallel):
            assert isinstance(D, pd.DataFrame)
            assert list(D.index) == list(D.columns) == list('abcd')
        serial, parallel = serial.to_numpy(), parallel.to_numpy()
    np.testing.assert_array_equal(serial, parallel)

    assert serial[1, 0] == serial[2, 1] == pytest.approx(0.0577350269, rel=1e-6)
    assert serial[2, 0] == 0
    np.testing.assert_array_equal(np.diag(serial), 0)
    np.testing.assert_array_equal(serial, serial.T)
    for i in range(4):
        for j in range(i):
            assert serial[i, j] == piccolo(items[i], items[j], tau=100)
