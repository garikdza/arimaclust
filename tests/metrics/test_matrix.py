import numpy as np
import pandas as pd
import pytest

from arimaclust.metrics import dist_matrix_parallel, piccolo
from arimaclust.modeling import ARIMA
from synthetic import EXAMPLE1, EXAMPLE2


@pytest.mark.parametrize('as_dict', [pytest.param(False, id='list'), pytest.param(True, id='dict')])
def test_dist_matrix_entries_and_parallel_equality(as_dict: bool):
    p1, p2 = (ARIMA(*params) for params in EXAMPLE2)
    items = [p1, p2, p1, ARIMA(*EXAMPLE1[1])]
    data = dict(zip('abcd', items, strict=True)) if as_dict else items

    serial = dist_matrix_parallel(data, metric=piccolo, tau=100)
    parallel = dist_matrix_parallel(data, metric=piccolo, n_jobs=2, tau=100)
    if as_dict:
        for D in (serial, parallel):
            assert isinstance(D, pd.DataFrame)
            assert list(D.index) == list(D.columns) == list('abcd')
        serial, parallel = serial.to_numpy(), parallel.to_numpy()
    np.testing.assert_array_equal(serial, parallel)

    # EXAMPLE2 shares theta(B) = 1 - 0.5B, so the pi difference is -0.05B(1 - B) / (1 - 0.5B):
    # -0.05 at lag 1 and 0.025 * 0.5^(k-2) at lag k >= 2, with squares summing to 0.0025 + 0.000625 * 4/3 = 1/300
    # (the tail beyond tau=100 is about 1e-60)
    assert serial[1, 0] == serial[2, 1] == pytest.approx(np.sqrt(1 / 300), rel=1e-12)
    assert serial[2, 0] == 0
    np.testing.assert_array_equal(np.diag(serial), 0)
    np.testing.assert_array_equal(serial, serial.T)
    for i in range(4):
        for j in range(i):
            assert serial[i, j] == piccolo(items[i], items[j], tau=100)
