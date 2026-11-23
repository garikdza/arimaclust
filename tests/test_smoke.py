import numpy as np
import pytest

from arimaclust.autoarima.estimation import to_arma_process
from arimaclust.data.examples import EXAMPLE2
from arimaclust.metrics.autoregressive import piccolo


def test_to_arma_process_accepts_lists():
    process = to_arma_process([0.5, -0.2], [0.3])
    np.testing.assert_allclose(process.arcoefs, [0.5, -0.2])
    np.testing.assert_allclose(process.macoefs, [0.3])


def test_piccolo_matches_known_value():
    p1, p2 = (to_arma_process(*params) for params in EXAMPLE2)
    assert piccolo(p1, p1) == 0
    assert piccolo(p1, p2, tau=100) == pytest.approx(0.0577350269, rel=1e-6)
