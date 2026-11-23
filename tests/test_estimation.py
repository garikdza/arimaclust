import numpy as np
import pytest

from arimaclust.autoarima.estimation import estimate_processes, to_arma_process


def test_to_arma_process_negates_ar_params_only():
    process = to_arma_process([0.5, -0.2], [0.3])
    # AR params enter the lag polynomial negated, 1 - 0.5B + 0.2B^2; MA params keep their sign, 1 + 0.3B
    np.testing.assert_allclose(process.ar, [1, -0.5, 0.2])
    np.testing.assert_allclose(process.ma, [1, 0.3])
    np.testing.assert_allclose(process.arcoefs, [0.5, -0.2])
    np.testing.assert_allclose(process.macoefs, [0.3])


def test_estimate_processes_rejects_unknown_process():
    with pytest.raises(NotImplementedError, match='not implemented'):
        estimate_processes([np.zeros(10)], underlying_process='GARCH')
